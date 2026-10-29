"""DQN на пикселях для динозавра из Chromium.

Что внутри: свёрточная сеть по 4 последним кадрам, Double DQN,
dueling-головы, n-шаговые возвраты, буфер воспроизведения на сотни тысяч
переходов прямо в видеопамяти, ε-жадное исследование.
Награда: +0.1 за каждый шаг, −1 за смерть.

    .venv/Scripts/python rl/train.py --name run1
"""
import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dino_env import N_ACTIONS, DinoVecEnv
from model import QNet

HERE = Path(__file__).resolve().parent


class Replay:
    """Кольцевой буфер по каждой партии отдельно, целиком в памяти устройства.
    Хранится только последний кадр наблюдения, стопка из 4 кадров
    собирается при выборке."""

    def __init__(self, n_envs, capacity, height, width, stack, n_step, gamma, device):
        self.n_envs, self.cap, self.stack, self.n_step, self.gamma = n_envs, capacity, stack, n_step, gamma
        self.device = device
        z = lambda dtype, *shape: torch.zeros((n_envs, capacity, *shape), dtype=dtype, device=device)
        self.frames = z(torch.uint8, height, width)
        self.actions = z(torch.int64)
        self.rewards = z(torch.float32)
        self.dones = z(torch.bool)
        self.truncs = z(torch.bool)
        self.ep_pos = z(torch.int64)  # номер шага внутри партии
        self.ptr = self.size = 0

    def add(self, frames, ep_pos, actions, rewards, dones, truncs):
        i = self.ptr
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device, non_blocking=True)
        self.frames[:, i], self.ep_pos[:, i], self.actions[:, i] = t(frames), t(ep_pos), t(actions)
        self.rewards[:, i], self.dones[:, i], self.truncs[:, i] = t(rewards), t(dones), t(truncs)
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def _stack(self, e, idx):
        # Кадры до начала партии подменяются её первым кадром — как в среде.
        offsets = torch.arange(self.stack - 1, -1, -1, device=self.device)[None]
        back = torch.minimum(offsets, self.ep_pos[e, idx][:, None])
        return self.frames[e[:, None], (idx[:, None] - back) % self.cap]

    def sample(self, batch):
        pos = torch.randint(self.stack - 1, self.size - self.n_step, (batch,), device=self.device)
        idx = (self.ptr - self.size + pos) % self.cap
        e = torch.randint(0, self.n_envs, (batch,), device=self.device)
        ret = torch.zeros(batch, device=self.device)
        alive = torch.ones(batch, dtype=torch.bool, device=self.device)
        valid = torch.ones(batch, dtype=torch.bool, device=self.device)
        for k in range(self.n_step):
            j = (idx + k) % self.cap
            ret += alive * self.gamma ** k * self.rewards[e, j]
            # Партию обрезали по лимиту очков: следующего кадра нет, такой пример пропускаем.
            valid &= ~(alive & self.truncs[e, j])
            alive &= ~(self.dones[e, j] | self.truncs[e, j])
        return (self._stack(e, idx), self.actions[e, idx], ret,
                self._stack(e, (idx + self.n_step) % self.cap), alive * self.gamma ** self.n_step, valid.float())


def evaluate(net, device, episodes, max_score, seed):
    """Жадная политика на новых трассах. Возвращает очки партий."""
    env = DinoVecEnv(n_envs=episodes, procs=min(4, episodes), seed=seed, max_score=max_score)
    scores = np.full(episodes, np.nan)
    try:
        obs = env.obs
        while np.isnan(scores).any():
            with torch.no_grad():
                actions = net(torch.from_numpy(obs).to(device)).argmax(1).cpu().numpy()
            obs, done, trunc, score = env.step(actions)
            first = (done | trunc) & np.isnan(scores)
            scores[first] = score[first]
    finally:
        env.close()
    return scores


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="dqn")
    p.add_argument("--steps", type=int, default=3_000_000, help="шагов среды всего (по всем партиям)")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--procs", type=int, default=8, help="процессов Node")
    p.add_argument("--buffer", type=int, default=400_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--learn-start", type=int, default=20_000)
    p.add_argument("--updates", type=int, default=2, help="шагов градиента на шаг всех партий")
    p.add_argument("--target-every", type=int, default=2000, help="шагов градиента между копиями целевой сети")
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-steps", type=int, default=500_000)
    p.add_argument("--train-cap", type=int, default=5000, help="обрезать учебную партию на стольких очках")
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--eval-cap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    out = HERE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    env = DinoVecEnv(n_envs=args.envs, procs=args.procs, seed=args.seed, max_score=args.train_cap)
    net = QNet(N_ACTIONS).to(device)
    target = QNet(N_ACTIONS).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1.5e-4)
    replay = Replay(args.envs, args.buffer // args.envs, env.h, env.w, env.stack, args.n_step, args.gamma, device)

    def update():
        o, a, r, o2, disc, valid = replay.sample(args.batch)
        q = net(o).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: действие выбирает текущая сеть, оценивает целевая.
            best = net(o2).argmax(1, keepdim=True)
            y = r + disc * target(o2).gather(1, best).squeeze(1)
        loss = (F.smooth_l1_loss(q, y, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        return torch.stack([loss.detach(), q.detach().mean()])

    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["env_steps", "seconds", "eps", "episodes", "score_avg100", "score_max100", "loss", "q_mean", "eval_mean", "eval_min"])

    obs = env.obs.copy()
    ep_pos = np.zeros(args.envs, np.int64)
    recent = deque(maxlen=100)
    episodes, grad_steps = 0, 0
    stats = torch.zeros(2, device=device)  # сумма loss и Q с последнего отчёта
    stats_n = 0
    best_eval, next_eval = -1.0, args.eval_every
    start = last_print = time.time()

    env_steps = 0
    while env_steps < args.steps:
        eps = max(args.eps_end, 1 - (1 - args.eps_end) * env_steps / args.eps_steps)
        with torch.no_grad():
            actions = net(torch.from_numpy(obs).to(device)).argmax(1).cpu().numpy()
        explore = rng.random(args.envs) < eps
        actions[explore] = rng.integers(0, N_ACTIONS, explore.sum())

        # Пока Node двигает партии, видеокарта учит сеть на старых переходах.
        env.step_async(actions)
        if env_steps >= args.learn_start:
            for _ in range(args.updates):
                stats += update()
                stats_n += 1
                grad_steps += 1
                if grad_steps % args.target_every == 0:
                    target.load_state_dict(net.state_dict())
        next_obs, done, trunc, score = env.step_wait()

        replay.add(obs[:, -1], ep_pos, actions, np.where(done, -1.0, 0.1).astype(np.float32), done, trunc)
        ended = done | trunc
        ep_pos = np.where(ended, 0, ep_pos + 1)
        obs = next_obs.copy()
        env_steps += args.envs
        for s in score[ended]:
            recent.append(float(s))
            episodes += 1

        eval_mean = eval_min = ""
        if env_steps >= next_eval:
            next_eval += args.eval_every
            scores = evaluate(net, device, args.eval_episodes, args.eval_cap, seed=10_000 + env_steps)
            eval_mean, eval_min = float(scores.mean()), float(scores.min())
            print(f"  проверка: в среднем {eval_mean:.0f}, худшая {eval_min:.0f}, все: {scores.astype(int).tolist()}", flush=True)
            # Проверка упирается в предел очков, и равных результатов много.
            # При равенстве берём более позднюю сеть: она дольше училась.
            if eval_mean >= best_eval:
                best_eval = eval_mean
                torch.save(net.state_dict(), out / "best.pt")
            torch.save(net.state_dict(), out / "last.pt")

        now = time.time()
        if now - last_print > 30 or eval_mean != "":
            last_print = now
            avg = np.mean(recent) if recent else 0.0
            mx = np.max(recent) if recent else 0.0
            loss_avg, q_avg = (stats / max(stats_n, 1)).tolist()
            stats.zero_()
            stats_n = 0
            writer.writerow([env_steps, round(now - start), round(eps, 3), episodes, round(avg, 1), mx, round(loss_avg, 4), round(q_avg, 3), eval_mean, eval_min])
            log.flush()
            print(f"{env_steps:>9} шагов  {env_steps / (now - start):5.0f}/с  ε={eps:.3f}  партий {episodes:>6}  "
                  f"очки (100 посл.) ср {avg:6.0f} макс {mx:5.0f}  loss {loss_avg:.4f}  Q {q_avg:.2f}", flush=True)

    torch.save(net.state_dict(), out / "last.pt")
    env.close()
    log.close()


if __name__ == "__main__":
    main()
