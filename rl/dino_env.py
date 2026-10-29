"""Векторная среда: партии игры живут в процессах Node (dino_env.js).

Наблюдение — 4 последних кадра 40×150 в оттенках серого (uint8),
по ним сеть видит и положение, и скорость. Действия: 0 — ничего,
1 — пробел зажат, 2 — «вниз» зажат.
"""
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_ACTIONS = 3
ACTION_NAMES = ["ничего", "прыжок", "вниз"]


class DinoVecEnv:
    # Размер кадра и число кадров на решение заданы в game.js.
    def __init__(self, n_envs=16, procs=4, seed=1, max_score=0, height=40, width=150, stack=4):
        procs = max(1, min(procs, n_envs))
        self.n, self.h, self.w, self.stack = n_envs, height, width, stack
        self.frame_size = height * width
        sizes = [n_envs // procs + (i < n_envs % procs) for i in range(procs)]
        self.workers = []
        for i, size in enumerate(sizes):
            cmd = ["node", str(HERE / "dino_env.js"), "--envs", str(size), "--seed", str(seed * 1000 + i),
                   "--max-score", str(max_score)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE, bufsize=0)
            self.workers.append((proc, size))
        frames = np.concatenate([self._read(p, size * self.frame_size) for p, size in self.workers])
        self.frames = frames.reshape(n_envs, height, width)
        self.obs = np.repeat(self.frames[:, None], stack, axis=1)

    @staticmethod
    def _read(proc, size):
        buf = bytearray()
        while len(buf) < size:
            chunk = proc.stdout.read(size - len(buf))
            if not chunk:
                raise RuntimeError("процесс среды завершился")
            buf += chunk
        return np.frombuffer(bytes(buf), dtype=np.uint8)

    def step(self, actions):
        """Возвращает (obs, done, truncated, score). При done/truncated obs уже
        от новой партии, score — очки законченной партии."""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """Отправить действия; пока Node считает, можно занять видеокарту."""
        actions = np.asarray(actions, dtype=np.uint8)
        start = 0
        for proc, size in self.workers:
            proc.stdin.write(b"S" + actions[start:start + size].tobytes())
            start += size

    def step_wait(self):
        done, trunc, score, frames = [], [], [], []
        for proc, size in self.workers:
            header = self._read(proc, 6 * size).reshape(size, 6)
            done.append(header[:, 0].astype(bool))
            trunc.append(header[:, 1].astype(bool))
            score.append(header[:, 2:6].copy().view("<f4")[:, 0])
            frames.append(self._read(proc, size * self.frame_size))
        done, trunc, score = np.concatenate(done), np.concatenate(trunc), np.concatenate(score)
        self.frames = np.concatenate(frames).reshape(self.n, self.h, self.w)
        self.obs = np.roll(self.obs, -1, axis=1)
        self.obs[:, -1] = self.frames
        ended = done | trunc
        self.obs[ended] = self.frames[ended][:, None]
        return self.obs, done, trunc, score

    def close(self):
        for proc, _ in self.workers:
            try:
                proc.stdin.write(b"Q")
                proc.stdin.close()
            except OSError:
                pass
            proc.wait(timeout=5)
