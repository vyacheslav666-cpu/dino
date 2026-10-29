"""Проверка обученной сети на новых трассах, жадная политика без исследования.

    .venv/Scripts/python rl/play.py runs/run1/best.pt --episodes 16 --cap 0
"""
import argparse
import sys
import time

import numpy as np
import torch

from dino_env import N_ACTIONS
from model import QNet
from train import evaluate


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--cap", type=int, default=0, help="остановить партию на стольких очках (0 — без предела)")
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QNet(N_ACTIONS).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    start = time.time()
    scores = evaluate(net, device, args.episodes, args.cap, args.seed)
    print(f"{args.episodes} партий за {time.time() - start:.0f} с")
    print("очки:", sorted(scores.astype(int).tolist()))
    print(f"среднее {scores.mean():.0f}, медиана {np.median(scores):.0f}, худшая {scores.min():.0f}, лучшая {scores.max():.0f}")


if __name__ == "__main__":
    main()
