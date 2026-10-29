"""Выгрузка весов для страницы web/: model.json с описанием слоёв и
weights.bin с числами float32. Свёртки переводятся в раскладку TensorFlow.js
[kh, kw, in, out], полносвязные слои — в [in, out].

    .venv/Scripts/python rl/export.py rl/runs/run1/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--out", default=str(HERE / "web" / "model"))
    args = p.parse_args()

    state = torch.load(args.model, map_location="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers, chunks, offset = [], [], 0
    for name, tensor in state.items():
        w = tensor.numpy().astype("<f4")
        if w.ndim == 4:
            w = w.transpose(2, 3, 1, 0)
        elif w.ndim == 2:
            w = w.T
        w = np.ascontiguousarray(w)
        layers.append({"name": name, "shape": list(w.shape), "offset": offset})
        chunks.append(w.tobytes())
        offset += w.size
    (out / "weights.bin").write_bytes(b"".join(chunks))
    (out / "model.json").write_text(json.dumps({"source": Path(args.model).as_posix(), "layers": layers}, indent=1))
    print(f"{len(layers)} тензоров, {offset} чисел -> {out}")


if __name__ == "__main__":
    main()
