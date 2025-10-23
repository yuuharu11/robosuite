import sys
import torch
import numpy as np
from pprint import pprint

PATH = sys.argv[1] if len(sys.argv) > 1 else "/work/buffer/cil_each/packnet_masks/pr_0.5/seed_1.pt"
MAX_SHOW = 3  # 表示するサンプル数

def summarize(obj, _depth=0):
    t = type(obj)
    if isinstance(obj, torch.Tensor):
        return f"torch.Tensor shape={tuple(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, np.ndarray):
        return f"np.ndarray shape={obj.shape} dtype={obj.dtype}"
    if isinstance(obj, dict):
        return {k: summarize(v, _depth+1) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return f"{t.__name__} len={len(obj)}; first={summarize(obj[0], _depth+1) if len(obj)>0 else 'empty'}"
    return f"{t.__name__} value={repr(obj)[:200]}"

def main(path):
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load '{path}': {e}")
        return

    print(f"Loaded type: {type(data)}")
    if isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        n = min(len(data), MAX_SHOW)
        for i in range(n):
            print(f"\n--- sample[{i}] summary ---")
            sample = data[i]
            if isinstance(sample, (list, tuple)):
                for j, el in enumerate(sample):
                    print(f" element[{j}]: {summarize(el)}")
            else:
                print(summarize(sample))
        if len(data) > MAX_SHOW:
            print(f"\n... (showing first {MAX_SHOW} samples)")
    elif isinstance(data, dict):
        print("Top-level dict keys:")
        pprint(list(data.keys()))
        # try to show small info per key
        for k in list(data.keys())[:MAX_SHOW]:
            print(f"\nkey='{k}': {summarize(data[k])}")
    else:
        print("Content summary:")
        print(summarize(data))

if __name__ == "__main__":
    main(PATH)