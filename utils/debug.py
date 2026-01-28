# utils/debug.py

import torch
import numpy as np

def print_info(x, prefix=""):
    if torch.is_tensor(x):
        print(f"{prefix}torch.Tensor | shape={tuple(x.shape)} | dtype={x.dtype} | device={x.device}")
    elif isinstance(x, np.ndarray):
        print(f"{prefix}np.ndarray | shape={x.shape} | dtype={x.dtype}")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}{type(x).__name__} | len={len(x)}")
        for i, v in enumerate(x):
            print_info(v, prefix=f"{prefix}  [{i}] ")
    else:
        print(f"{prefix}{type(x)}")
    print()
