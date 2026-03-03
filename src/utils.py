import torch
import numpy as np

from __future__ import annotations

import re
from pathlib import Path




def print_info(x, prefix: str = "") -> None:
    """Recursively print type, shape and dtype information.

    Parameters
    ----------
    x : Any
        The object to inspect.  Can be a tensor, numpy array, list, tuple
        or arbitrary object.
    prefix : str, optional
        A string printed at the beginning of each line to aid nesting.
    """
    print('\n' + '*' * 80)
    if torch.is_tensor(x):
        print(f"{prefix}: torch.Tensor | shape={tuple(x.shape)} | dtype={x.dtype} | device={x.device}")
    elif isinstance(x, np.ndarray):
        print(f"{prefix}: np.ndarray | shape={x.shape} | dtype={x.dtype}")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}: {type(x).__name__} | len={len(x)}")
        for i, v in enumerate(x):
            print_info(v, prefix=f"{prefix}  [{i}] ")
    else:
        print(f"{prefix}: {type(x)}")
    print('*' * 80 + '\n')





_EXPERIMENT_RE = re.compile(r"^experiment_(\d+)$")


def create_next_experiment_dir(checkpoint_dir: Path) -> Path:
    """
    Inspect `checkpoint_dir` for `experiment_{n}` folders and create `experiment_{n+1}`
    where `n` is the largest existing integer. If none exist, create `experiment_0`.

    Returns the created experiment directory path.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_n = -1
    for p in checkpoint_dir.iterdir():
        if not p.is_dir():
            continue
        m = _EXPERIMENT_RE.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))

    next_n = max_n + 1
    exp_dir = checkpoint_dir / f"experiment_{next_n}"
    exp_dir.mkdir(parents=True, exist_ok=False)  # fail if somehow already exists
    return exp_dir