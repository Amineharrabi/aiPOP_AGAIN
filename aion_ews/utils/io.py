import os
from typing import List


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        if p and not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
