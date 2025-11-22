import logging
import os
from datetime import datetime

from .io import ensure_dirs


def setup_logging(name: str = "AION-EWS", log_dir: str = "logs", level: int = logging.INFO) -> str:
    ensure_dirs([log_dir])
    log_path = os.path.join(log_dir, f"run_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log")

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root.addHandler(sh)
    root.addHandler(fh)

    logging.getLogger(__name__).info("Logging initialized: %s -> %s", name, log_path)
    return log_path
