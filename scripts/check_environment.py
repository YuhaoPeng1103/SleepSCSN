"""Check Python, PyTorch, CUDA, and package versions for SleepCP."""

from __future__ import annotations

import shutil
import sys


def main() -> None:
    print(f"python: {sys.version.split()[0]}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"cuda_memory_gb: {props.total_memory / 1024**3:.1f}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"torch: ERROR ({exc})")

    for name in ("numpy", "scipy", "pandas", "matplotlib", "mne"):
        try:
            module = __import__(name)
            print(f"{name}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"{name}: ERROR ({exc})")

    usage = shutil.disk_usage(".")
    print(f"disk_free_gb: {usage.free / 1024**3:.1f}")


if __name__ == "__main__":
    main()
