"""
scripts/p100_environment_setup.py
==================================
AUDIT-PRODUCED, P100-VERIFIED environment bootstrap.

WHY THIS FILE EXISTS
--------------------
The single most severe failure encountered in real Kaggle runs was:
    "CUDA error: no kernel image is available for execution on the device"
Root cause: the newest PyTorch wheels (torch 2.10 / cu128) DROPPED the
compiled kernels for the P100's GPU architecture (sm_60 / compute
capability 6.0). Their binaries only ship kernels for sm_75 and newer.

The fix, verified against the official PyTorch version matrix, is to pin
torch to a build whose cu121 wheel STILL contains sm_60 kernels:
    torch==2.4.1  +  torchvision==0.19.1  (cu121 index)

Run this FIRST in any Kaggle session, then restart the kernel once so the
freshly-installed torch is the one loaded into memory.

USAGE (Kaggle notebook, as the very first code cell):
    import subprocess, sys
    subprocess.check_call([sys.executable,
        "scripts/p100_environment_setup.py"])
    # then: Run -> Restart & Clear Cell Outputs, then run all cells.
"""

import subprocess
import sys


# Exact matched, P100-compatible pin (verified vs pytorch.org previous-versions).
TORCH_SPEC = "torch==2.4.1"
TORCHVISION_SPEC = "torchvision==0.19.1"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"


def _pip(args: list) -> None:
    """Run a pip install command, raising if it fails."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + args)


def install_p100_compatible_torch() -> None:
    """
    Force-install the P100-compatible torch/torchvision pair, replacing
    whatever incompatible newer build is present in the base image.
    --force-reinstall guarantees the swap; --no-deps stops it from dragging
    in a mismatched CUDA runtime.
    """
    print("Installing P100-compatible PyTorch (torch 2.4.1 / cu121)...")
    _pip([
        "--force-reinstall", "--no-deps",
        TORCH_SPEC, TORCHVISION_SPEC,
        "--index-url", CUDA_INDEX_URL,
    ])
    # torch 2.4.1's own runtime deps (installed above with --no-deps).
    _pip(["filelock", "typing-extensions", "sympy", "networkx",
          "jinja2", "fsspec", "numpy<2.0"])


def install_project_dependencies() -> None:
    """
    Install the project's helper packages WITHOUT letting any of them
    replace the pinned torch (that is what caused the original failure).
    """
    print("Installing project dependencies (torch protected)...")
    for pkg in ["lpips>=0.1.4", "albumentations>=1.3.0", "qudida"]:
        _pip(["--no-deps", pkg])
    for pkg in ["noise==1.2.2", "fastapi>=0.104.0", "uvicorn[standard]>=0.24.0",
                "python-multipart>=0.0.6", "opencv-python-headless",
                "scikit-image", "scipy", "PyYAML", "pandas", "tqdm",
                "Pillow", "kaggle"]:
        _pip([pkg])


def verify_gpu() -> bool:
    """
    Run a REAL GPU operation and force synchronisation so any kernel
    incompatibility surfaces here (loudly) instead of 40 minutes later
    inside the training loop. Returns True on success.
    """
    import torch
    print(f"torch: {torch.__version__} | CUDA: {torch.version.cuda} "
          f"| available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  No GPU detected. In Kaggle: Session options -> "
              "Accelerator -> GPU P100.")
        return False

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | compute capability: {cap}")

    try:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        c = a @ b
        torch.cuda.synchronize()   # force the kernel to execute NOW
        print(f"GPU compute check PASSED: {tuple(c.shape)}")
        return True
    except Exception as e:
        print(f"GPU compute check FAILED: {e!r}")
        print("  This means the installed torch lacks kernels for this GPU.")
        print("  Confirm Accelerator = GPU P100 and re-run this script.")
        return False


def main() -> None:
    install_p100_compatible_torch()
    install_project_dependencies()
    print("\n--- Verifying GPU ---")
    ok = verify_gpu()
    print("\n" + "=" * 56)
    if ok:
        print("ENVIRONMENT READY. Now: Run -> Restart & Clear Cell Outputs,")
        print("then run all cells. (Restart loads the freshly-pinned torch.)")
    else:
        print("ENVIRONMENT NOT READY — see messages above.")
    print("=" * 56)


if __name__ == "__main__":
    main()
