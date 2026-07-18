"""
SupplyShield — Environment Verification Script

Checks that all dependencies are installed and the project structure is correct.
Run this after Phase 1 setup to verify everything works.

Usage:
    python verify_setup.py
"""

import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("flwr", "Flower (Federated Learning)"),
    ("shap", "SHAP (Explainability)"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("dotenv", "python-dotenv"),
    ("httpx", "HTTPX"),
    ("tqdm", "tqdm"),
    ("groq", "Groq"),
    ("pydantic", "Pydantic"),
]

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/partitions/novamart",
    "data/partitions/titanelec",
    "data/partitions/swiftlog",
    "models",
    "training",
    "fl",
    "explainability",
    "guidance",
    "api",
    "dashboard",
    "evaluation",
    "notebooks",
    "paper",
]

REQUIRED_FILES = [
    "config.py",
    "utils.py",
    "requirements.txt",
    "README.md",
    ".env.example",
    ".gitignore",
    "models/__init__.py",
    "models/delay_predictor.py",
    "training/__init__.py",
    "training/preprocess.py",
    "training/dataset.py",
    "fl/__init__.py",
    "explainability/__init__.py",
    "guidance/__init__.py",
    "api/__init__.py",
    "evaluation/__init__.py",
]


def check_packages() -> tuple[list, list]:
    """Check that all required Python packages are importable."""
    passed = []
    failed = []
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            passed.append((display_name, version))
        except ImportError:
            failed.append(display_name)
    return passed, failed


def check_directories(root: Path) -> tuple[list, list]:
    """Check that all required directories exist."""
    passed = []
    failed = []
    for d in REQUIRED_DIRS:
        path = root / d
        if path.is_dir():
            passed.append(d)
        else:
            failed.append(d)
    return passed, failed


def check_files(root: Path) -> tuple[list, list]:
    """Check that all required files exist."""
    passed = []
    failed = []
    for f in REQUIRED_FILES:
        path = root / f
        if path.is_file():
            passed.append(f)
        else:
            failed.append(f)
    return passed, failed


def check_model_sanity() -> bool:
    """Quick sanity check on the model definition."""
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from models.delay_predictor import create_model

        model = create_model(input_dim=15, hidden_dim=64)
        x = torch.randn(2, 15)
        out = model(x)
        assert out.shape == (2,), f"Unexpected output shape: {out.shape}"
        params = model.count_parameters()
        assert params > 0, "Model has no parameters"
        return True
    except Exception as e:
        print(f"  Model sanity check failed: {e}")
        return False


def main():
    root = Path(__file__).resolve().parent
    all_good = True

    print("=" * 60)
    print("  SupplyShield -- Environment Verification")
    print("=" * 60)

    # 1. Check packages
    print("\n[PACKAGES] Python Packages:")
    pkg_passed, pkg_failed = check_packages()
    for name, version in pkg_passed:
        print(f"  [OK] {name} ({version})")
    for name in pkg_failed:
        print(f"  [FAIL] {name} -- NOT INSTALLED")
        all_good = False

    # 2. Check directories
    print("\n[DIRS] Directory Structure:")
    dir_passed, dir_failed = check_directories(root)
    for d in dir_passed:
        print(f"  [OK] {d}/")
    for d in dir_failed:
        print(f"  [FAIL] {d}/ -- MISSING")
        all_good = False

    # 3. Check files
    print("\n[FILES] Required Files:")
    file_passed, file_failed = check_files(root)
    for f in file_passed:
        print(f"  [OK] {f}")
    for f in file_failed:
        print(f"  [FAIL] {f} -- MISSING")
        all_good = False

    # 4. Model sanity check
    print("\n[MODEL] Model Sanity Check:")
    if check_model_sanity():
        print("  [OK] DelayPredictor forward pass OK")
    else:
        print("  [FAIL] DelayPredictor -- FAILED")
        all_good = False

    # 5. PyTorch info
    print("\n[INFO] PyTorch Info:")
    try:
        import torch
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, "mps"):
            print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("  [FAIL] PyTorch not available")

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("  [OK] ALL CHECKS PASSED -- Phase 1 Complete!")
        print("  Next: Phase 2 -- Download dataset and run preprocessing")
    else:
        print("  [WARN] SOME CHECKS FAILED -- See above for details")
        print("  Fix the issues and re-run this script")
    print("=" * 60)

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
