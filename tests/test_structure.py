"""Repository structuurtests voor Drone Detector project."""

from pathlib import Path


REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/data",
    "src/models",
    "src/utils",
    "reports",
    "reports/figures",
    "models",
    "tests",
    "configs",
]

REQUIRED_FILES = [
    "configs/config.yaml",
    "src/data/dataset_loader.py",
    "src/data/preprocess.py",
    "src/models/train_model.py",
    "src/models/evaluate_model.py",
    "src/models/predict.py",
    "src/utils/visualize.py",
    "src/data/__init__.py",
    "src/models/__init__.py",
    "src/utils/__init__.py",
    "tests/test_structure.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "reports/figures/.gitkeep",
    "notebooks/01_data_exploration.ipynb",
]


def test_required_directories_exist():
    """Ensure all core starter directories are present."""
    repo_root = Path(__file__).resolve().parents[1]
    missing = [d for d in REQUIRED_DIRS if not (repo_root / d).is_dir()]
    assert not missing, f"Missing required directories: {missing}"


def test_required_files_exist():
    """Controleer dat alle verplichte projectbestanden aanwezig zijn."""
    repo_root = Path(__file__).resolve().parents[1]
    missing = [f for f in REQUIRED_FILES if not (repo_root / f).is_file()]
    assert not missing, f"Missing required files: {missing}"
