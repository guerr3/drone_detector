"""Basic repository structure tests for the project starter setup."""

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


def test_required_directories_exist():
    """Ensure all core starter directories are present."""
    repo_root = Path(__file__).resolve().parents[1]
    missing = [d for d in REQUIRED_DIRS if not (repo_root / d).is_dir()]
    assert not missing, f"Missing required directories: {missing}"
