"""Central experiment settings so every script and notebook shares them."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "figures"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class Settings:
    seed: int = 7
    test_fraction: float = 0.25
    cv_folds: int = 5
    cv_repeats: int = 3
    search_iterations: int = 25
