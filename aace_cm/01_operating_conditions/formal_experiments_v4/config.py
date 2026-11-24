"""Package-local paths and settings for the Exp1 scripts."""

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    vw_features_dir: Path = ROOT_DIR / "aace_cm" / "02-features"
    vw_labels_dir: Path = ROOT_DIR / "aace_cm" / "03-semantic_labels"
    vw_labels_old_dir: Path = ROOT_DIR / "03-semantic_labels-old"
    outputs_dir: Path = ROOT_DIR / "formal_experiments_v4" / "outputs"
    figures_dir: Path = ROOT_DIR / "formal_experiments_v4" / "figures"
    exp0_recommended_dates_csv: Path = (
        outputs_dir / "exp0_data_audit" / "recommended_dates.csv"
    )
    exp1_semantic_dir: Path = outputs_dir / "exp1_semantic"
    exp1_figures_dir: Path = outputs_dir / "exp1_figures"

    def ensure_dirs(self) -> None:
        for directory in (self.exp1_semantic_dir, self.exp1_figures_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CommonConfig:
    device: str = "BA1"
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class Exp1Config:
    feature_cols: tuple[str, ...] = ("BA1_X", "BA1_Y", "BA1_Z")
    fast_num_days: int = 8
    figure_random_seed: int = 2026


PATHS = Paths()
COMMON = CommonConfig()
EXP1 = Exp1Config()


def get_exp0_recommended_dates(path: Path | None = None) -> list[str]:
    """Read eligible dates in the order of the supplied audit manifest."""
    csv_path = path or PATHS.exp0_recommended_dates_csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"Exp0 date manifest is required for exp0_recommended: {csv_path}")

    dates = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            date = (row.get("date") or "").strip()
            eligible = (row.get("recommended_for_both_devices") or "True").strip().lower()
            if date and eligible in {"true", "1", "yes", "y"}:
                dates.append(date)
    return dates
