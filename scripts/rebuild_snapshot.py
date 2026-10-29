"""Rebuild bundled segment labels without downloads, credentials, or changing snapshot dates."""

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vehicle_rules import segment_vehicles  # noqa: E402


def rebuild(directory: Path) -> None:
    source = sorted(directory.glob("epa_vehicles_*.csv"))[-1]
    vehicles, _, stats = segment_vehicles(pd.read_csv(source, low_memory=False))
    vehicles.to_csv(source, index=False)
    (directory / "segment_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    metadata_path = sorted(directory.glob("metadata_*.json"))[-1]
    meta = json.loads(metadata_path.read_text())
    meta["datasets"]["vehicles"] = {"rows": len(vehicles), "columns": len(vehicles.columns)}
    meta["datasets"].pop("sales", None)
    meta["provenance"] = {
        "vehicles": "EPA bulk export; optional OpenEV enrichment (may use cross-year or fuzzy matches)",
        "stations": "AFDC public charging-station snapshot; not live availability",
        "sales": "not_shipped; former unused sales snapshot was synthetic",
        "normalization": "bundled model-name lookup plus deterministic regex fallback",
    }
    meta["segmentation"] = {
        "method": "rule-based quantiles on latest-year base-model averages",
        "cohort_min_year": 2022,
        "efficiency_quantiles": [0.33, 0.67],
        "range_quantile": 0.67,
    }
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    rebuild(ROOT / "outputs/processed_data")
