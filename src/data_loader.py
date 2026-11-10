"""Dataset loading.

The raw CSV is not distributed with this repository. See data/README.md for
download instructions.
"""

from pathlib import Path

import pandas as pd

from . import config


def load_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw used-car auction dataset.

    Parameters
    ----------
    csv_path
        Path to ``car_prices.csv``. Defaults to ``data/car_prices.csv``.
    """
    path = Path(csv_path) if csv_path else config.DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download car_prices.csv as described in data/README.md and place "
            "it in the data/ directory, or pass --data /path/to/car_prices.csv."
        )

    df = pd.read_csv(path, low_memory=False)
    required = {
        "year", "make", "body", "transmission", "state", "color",
        "interior", "condition", "odometer", "mmr", "sellingprice",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    return df
