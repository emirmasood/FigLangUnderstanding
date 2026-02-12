
from pathlib import Path
import pandas as pd
import shutil

def load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def safe_name(s: str) -> str:
    # Aggressive filename safety: lowercase, no spaces, standard hyphens
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    # minimal fallback for weird chars
    s = "".join([c if c.isalnum() or c in "-_." else "" for c in s])
    return s

def clean_dir(path: Path):
    """Safely delete and recreate a directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
