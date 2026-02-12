
import pandas as pd

CANON = {
    "text": ["text"],
    "label": ["label"],
    "task": ["task"],
    "variety_name": ["variety", "variety_name"],
    "source_name": ["source", "source_name"],
}

# Fix A: Canonical Source Mapping
SOURCE_MAP = {
    "reddit": "Reddit",
    "google": "Google",
    "twitter": "Twitter",
    "youtube": "YouTube"
}

def fix_dashes(s: str) -> str:
    """Normalize weird unicode dashes (en-dash, em-dash) to standard hyphen."""
    if not isinstance(s, str): return str(s)
    return s.replace(u'\u2013', '-').replace(u'\u2014', '-').replace(u'\u00AD', '-')

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    found = {}
    for std, aliases in CANON.items():
        for a in aliases:
            if a in lower_map:
                found[std] = lower_map[a]
                break

    missing = [k for k in CANON.keys() if k not in found]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found mapping={found}.")

    df = df.rename(columns={found[k]: k for k in found})

    # --- CANONICAL VALUE NORMALIZATION ---
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    
    # Force task to lowercase (sentiment/sarcasm)
    df["task"] = df["task"].astype(str).str.strip().str.lower()
    
    # Fix variety names (unicode dashes)
    df["variety_name"] = df["variety_name"].apply(fix_dashes).str.strip()
    
    # Fix source names (Map 'reddit' -> 'Reddit')
    df["source_name"] = df["source_name"].astype(str).str.strip()
    df["source_name"] = df["source_name"].apply(lambda x: SOURCE_MAP.get(x.lower(), x))

    uniq = sorted(df["label"].unique().tolist())
    if set(uniq) - {0, 1}:
        raise ValueError(f"Label values are {uniq} (expected binary 0/1).")

    return df
