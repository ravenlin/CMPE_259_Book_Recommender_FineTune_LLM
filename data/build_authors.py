# build_authors_parquet.py
#
# Build data/processed/authors.parquet so that pseudo_user_teacher.py
# can map author NAMES (e.g. "Stephen King") to author_ids.
#
# This script assumes you have the UCSD Book Graph "book authors" file
# somewhere under data/raw/, typically something like:
#   - goodreads_book_authors.json.gz   (most common)
#   - or goodreads_book_authors.json
#   - or goodreads_book_authors.csv
#
# Adjust RAW_AUTHORS_PATH below to match your actual file.

from pathlib import Path
import pandas as pd

# -----------------------------
# Config – adjust this path for your setup
# -----------------------------

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw"

# Try JSON first (common UCSD format), then fall back to CSV
CANDIDATES = [
    RAW_DIR / "goodreads_book_authors.json.gz",
    RAW_DIR / "goodreads_book_authors.json",
    RAW_DIR / "goodreads_book_authors.csv",
]

ROOT = Path(__file__).parent
PROC_DIR = ROOT / "processed"
AUTHORS_OUT = PROC_DIR / "authors.parquet"

# -----------------------------
# Load raw authors file
# -----------------------------

def find_authors_file():
    for path in CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a goodreads_book_authors file. "
        "Looked for:\n" + "\n".join(str(p) for p in CANDIDATES)
    )

def load_authors_raw(path: Path) -> pd.DataFrame:
    print(f"[load] Loading authors from {path}")
    suffix = path.suffix.lower()
    if suffix in [".gz", ".json"]:
        # UCSD BookGraph authors file is typically JSON lines (.json.gz)
        # with fields: author_id, name, etc.
        df = pd.read_json(path, lines=True)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type for authors: {path}")
    return df

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = find_authors_file()
    df = load_authors_raw(raw_path)

    print("[info] raw authors columns:", df.columns.tolist())
    # We expect at least "author_id" and "name"
    # If your file uses different column names, edit here.
    if "author_id" not in df.columns:
        raise KeyError(
            "Expected column 'author_id' not found in authors file. "
            "Please adjust build_authors_parquet.py to match your schema."
        )
    if "name" not in df.columns:
        # Sometimes it's 'author_name' instead of 'name'
        if "author_name" in df.columns:
            df["name"] = df["author_name"]
        else:
            raise KeyError(
                "Expected column 'name' (or 'author_name') not found in authors file."
            )

    # Keep only the essential columns
    df_authors = df[["author_id", "name"]].copy()

    # Enforce types
    df_authors["author_id"] = df_authors["author_id"].astype(int)
    df_authors["name"] = df_authors["name"].astype(str)

    # Drop duplicates on author_id, keeping first
    before = df_authors.shape[0]
    df_authors = df_authors.drop_duplicates(subset=["author_id"])
    after = df_authors.shape[0]
    print(f"[info] authors deduped on author_id: {before} -> {after}")

    # Save as parquet
    df_authors.to_parquet(AUTHORS_OUT, index=False)
    print(f"[done] Saved authors parquet to {AUTHORS_OUT}")
    print(df_authors.head())

if __name__ == "__main__":
    main()
