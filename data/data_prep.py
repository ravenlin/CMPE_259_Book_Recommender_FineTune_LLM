# data_prep.py
#
# Base data preparation for the UCSD BookGraph project.
#
# Responsibilities:
#   - (Optional) Download raw Goodreads data from UCSD
#   - Load raw books + interactions from data/raw
#   - For books:
#       * Filter to ENGLISH ONLY
#       * Select TOP 200,000 English books by:
#           1) highest ratings_count-like column (popularity)
#           2) then highest average_rating
#       * Implement STREAMING when loading large JSON (.json/.json.gz)
#       * Preserve author-related columns if present
#   - For interactions:
#       * STREAM JSON (.json/.json.gz) to avoid OOM
#       * Filter to the selected 200k books while streaming
#       * Keep only is_read=True rows
#   - Save:
#       data/processed/books.parquet
#       data/processed/interactions.parquet
#
# CLI usage:
#   /usr/local/bin/python3 data_prep.py
#
# Notebook usage:
#   from data_prep import download_raw_goodreads, prepare_data, run_pipeline
#   download_raw_goodreads(download=True)
#   prepare_data()
#
# Source:
#   https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import urllib.request

import numpy as np
import pandas as pd


# -----------------------------
# Paths / config
# -----------------------------

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw"
PROC_DIR = ROOT / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

GOODREADS_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
# If True, prefer downloading the large interactions CSV instead of the dedup JSON.
DOWNLOAD_ALL = False

BOOK_CANDIDATES: List[Path] = [
    RAW_DIR / "goodreads_books.parquet",
    RAW_DIR / "goodreads_books.json.gz",
    RAW_DIR / "goodreads_books.json",
    RAW_DIR / "goodreads_books.csv",
]

INTERACTION_CANDIDATES: List[Path] = [
    RAW_DIR / "goodreads_interactions.parquet",
    RAW_DIR / "goodreads_interactions_dedup.json.gz",
    RAW_DIR / "goodreads_interactions.json.gz",
    RAW_DIR / "goodreads_interactions.json",
    RAW_DIR / "goodreads_interactions.csv",
]

AUTHOR_CANDIDATES: List[Path] = [
    RAW_DIR / "goodreads_book_authors.json.gz",
    RAW_DIR / "goodreads_book_authors.json",
    RAW_DIR / "goodreads_book_authors.csv",
]


BOOKS_OUT = PROC_DIR / "books.parquet"
INTERACTIONS_OUT = PROC_DIR / "interactions.parquet"

TOP_BOOKS = 200_000
EN_CODES = ["en", "eng", "en-US", "en-GB", "en-CA", "en-AU"]

INTERACTIONS_MAX_ROWS: int | None = 500_000  # or 500_000, or None for full


# -----------------------------
# Download utilities
# -----------------------------

def _fetch_goodreads_index() -> str:
    with urllib.request.urlopen(GOODREADS_BASE_URL) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_index_filenames(index_html: str) -> Dict[str, str]:
    hrefs = re.findall(r'href="([^"]+)"', index_html, flags=re.IGNORECASE)
    out: Dict[str, str] = {}
    for h in hrefs:
        if h.endswith("/") or h in ("../", "./"):
            continue
        fname = h.split("/")[-1].split("?")[0].split("#")[0]
        if fname:
            out[fname] = fname
    return out


def _download_with_progress(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0

        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)

            if total_bytes:
                pct = 100 * downloaded / total_bytes
                print(f"\r[download] {dest.name}: {pct:6.2f}%", end="")
            else:
                print(f"\r[download] {dest.name}: {downloaded / 1e6:.1f} MB", end="")

    print()
    tmp.replace(dest)


def download_goodreads_files(dest_dir: Path = RAW_DIR, download_all: bool = DOWNLOAD_ALL) -> None:
    index_html = _fetch_goodreads_index()
    available = _parse_index_filenames(index_html)

    # Always grab:
    #   - books
    #   - book_authors (needed by build_authors.py)
    #   - interactions (dedup JSON by default)
    required = [
        "goodreads_books.json.gz",
        "goodreads_book_authors.json.gz",
    ]
    if download_all:
        required.append("goodreads_interactions.csv")
    else:
        required.append("goodreads_interactions_dedup.json.gz")

    for fname in required:
        dest = dest_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[download] already exists: {dest}")
            continue

        if fname not in available:
            raise FileNotFoundError(
                f"[download] {fname} not found on UCSD server at {GOODREADS_BASE_URL}"
            )

        url = GOODREADS_BASE_URL + fname
        print(f"[download] downloading {fname} from {url}")
        _download_with_progress(url, dest)


def download_raw_goodreads(download: bool = False) -> None:
    """
    Public API for notebooks / scripts:
        download_raw_goodreads(download=True)

    If download is False, this is a no-op.
    If True, it only downloads what is missing.
    """
    if not download:
        print("[download] download=False; skipping raw downloads.")
        return

    have_books = any(p.exists() for p in BOOK_CANDIDATES)
    have_inter = any(p.exists() for p in INTERACTION_CANDIDATES)
    have_authors = any(p.exists() for p in AUTHOR_CANDIDATES)

    if have_books and have_inter and have_authors:
        print("[download] raw Goodreads files already present (books, interactions, authors).")
        return

    print("[download] some raw Goodreads files missing; downloading from UCSD...")
    download_goodreads_files(dest_dir=RAW_DIR, download_all=DOWNLOAD_ALL)



# -----------------------------
# Helper functions
# -----------------------------

def find_first_existing(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the following candidate files exist:\n"
        + "\n".join(str(p) for p in candidates)
    )


def detect_cols(df_sample: pd.DataFrame) -> Tuple[str, str]:
    """
    Given a sample of the books DF, detect:
      - language column
      - popularity column (ratings_count-like)
    """
    cols = df_sample.columns.tolist()

    if "average_rating" not in cols:
        raise KeyError(
            "Expected 'average_rating' column in books file for ranking. "
            f"Sample columns: {cols}"
        )

    lang_col = next(
        (
            c for c in cols
            if c in ["language_code", "language", "original_language", "original_language_code"]
        ),
        None,
    )
    if lang_col is None:
        raise KeyError(
            "Could not find a language column in books raw file. "
            "Expected one of: 'language_code', 'language', "
            "'original_language', 'original_language_code'. "
            f"Sample columns: {cols}"
        )

    pop_col = next(
        (c for c in ["ratings_count", "ratings", "work_ratings_count"] if c in cols),
        None,
    )
    if pop_col is None:
        raise KeyError(
            "Expected a ratings-count column in books file for popularity tie-break. "
            "Tried: 'ratings_count', 'ratings', 'work_ratings_count'. "
            f"Sample columns: {cols}"
        )

    return lang_col, pop_col


# -----------------------------
# Books loading
# -----------------------------

def load_books_stream_json(path: Path) -> pd.DataFrame:
    """
    Stream goodreads_books.json(.gz) and keep only the top TOP_BOOKS
    English books by (popularity, average_rating).
    """
    print(f"[books] streaming JSON from {path}")
    chunks = pd.read_json(path, lines=True, chunksize=100_000)

    top_df: Optional[pd.DataFrame] = None
    lang_col: Optional[str] = None
    pop_col: Optional[str] = None

    for i, chunk in enumerate(chunks):
        # Avoid SettingWithCopyWarning
        chunk = chunk.copy()

        if i == 0:
            print(f"[books] first chunk shape: {chunk.shape}")
            lang_col, pop_col = detect_cols(chunk)
            print(f"[books] using language column: {lang_col}")
            print(f"[books] using popularity column: {pop_col}")

        # Basic column checks
        if "book_id" not in chunk.columns or "title" not in chunk.columns:
            raise KeyError(
                "Expected 'book_id' and 'title' columns in books file. "
                f"Chunk columns: {chunk.columns.tolist()}"
            )

        # Clean numeric columns
        chunk.loc[:, "average_rating"] = pd.to_numeric(
            chunk["average_rating"], errors="coerce"
        )
        chunk.loc[:, pop_col] = pd.to_numeric(
            chunk[pop_col], errors="coerce"
        ).fillna(0)

        # Drop rows with no rating
        chunk = chunk[chunk["average_rating"].notna()]

        # Filter to English
        chunk.loc[:, lang_col] = chunk[lang_col].fillna("unknown")
        is_en = chunk[lang_col].isin(EN_CODES)
        chunk_en = chunk[is_en].copy()

        if chunk_en.empty:
            continue

        # Keep only relevant columns (incl. author info if present)
        base_keep = {
            "book_id",
            "title",
            "average_rating",
            pop_col,
            lang_col,
            "description",
            "popular_shelves",
            "original_publication_year",
            "authors",
            "authors_text",
            "author",
        }
        keep_cols = list(base_keep.intersection(chunk_en.columns))
        chunk_en = chunk_en[keep_cols]

        if top_df is None:
            top_df = chunk_en
        else:
            top_df = pd.concat([top_df, chunk_en], ignore_index=True)

        # Sort and truncate buffer
        top_df = top_df.sort_values([pop_col, "average_rating"], ascending=[False, False])
        if len(top_df) > TOP_BOOKS * 2:
            top_df = top_df.head(TOP_BOOKS)

        if (i + 1) % 10 == 0:
            print(
                f"[books] processed {(i + 1) * 100_000} rows, "
                f"current top_df size={len(top_df)}"
            )

    if top_df is None or top_df.empty:
        raise RuntimeError("No English books with ratings found while streaming JSON.")

    # Final sort and select top N
    top_df = top_df.sort_values([pop_col, "average_rating"], ascending=[False, False])
    n = min(TOP_BOOKS, len(top_df))
    df_sel = top_df.head(n).copy()
    df_sel.loc[:, "book_id"] = df_sel["book_id"].astype(int)

    print(
        f"[books] selected top {n} English books "
        f"(mean rating={df_sel['average_rating'].mean():.3f}, "
        f"median {pop_col}={df_sel[pop_col].median():.1f})"
    )

    show_cols = ["book_id", "title", "average_rating", pop_col, lang_col]
    for ac in ["authors", "authors_text", "author"]:
        if ac in df_sel.columns:
            show_cols.append(ac)
            break
    print(df_sel[show_cols].head(10))

    return df_sel


def load_books_raw() -> pd.DataFrame:
    path = find_first_existing(BOOK_CANDIDATES)
    print(f"[books] loading from {path}")
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        print("[books] raw parquet shape:", df.shape)
        return df
    elif suffix in [".gz", ".json"]:
        return load_books_stream_json(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        print("[books] raw csv shape:", df.shape)
        return df
    else:
        raise ValueError(f"Unsupported books file type: {path}")


def select_top_english_books(df_books_raw: pd.DataFrame) -> pd.DataFrame:
    """
    If df_books_raw is already a manageable DataFrame (e.g., from parquet/csv),
    apply the English + top-200k selection here.

    If we came from JSON streaming, df_books_raw is already filtered and top-200k.
    We'll detect that by checking if it's <= TOP_BOOKS rows and has 'average_rating'.
    """
    if len(df_books_raw) <= TOP_BOOKS + 1 and "average_rating" in df_books_raw.columns:
        df = df_books_raw.copy()
        df.loc[:, "book_id"] = df["book_id"].astype(int)
        return df

    df = df_books_raw.copy()

    # Ensure required columns
    if "book_id" not in df.columns:
        raise KeyError(
            "Expected column 'book_id' in books file. "
            f"Found columns: {df.columns.tolist()}"
        )
    if "title" not in df.columns:
        raise KeyError(
            "Expected column 'title' in books file. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Detect cols from sample
    sample = df.head(1000)
    lang_col, pop_col = detect_cols(sample)
    print(f"[books] using language column: {lang_col}")
    print(f"[books] using popularity column: {pop_col}")

    df.loc[:, "average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df.loc[:, pop_col] = pd.to_numeric(df[pop_col], errors="coerce").fillna(0)

    df = df[df["average_rating"].notna()]

    df.loc[:, lang_col] = df[lang_col].fillna("unknown")
    is_en = df[lang_col].isin(EN_CODES)
    df_en = df[is_en].copy()

    print("[books] english-only shape:", df_en.shape)

    # Sort by popularity first, then rating
    df_en = df_en.sort_values([pop_col, "average_rating"], ascending=[False, False])

    n = min(TOP_BOOKS, len(df_en))
    df_sel = df_en.head(n).copy()
    df_sel.loc[:, "book_id"] = df_sel["book_id"].astype(int)

    print(
        f"[books] selected top {n} english books "
        f"(mean rating={df_sel['average_rating'].mean():.3f}, "
        f"median {pop_col}={df_sel[pop_col].median():.1f})"
    )

    # Columns to display for sanity check
    show_cols = ["book_id", "title", "average_rating", pop_col, lang_col]
    # Add any available author-related column
    for ac in ["authors", "authors_text", "author"]:
        if ac in df_sel.columns:
            show_cols.append(ac)
            break

    print(df_sel[show_cols].head(10))
    return df_sel


# -----------------------------
# Interactions (STREAMING JSON)
# -----------------------------

def load_and_prepare_interactions_stream_json(
    path: Path,
    valid_book_ids: set[int],
    max_rows: int | None = INTERACTIONS_MAX_ROWS,
) -> pd.DataFrame:
    """
    Stream goodreads_interactions*.json(.gz) and:
      - enforce types per chunk
      - filter to valid_book_ids
      - keep only is_read == True
      - optionally cap total rows kept at `max_rows`

    NOTE:
      - user_id is kept as a string (Goodreads dedup file uses hashed IDs).
      - book_id is coerced to int safely.
    """
    print(f"[interactions] streaming JSON from {path}")
    chunks = pd.read_json(path, lines=True, chunksize=1_000_000)

    out: list[pd.DataFrame] = []
    required = ["user_id", "book_id"]
    total_kept = 0

    for i, chunk in enumerate(chunks):
        chunk = chunk.copy()

        # Column checks
        for col in required:
            if col not in chunk.columns:
                raise KeyError(
                    f"Expected column '{col}' in interactions file. "
                    f"Found columns: {chunk.columns.tolist()}"
                )

        # rating, is_read, is_reviewed handling per chunk
        if "rating" not in chunk.columns:
            chunk.loc[:, "rating"] = 0.0
        if "is_read" not in chunk.columns:
            # If rating exists, infer from that; otherwise assume read
            chunk.loc[:, "is_read"] = chunk["rating"].fillna(0) > 0
        if "is_reviewed" not in chunk.columns:
            chunk.loc[:, "is_reviewed"] = False

        # --- TYPE COERCIONS ---

        # user_id: keep as string
        chunk.loc[:, "user_id"] = chunk["user_id"].astype(str)

        # book_id: force numeric, drop bad, cast to int
        chunk.loc[:, "book_id"] = pd.to_numeric(chunk["book_id"], errors="coerce")
        chunk = chunk[chunk["book_id"].notna()]
        chunk.loc[:, "book_id"] = chunk["book_id"].astype(int)

        # rating + flags
        chunk.loc[:, "rating"] = pd.to_numeric(chunk["rating"], errors="coerce").fillna(0).astype(float)
        chunk.loc[:, "is_read"] = chunk["is_read"].astype(bool)
        chunk.loc[:, "is_reviewed"] = chunk["is_reviewed"].astype(bool)

        # Filter to valid books + read interactions
        chunk = chunk[chunk["book_id"].isin(valid_book_ids)]
        chunk = chunk[chunk["is_read"]]

        if chunk.empty:
            continue

        out.append(chunk)
        total_kept += len(chunk)

        if (i + 1) % 10 == 0:
            print(
                f"[interactions] processed {(i + 1) * 1_000_000} rows, "
                f"kept so far: {total_kept}"
            )

        # --- EARLY STOP: respect cap ---
        if max_rows is not None and total_kept >= max_rows:
            print(f"[interactions] reached cap of {max_rows} kept rows; stopping early.")
            break

    if not out:
        raise RuntimeError("No interactions found for selected books while streaming JSON.")

    df = pd.concat(out, ignore_index=True)

    # If we overshot slightly, trim exact max_rows
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    print("[interactions] final (typed + filtered) shape:", df.shape)
    print("[interactions] unique users:", df["user_id"].nunique())
    print("[interactions] unique books:", df["book_id"].nunique())
    return df


def prepare_interactions(df_inter_raw: pd.DataFrame, valid_book_ids: set[int]) -> pd.DataFrame:
    """
    Non-streaming path for parquet/csv interactions.
    """
    df = df_inter_raw.copy()

    required = ["user_id", "book_id"]
    for col in required:
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' in interactions file. "
                f"Found columns: {df.columns.tolist()}"
            )

    if "is_read" not in df.columns:
        if "rating" in df.columns:
            df.loc[:, "is_read"] = df["rating"].fillna(0) > 0
        else:
            df.loc[:, "is_read"] = True

    if "rating" not in df.columns:
        df.loc[:, "rating"] = 0.0

    if "is_reviewed" not in df.columns:
        df.loc[:, "is_reviewed"] = False

    df.loc[:, "user_id"] = df["user_id"].astype(int)
    df.loc[:, "book_id"] = df["book_id"].astype(int)
    df.loc[:, "rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(float)
    df.loc[:, "is_read"] = df["is_read"].astype(bool)
    df.loc[:, "is_reviewed"] = df["is_reviewed"].astype(bool)

    print("[interactions] raw (typed) shape:", df.shape)

    df = df[df["book_id"].isin(valid_book_ids)]
    print("[interactions] after restricting to selected books:", df.shape)
    print("[interactions] unique users:", df["user_id"].nunique())
    print("[interactions] unique books:", df["book_id"].nunique())

    df = df[df["is_read"]]
    print("[interactions] after is_read filter:", df.shape)

    return df


def load_interactions_raw(
    valid_book_ids: set[int],
    max_rows: int | None = INTERACTIONS_MAX_ROWS,
) -> pd.DataFrame:
    """
    Wrapper that:
      - finds the interactions file
      - uses streaming for JSON(.gz)
      - uses in-memory prep for parquet/csv
      - optionally caps total rows
    """
    path = find_first_existing(INTERACTION_CANDIDATES)
    print(f"[interactions] loading from {path}")
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return prepare_interactions(df, valid_book_ids)
    elif suffix in [".gz", ".json"]:
        # Use streaming JSON loader to avoid OOM
        return load_and_prepare_interactions_stream_json(path, valid_book_ids, max_rows=max_rows)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        return prepare_interactions(df, valid_book_ids)
    else:
        raise ValueError(f"Unsupported interactions file type: {path}")


# -----------------------------
# Public pipeline API
# -----------------------------

def prepare_data() -> None:
    """
    Run the full data preparation pipeline assuming raw files exist.
    """
    print("=== data_prep.py: base data preparation ===")

    # 1) Books (streaming if JSON)
    df_books_raw = load_books_raw()
    df_books_sel = select_top_english_books(df_books_raw)

    # 2) Interactions (stream JSON, filter by selected book_ids, with cap)
    valid_book_ids = set(df_books_sel["book_id"].tolist())
    df_inter = load_interactions_raw(valid_book_ids, max_rows=INTERACTIONS_MAX_ROWS)

    # 3) Save processed
    df_books_sel.to_parquet(BOOKS_OUT, index=False)
    df_inter.to_parquet(INTERACTIONS_OUT, index=False)

    print(f"[save] books -> {BOOKS_OUT}")
    print(f"[save] interactions -> {INTERACTIONS_OUT}")
    print("[done] data_prep.py complete.")


def run_pipeline(download: bool = False) -> None:
    """
    High-level entry point for scripts / notebooks.

    Parameters
    ----------
    download : bool
        If True, download missing raw Goodreads files before running prep.
    """
    download_raw_goodreads(download=download)
    prepare_data()


# -----------------------------
# CLI entry point
# -----------------------------

def main():
    # For CLI: default to NO download (so we don't surprise people)
    run_pipeline(download=False)


if __name__ == "__main__":
    main()
