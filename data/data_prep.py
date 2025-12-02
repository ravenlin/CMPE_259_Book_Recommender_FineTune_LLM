# data_prep.py
#
# Base data preparation for the UCSD BookGraph project.
#
# Responsibilities:
#   - Load raw books + interactions from data/raw
#   - For books:
#       * Filter to ENGLISH ONLY
#       * Select TOP 200,000 English books by:
#           1) highest ratings_count-like column (popularity)
#           2) then highest average_rating
#       * Implement STREAMING when loading large JSON (.json/.json.gz) to avoid OOM
#       * Preserve author-related columns if present
#   - Restrict interactions to those 200k books
#   - Save:
#       data/processed/books.parquet
#       data/processed/interactions.parquet
#
# Run:
#   /usr/local/bin/python3 data_prep.py
#
# Called by prep_all.py as step 1.
# https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# -----------------------------
# Paths / config
# -----------------------------

ROOT = Path(__file__).parent        # .../Final_Project_Code/data
RAW_DIR = ROOT / "raw"
PROC_DIR = ROOT / "processed"

PROC_DIR.mkdir(parents=True, exist_ok=True)

BOOK_CANDIDATES: List[Path] = [
    RAW_DIR / "goodreads_books.parquet",
    RAW_DIR / "goodreads_books.json.gz",
    RAW_DIR / "goodreads_books.json",
    RAW_DIR / "goodreads_books.csv",
]

INTERACTION_CANDIDATES: List[Path] = [
    RAW_DIR / "goodreads_interactions.parquet",
    RAW_DIR / "goodreads_interactions.json.gz",
    RAW_DIR / "goodreads_interactions.json",
    RAW_DIR / "goodreads_interactions.csv",
]

BOOKS_OUT = PROC_DIR / "books.parquet"
INTERACTIONS_OUT = PROC_DIR / "interactions.parquet"

TOP_BOOKS = 200_000
EN_CODES = ["en", "eng", "en-US", "en-GB", "en-CA", "en-AU"]


# -----------------------------
# Helper: find first existing file
# -----------------------------

def find_first_existing(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the following files were found:\n"
        + "\n".join(str(p) for p in candidates)
    )


# -----------------------------
# Books loader (with JSON streaming)
# -----------------------------

def detect_cols(df_sample: pd.DataFrame) -> Tuple[str, str]:
    """
    Given a small books sample, detect:
      - language column
      - popularity column (ratings_count-ish)
    We assume 'average_rating' exists.
    """
    cols = df_sample.columns.tolist()
    if "average_rating" not in cols:
        raise KeyError(
            "Expected 'average_rating' column in books file for ranking. "
            f"Sample columns: {cols}"
        )

    # Find language column
    possible_lang_cols = [
        c for c in cols
        if c in ["language_code", "language", "original_language", "original_language_code"]
    ]
    if not possible_lang_cols:
        raise KeyError(
            "Could not find a language column in books raw file. "
            "Expected one of: 'language_code', 'language', 'original_language', "
            "'original_language_code'. "
            f"Sample columns: {cols}"
        )
    lang_col = possible_lang_cols[0]

    # Find popularity column
    pop_col = None
    for cand in ["ratings_count", "ratings", "work_ratings_count"]:
        if cand in cols:
            pop_col = cand
            break
    if pop_col is None:
        raise KeyError(
            "Expected a ratings-count column in books file for popularity tie-break. "
            "Tried: 'ratings_count', 'ratings', 'work_ratings_count'. "
            f"Sample columns: {cols}"
        )

    return lang_col, pop_col


def load_books_stream_json(path: Path) -> pd.DataFrame:
    """
    Stream goodreads_books.json(.gz) and keep only the top TOP_BOOKS
    English books by (popularity, average_rating).
    This avoids loading the entire file into memory.
    """
    print(f"[books] streaming JSON from {path}")
    chunks = pd.read_json(path, lines=True, chunksize=100_000)
    top_df: Optional[pd.DataFrame] = None
    lang_col: Optional[str] = None
    pop_col: Optional[str] = None

    for ci, chunk in enumerate(chunks):
        if ci == 0:
            print(f"[books] first chunk shape: {chunk.shape}")
            lang_col, pop_col = detect_cols(chunk)
            print(f"[books] using language column: {lang_col}")
            print(f"[books] using popularity column: {pop_col}")

        # Ensure required columns exist
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

        # Keep only columns we care about, INCLUDING author info if present
        base_keep = {
            "book_id",
            "title",
            "average_rating",
            pop_col,
            lang_col,
            "description",
            "popular_shelves",
            "original_publication_year",
            # author-related columns:
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

        # Sort by popularity first, then rating
        top_df = top_df.sort_values(
            [pop_col, "average_rating"],
            ascending=[False, False],
        )
        # Keep at most 2 * TOP_BOOKS rows while streaming
        if len(top_df) > TOP_BOOKS * 2:
            top_df = top_df.head(TOP_BOOKS)

        if (ci + 1) % 10 == 0:
            print(
                f"[books] processed {(ci + 1) * 100_000} rows, "
                f"current top_df size={len(top_df)}"
            )

    if top_df is None:
        raise RuntimeError("No English books with ratings found while streaming JSON.")

    # Final sort + top N (popularity first, then rating)
    top_df = top_df.sort_values(
        [pop_col, "average_rating"],
        ascending=[False, False],
    )
    n = min(TOP_BOOKS, len(top_df))
    df_sel = top_df.head(n).copy()

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


def load_books_raw() -> pd.DataFrame:
    path = find_first_existing(BOOK_CANDIDATES)
    print(f"[books] loading from {path}")
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        print("[books] raw parquet shape:", df.shape)
        return df
    elif suffix in [".gz", ".json"]:
        # Use streaming loader for large JSON
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
        # Likely came from streaming; already filtered & ranked.
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

    df.loc[:, "average_rating"] = pd.to_numeric(
        df["average_rating"], errors="coerce"
    )
    df.loc[:, pop_col] = pd.to_numeric(
        df[pop_col], errors="coerce"
    ).fillna(0)

    df = df[df["average_rating"].notna()]

    df.loc[:, lang_col] = df[lang_col].fillna("unknown")
    is_en = df[lang_col].isin(EN_CODES)
    df_en = df[is_en].copy()

    print("[books] english-only shape:", df_en.shape)

    # Sort by popularity first, then rating
    df_en = df_en.sort_values(
        [pop_col, "average_rating"],
        ascending=[False, False],
    )

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
# Interactions loader
# -----------------------------

def load_interactions_raw() -> pd.DataFrame:
    path = find_first_existing(INTERACTION_CANDIDATES)
    print(f"[interactions] loading from {path}")
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in [".gz", ".json"]:
        df = pd.read_json(path, lines=True)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported interactions file type: {path}")

    print("[interactions] raw shape:", df.shape)
    return df


def prepare_interactions(df_inter_raw: pd.DataFrame, valid_book_ids: set[int]) -> pd.DataFrame:
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
    df.loc[:, "rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    ).fillna(0).astype(float)
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


# -----------------------------
# Main
# -----------------------------

def main():
    print("=== data_prep.py: base data preparation ===")

    # 1) Load raw books (with streaming for JSON if needed)
    df_books_raw = load_books_raw()

    # 2) Select top 200k English books (if not already done by streaming)
    df_books_sel = select_top_english_books(df_books_raw)

    # 3) Load interactions and restrict to selected books
    df_inter_raw = load_interactions_raw()
    valid_book_ids = set(df_books_sel["book_id"].tolist())
    df_inter = prepare_interactions(df_inter_raw, valid_book_ids)

    # 4) Save processed
    df_books_sel.to_parquet(BOOKS_OUT, index=False)
    df_inter.to_parquet(INTERACTIONS_OUT, index=False)

    print(f"[save] books -> {BOOKS_OUT}")
    print(f"[save] interactions -> {INTERACTIONS_OUT}")
    print("[done] data_prep.py complete.")


if __name__ == "__main__":
    main()
