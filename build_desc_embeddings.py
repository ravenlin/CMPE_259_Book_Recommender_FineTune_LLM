# build_desc_embeddings.py
#
# Build text-based description embeddings for all books in:
#   data/processed/books_with_genres.parquet
#
# It will:
#   - Construct a text string per book: title + author names + (optional) description
#   - Encode with a sentence-transformers model
#   - Save embeddings to: models/desc_embeddings.npy
#
# Assumes:
#   - books_with_genres.parquet has columns: book_id, title, maybe authors / authors_text / author /
#     description, etc.
#   - If authors.parquet exists, it maps author_id -> author_name and we use that to resolve IDs.

from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path
import ast
import numpy as np
import pandas as pd



from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

BOOKS_PATH = PROC_DIR / "books_with_genres.parquet"
AUTHORS_PATH = PROC_DIR / "authors.parquet"
EMB_PATH = MODEL_DIR / "desc_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


# -----------------------------
# Loading
# -----------------------------

def load_books() -> pd.DataFrame:
    print(f"[load] Loading books from {BOOKS_PATH}")
    df = pd.read_parquet(BOOKS_PATH)
    df["book_id"] = df["book_id"].astype(int)
    print("[info] books shape:", df.shape)
    return df


def load_authors_mapping() -> dict[int, str]:
    """
    Load authors.parquet (if present) and build a mapping author_id -> author_name.

    We try to infer the id and name columns from typical keys:
      id_col in ['author_id', 'id']
      name_col in ['name', 'author_name']
    """
    if not AUTHORS_PATH.exists():
        print(f"[warn] {AUTHORS_PATH} not found; will fall back to any author text in books.")
        return {}

    print(f"[load] Loading authors from {AUTHORS_PATH}")
    df_auth = pd.read_parquet(AUTHORS_PATH)
    print("[info] authors shape:", df_auth.shape)

    cols = list(df_auth.columns)

    id_col = None
    for c in ["author_id", "id"]:
        if c in cols:
            id_col = c
            break

    name_col = None
    for c in ["name", "author_name"]:
        if c in cols:
            name_col = c
            break

    if id_col is None or name_col is None:
        print(
            f"[warn] Could not infer id/name columns in authors.parquet. "
            f"Columns: {cols}. Will not use authors.parquet."
        )
        return {}

    df_auth = df_auth[[id_col, name_col]].dropna()
    df_auth[id_col] = df_auth[id_col].astype(int)
    df_auth[name_col] = df_auth[name_col].astype(str)

    mapping = dict(zip(df_auth[id_col].tolist(), df_auth[name_col].tolist()))
    print(f"[info] built author_id→name mapping of size {len(mapping)}")
    return mapping


# -----------------------------
# Author enrichment
# -----------------------------

def parse_authors_value(val) -> list[int]:
    """
    Try to parse the 'authors' value into a list of integer author IDs.

    Handles:
      - list/tuple/ndarray of ints or strings
      - stringified list like "[123, 456]"
      - comma-separated string like "123,456"
      - single id (string or int)
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        raw_list = list(val)
    elif isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # Try to literal_eval a list-like string
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                raw_list = list(parsed)
            else:
                raw_list = [parsed]
        except Exception:
            # Fallback: maybe comma-separated
            if "," in s:
                raw_list = [x.strip() for x in s.split(",") if x.strip()]
            else:
                raw_list = [s]
    else:
        # int, float, etc.
        raw_list = [val]

    ids: list[int] = []
    for item in raw_list:
        try:
            aid = int(item)
        except Exception:
            continue
        ids.append(aid)
    return ids


def attach_author_names(df_books: pd.DataFrame, author_map: dict[int, str]) -> pd.DataFrame:
    """
    Add an 'author_names' column to df_books:

      - Prefer resolving df_books['authors'] (list of IDs) via author_map.
      - Fall back to 'authors_text' or 'author' if those exist.
      - If everything fails, leave author_names empty string.
    """
    df = df_books.copy()

    if author_map and "authors" in df.columns:
        print("[authors] Resolving author IDs via authors.parquet → author_names...")

        def _ids_to_names(val) -> str:
            ids = parse_authors_value(val)
            names: list[str] = []
            for aid in ids:
                name = author_map.get(aid)
                if name and name not in names:
                    names.append(str(name))
            return ", ".join(names)

        df["author_names"] = df["authors"].apply(_ids_to_names)

    else:
        # No usable author_map+authors; fall back to any text columns on books
        fallback_col = None
        for c in ["authors_text", "author"]:
            if c in df.columns:
                fallback_col = c
                break

        if fallback_col is not None:
            print(f"[authors] Using books column '{fallback_col}' as author_names.")
            df["author_names"] = df[fallback_col].astype(str).fillna("")
        else:
            print("[warn] No authors/author text columns found on books; author_names will be empty.")
            df["author_names"] = ""

    # Debug print
    print("[authors] Sample author_names:")
    print(df[["book_id", "title", "author_names"]].head(10))

    return df


# -----------------------------
# Text construction
# -----------------------------

def make_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "").strip()
    author_names = str(row.get("author_names", "") or "").strip()

    # Description
    desc = ""
    for col in ["description", "desc", "summary", "text"]:
        if col in row.index:
            val = row[col]
            if isinstance(val, str) and val.strip():
                desc = val.strip()
                break

    # Build header "Book title: X by Author(s)"
    header = f"Book title: {title}"
    if author_names:
        header += f" by {author_names}"

    if not desc:
        return header

    return f"{header}\n\nDescription: {desc}"


def build_text_corpus(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        texts.append(make_text(row))
    return texts


# -----------------------------
# Main
# -----------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load books and authors mapping
    df_books = load_books()
    author_map = load_authors_mapping()
    df_books = attach_author_names(df_books, author_map)

    # 2) Build text corpus (title + author + description)
    texts = build_text_corpus(df_books)
    print(f"[build] Built text corpus of length {len(texts)}")

    # 3) Load encoder and compute embeddings
    print(f"[model] Loading text encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    all_embs = []
    print("[encode] Encoding texts in batches...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        embs = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # we'll normalize later in the teacher
        )
        all_embs.append(embs.astype(np.float32))

    desc_embeddings = np.vstack(all_embs)
    print("[info] desc_embeddings shape:", desc_embeddings.shape)

    np.save(EMB_PATH, desc_embeddings)
    print(f"[done] Saved description embeddings to {EMB_PATH}")


if __name__ == "__main__":
    main()
