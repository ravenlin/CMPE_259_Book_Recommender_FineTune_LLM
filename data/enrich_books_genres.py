# enrich_books_genres.py
#
# Load books.parquet, infer "real" genres from popular_shelves when possible.
# If no real genre is found, assign exactly ONE of 5 synthetic genres that
# do NOT overlap with the real genre names.
#
# This lets you:
#   - distinguish clearly between inferred genres vs synthetic buckets
#   - still test "I don't like <genre>" and filtering behavior even for books
#     where we couldn't infer a real genre.

from pathlib import Path
import random
import pandas as pd

ROOT = Path(__file__).parent
PROC_DIR = ROOT / "processed"
BOOKS_IN = PROC_DIR / "books.parquet"
BOOKS_OUT = PROC_DIR / "books_with_genres.parquet"

print("[load] Loading books...")
df_books = pd.read_parquet(BOOKS_IN)

# -------------------------------------------------------------------
# 1) Real canonical genres + keyword mapping
# -------------------------------------------------------------------
# These are "real" genre names we might infer from shelves.

REAL_GENRE_KEYWORDS = {
    "fantasy": [
        "fantasy", "urban-fantasy", "epic-fantasy", "high-fantasy",
        "paranormal", "paranormal-romance"
    ],
    "science fiction": [
        "science-fiction", "sci-fi", "scifi", "space-opera", "dystopia",
        "post-apocalyptic", "cyberpunk"
    ],
    "romance": [
        "romance", "contemporary-romance", "historical-romance",
        "chick-lit", "erotica", "new-adult"
    ],
    "mystery": [
        "mystery", "crime", "thriller", "suspense", "detective",
        "cozy-mystery", "noir"
    ],
    "young adult": [
        "young-adult", "ya", "ya-fantasy", "ya-fiction", "teen"
    ],
    "nonfiction": [
        "non-fiction", "nonfiction", "biography", "autobiography",
        "memoir", "self-help", "business", "history", "science",
        "psychology", "philosophy", "politics", "economics", "true-crime"
    ],
    "historical": [
        "historical", "historical-fiction"
    ],
    "horror": [
        "horror", "gothic", "creepy"
    ],
    "classics": [
        "classics", "modern-classics", "literature"
    ],
    "graphic novel": [
        "graphic-novels", "graphic-novel", "comics", "manga"
    ],
    "poetry": [
        "poetry", "poems"
    ],
    "spirituality": [
        "religion", "spirituality", "christian", "christian-fiction",
        "christian-non-fiction"
    ],
}

REAL_GENRES = list(REAL_GENRE_KEYWORDS.keys())

# -------------------------------------------------------------------
# 2) Synthetic genres (no overlap with REAL_GENRES)
# -------------------------------------------------------------------

SYNTH_GENRES = [
    "synthetic_cluster_1",
    "synthetic_cluster_2",
    "synthetic_cluster_3",
    "synthetic_cluster_4",
    "synthetic_cluster_5",
]

# Important: none of these appear in REAL_GENRES, so you can tell them apart.

# Seed for overall reproducibility (we'll still make it per-book deterministic)
GLOBAL_RANDOM_SEED = 12345
random.seed(GLOBAL_RANDOM_SEED)

def infer_real_genres_from_shelves(popular_shelves):
    """
    Try to infer one or more REAL genres from popular_shelves using
    REAL_GENRE_KEYWORDS. Return a list of real genre names (may be empty).

    Robust to:
      - list of shelf names
      - string representation of shelves
      - other object types (falls back to str)
    """
    if popular_shelves is None:
        return []

    # Case 1: already a list of shelf names
    if isinstance(popular_shelves, list):
        shelves_lower = [str(s).lower() for s in popular_shelves]

    # Case 2: some other type (e.g., string repr of list) → just treat as one big string
    else:
        shelves_lower = [str(popular_shelves).lower()]

    found = []

    for genre_name, keywords in REAL_GENRE_KEYWORDS.items():
        for kw in keywords:
            # If any shelf string contains this keyword, count it
            if any(kw in shelf for shelf in shelves_lower):
                found.append(genre_name)
                break  # don't add same genre twice

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for g in found:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def assign_synthetic_genre(book_id):
    """
    Deterministically assign exactly ONE synthetic genre to a book
    using its book_id as a seed. Ensures stable labels across runs.
    """
    rng = random.Random(GLOBAL_RANDOM_SEED + int(book_id))
    return rng.choice(SYNTH_GENRES)


def extract_genres(book_id, popular_shelves):
    """
    Main function:
      1. Try to infer real genres from shelves.
      2. If none found, assign ONE synthetic genre.
    """
    real = infer_real_genres_from_shelves(popular_shelves)
    if real:
        return real

    # No real genre found -> one synthetic genre
    synth = assign_synthetic_genre(book_id)
    return [synth]


def get_main_genre(genres_list):
    """Pick a single main genre for convenience (first in list)."""
    if isinstance(genres_list, list) and genres_list:
        return genres_list[0]
    return None

# -------------------------------------------------------------------
# 3) Apply to DataFrame and save
# -------------------------------------------------------------------

print("[compute] Assigning real or synthetic genres...")

if "book_id" not in df_books.columns:
    raise ValueError("books.parquet must contain a 'book_id' column")

df_books["genres"] = [
    extract_genres(bid, shelves)
    for bid, shelves in zip(df_books["book_id"], df_books["popular_shelves"])
]

df_books["main_genre"] = df_books["genres"].apply(get_main_genre)

print("[info] Sample rows with genres:")
print(df_books[["book_id", "title", "popular_shelves", "genres"]].head(10))

print(f"[save] Saving enriched books to {BOOKS_OUT}")
df_books.to_parquet(BOOKS_OUT, index=False)

print("[done] books_with_genres.parquet updated with real+synthetic genres.")
