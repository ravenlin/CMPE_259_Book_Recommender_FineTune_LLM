# generate_llm_data.py
#
# Generate synthetic (prompt, response) pairs for LLM fine-tuning, using:
#   - teacher_recommender.recommend
#   - teacher_recommender.recommend_avoid_genres
#
# Output format: JSONL file where each line is:
#   {"prompt": "...", "response": "..."}

import json
from pathlib import Path
import random

import pandas as pd

from teacher_recommender import (
    PROC_DIR,
    df_books,
    df_interactions,
    recommend,
    recommend_avoid_genres,
)

OUT_DIR = PROC_DIR / "llm_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASIC_OUT = OUT_DIR / "train_basic_recs.jsonl"
DISLIKE_GENRE_OUT = OUT_DIR / "train_dislike_genre.jsonl"

# -----------------------------
# Helpers
# -----------------------------

df_books_indexed = df_books.set_index("book_id")

def get_book_title(book_id: int) -> str:
    try:
        return str(df_books_indexed.loc[book_id, "title"])
    except KeyError:
        return f"Book #{book_id}"

def get_book_authors(book_id: int) -> str:
    # You only have author_ids right now; we'll just return placeholder.
    # Later you can join to book_authors to get real names.
    return "Unknown author"

def build_book_description(book_id: int) -> str:
    """Text snippet LLM can see for a rec item (title + maybe rating/genre)."""
    row = df_books_indexed.loc[book_id]
    title = str(row["title"])
    avg = row.get("average_rating", None)
    genres = row.get("genres", [])
    parts = [title]
    if avg is not None:
        parts.append(f"(avg rating {avg:.2f})")
    if isinstance(genres, list) and genres:
        parts.append(f"[genres: {', '.join(genres)}]")
    return " ".join(parts)

def format_recommendation_list(book_ids):
    """Turn list of book_ids into a human-readable numbered list."""
    lines = []
    for i, bid in enumerate(book_ids, start=1):
        desc = build_book_description(bid)
        lines.append(f"{i}. {desc}")
    return "\n".join(lines)

# -----------------------------
# 1) Basic recs: "I liked X, recommend 5 similar books"
# -----------------------------

def generate_basic_examples(n_examples: int = 2000, k: int = 5):
    """
    For each example:
      - Sample a popular book from interactions
      - Ask teacher for top-k similar
      - Build (prompt, response)
    """
    # Use books that appear a lot in interactions as seeds
    book_counts = df_interactions["book_id"].value_counts()
    popular_book_ids = list(book_counts.index[:50_000])  # adjust as you like

    with BASIC_OUT.open("w", encoding="utf-8") as f:
        made = 0
        attempts = 0
        max_attempts = n_examples * 10

        while made < n_examples and attempts < max_attempts:
            attempts += 1

            seed_id = int(random.choice(popular_book_ids))
            seed_title = get_book_title(seed_id)

            rec_ids = recommend(seed_id, top_k=k)
            if len(rec_ids) < k:
                continue  # skip seeds with too few co-occurrence neighbours

            prompt = (
                f"I really enjoyed the book '{seed_title}'. "
                f"Please recommend {k} similar books I might like. "
                f"Return them as a numbered list with one book per line."
            )

            rec_list_text = format_recommendation_list(rec_ids)

            response = (
                f"Here are {k} books you might like based on '{seed_title}':\n\n"
                f"{rec_list_text}"
            )

            record = {"prompt": prompt, "response": response}
            f.write(json.dumps(record) + "\n")
            made += 1

            if made % 100 == 0:
                print(f"[basic] generated {made}/{n_examples} examples")

        print(f"[basic] Done. Generated {made} examples, wrote to {BASIC_OUT}")


# -----------------------------
# 2) Dislike genre: "I don't like X genre"
# -----------------------------

# Hard-code a small set of genre words you expect users to mention
USER_FACING_GENRES = ["fantasy", "science fiction", "romance", "mystery", "young adult"]

def pick_genre_for_book(book_id: int):
    """Pick a genre from this book's genres that overlaps with USER_FACING_GENRES, else None."""
    try:
        genres = df_books_indexed.loc[book_id, "genres"]
    except KeyError:
        return None
    if not isinstance(genres, list):
        return None

    genres_lower = [g.lower() for g in genres]
    for g in genres_lower:
        for uf in USER_FACING_GENRES:
            if uf in g:
                return uf
    return None


def generate_dislike_genre_examples(n_examples: int = 2000, k: int = 5):
    """
    For each example:
      - Sample a popular seed book
      - Pick one of its genres as "disliked"
      - Ask teacher for recs avoiding that genre
      - Build (prompt, response)
    """
    book_counts = df_interactions["book_id"].value_counts()
    popular_book_ids = list(book_counts.index[:50_000])

    with DISLIKE_GENRE_OUT.open("w", encoding="utf-8") as f:
        made = 0
        attempts = 0
        max_attempts = n_examples * 20  # more attempts because of strict filters

        while made < n_examples and attempts < max_attempts:
            attempts += 1

            seed_id = int(random.choice(popular_book_ids))
            seed_title = get_book_title(seed_id)

            # Choose a genre to dislike
            disliked_genre = pick_genre_for_book(seed_id)
            if disliked_genre is None:
                continue  # this seed doesn't have a usable genre label

            rec_ids = recommend_avoid_genres(
                seed_id, disliked_genres=[disliked_genre], top_k=k, oversample=100
            )
            if len(rec_ids) < k:
                continue  # skip seeds where avoiding genre yields too few recs

            prompt = (
                f"I really enjoyed the book '{seed_title}', but I don't like {disliked_genre} books. "
                f"Please recommend {k} similar books that are not {disliked_genre}. "
                f"Return them as a numbered list with one book per line."
            )

            rec_list_text = format_recommendation_list(rec_ids)

            response = (
                f"Here are {k} books similar to '{seed_title}' that avoid {disliked_genre}:\n\n"
                f"{rec_list_text}"
            )

            record = {"prompt": prompt, "response": response}
            f.write(json.dumps(record) + "\n")
            made += 1

            if made % 100 == 0:
                print(f"[dislike-genre] generated {made}/{n_examples} examples")

        print(f"[dislike-genre] Done. Generated {made} examples, wrote to {DISLIKE_GENRE_OUT}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # Smaller numbers at first to sanity-check output:
    generate_basic_examples(n_examples=200, k=5)
    generate_dislike_genre_examples(n_examples=200, k=5)

    print("\nAll done. Check:", BASIC_OUT, "and", DISLIKE_GENRE_OUT)
