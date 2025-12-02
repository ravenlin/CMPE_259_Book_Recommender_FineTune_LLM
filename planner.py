# planner.py
#
# LLM-based "planner" that:
#   - calls Ollama (Llama 2) locally
#   - parses a user's free-form request into structured JSON
#   - infers how many recommendations the user wants (if mentioned)
#   - feeds that JSON into DescEmbeddingTeacher.recommend_from_query

from __future__ import annotations

import json
from typing import Any, Dict

import requests  # pip install requests

from pseudo_user_teacher import DescEmbeddingTeacher


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama2"  # or whatever tag you pulled (e.g. "llama2:7b-chat")


SYSTEM_PROMPT = """You are a 'planner' for a book recommender system.

Your job:
- Read a user's free-form request about books they like, dislike, genres, authors, exclusions,
  and how many recommendations they want (if they mention a number).
- Extract a structured JSON object describing their preferences.
- DO NOT call tools, and DO NOT explain your reasoning.
- Output ONLY valid JSON following the schema below.

JSON schema (all fields must exist):

{
  "liked_books":         [{"title": string, "rating": number}],
  "disliked_books":      [{"title": string, "rating": number}],
  "excluded_books":      [{"title": string}],
  "liked_genres":        [string],
  "disliked_genres":     [string],
  "liked_authors":       [string],
  "disliked_authors":    [string],
  "num_recommendations": number
}

Guidelines:

- Use 'liked_books' for books the user clearly likes or loves.
- Use 'disliked_books' for books the user clearly dislikes or hates.
- Use 'excluded_books' for books the user says they have already read or explicitly don't want recommended again.
- If the user expresses strong positive or negative sentiment, map it to a rating:
  - "absolutely loved / favorite / all-time favorite" → 5.0
  - "really liked / loved" → 4.5
  - "liked" → 4.0
  - "it was okay / fine" → 3.0
  - "didn't like" → 2.0
  - "hated / awful / terrible" → 1.0
- If the user only lists a liked book with no sentiment, default rating to 4.5.
- If the user only lists a disliked book with no sentiment, default rating to 1.5.

- 'liked_genres' and 'disliked_genres' are coarse genres like "fantasy", "science fiction", "romance",
  "mystery", "young adult", "nonfiction", "historical", "horror", "classics", "graphic novel", "poetry",
  "spirituality", etc. Infer from phrases like "I enjoy fantasy and sci-fi" or
  "I don't want anything scary or horror".
- 'liked_authors' and 'disliked_authors' should contain the exact author names mentioned.
- Never invent book titles or authors that the user did not mention.
- Always return arrays (possibly empty) for each field except 'num_recommendations'.

Inferring num_recommendations:

- If the user explicitly mentions a number of books, use that:
  - "recommend 5 books" → 5
  - "give me top 20" → 20
  - "a couple of books" → 2
  - "a few books" → 3 or 4 (choose 4)
  - "a dozen" → 12
- If the user says "some", "several", or similar without a number, choose a small integer between 5 and 10.
- If the user does not specify how many recommendations they want, set "num_recommendations" to 10.

Output requirements:
- Output ONLY the JSON object, no backticks, no extra text.
"""


def build_user_prompt(user_message: str) -> str:
    example = """
Example:

User request:
"I absolutely loved 'The Hobbit' and 'Mistborn'. I hated 'Twilight'.
I don't want any romance or YA. Please also exclude 'The Hobbit'
because I've already read it multiple times. I don't like Stephen King.
Give me around 5 recommendations."

Expected JSON:

{
  "liked_books": [
    {"title": "The Hobbit", "rating": 5.0},
    {"title": "Mistborn", "rating": 4.5}
  ],
  "disliked_books": [
    {"title": "Twilight", "rating": 1.0}
  ],
  "excluded_books": [
    {"title": "The Hobbit"}
  ],
  "liked_genres": [],
  "disliked_genres": ["romance", "young adult"],
  "liked_authors": [],
  "disliked_authors": ["Stephen King"],
  "num_recommendations": 5
}

Now process this request:

User request:
"""
    return example + user_message


def call_ollama_planner(user_message: str) -> Dict[str, Any]:
    """
    Send the user's request to Ollama (Llama 2) and parse the JSON response.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(user_message)},
        ],
        # 'format': 'json' tells Ollama we want valid JSON only
        "format": "json",
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Ollama's /api/chat returns {"message": {"role": "...", "content": "..."}}
    content = data["message"]["content"]

    # Parse JSON (the model should output pure JSON)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Very defensive fallback: try to strip any leading/trailing junk
        content_stripped = content.strip()
        parsed = json.loads(content_stripped)

    # Ensure all expected list keys exist with defaults
    def ensure_list(d: Dict[str, Any], key: str):
        if key not in d or d[key] is None:
            d[key] = []
        if not isinstance(d[key], list):
            d[key] = [d[key]]

    for key in [
        "liked_books",
        "disliked_books",
        "excluded_books",
        "liked_genres",
        "disliked_genres",
        "liked_authors",
        "disliked_authors",
    ]:
        ensure_list(parsed, key)

    # Normalize liked/disliked books entries to have title + rating
    def normalize_book_entry(b: Any, default_rating: float) -> Dict[str, Any]:
        if not isinstance(b, dict):
            return {"title": str(b), "rating": default_rating}
        title = str(b.get("title", "")).strip()
        rating = b.get("rating", default_rating)
        try:
            rating = float(rating)
        except Exception:
            rating = default_rating
        return {"title": title, "rating": rating}

    parsed["liked_books"] = [
        normalize_book_entry(b, 4.5) for b in parsed["liked_books"] if str(b).strip()
    ]
    parsed["disliked_books"] = [
        normalize_book_entry(b, 1.5) for b in parsed["disliked_books"] if str(b).strip()
    ]

    # excluded_books only need titles
    normalized_excluded = []
    for b in parsed["excluded_books"]:
        if isinstance(b, dict):
            title = str(b.get("title", "")).strip()
        else:
            title = str(b).strip()
        if title:
            normalized_excluded.append({"title": title})
    parsed["excluded_books"] = normalized_excluded

    # Normalize simple string lists
    parsed["liked_genres"] = [str(g).strip() for g in parsed["liked_genres"] if str(g).strip()]
    parsed["disliked_genres"] = [str(g).strip() for g in parsed["disliked_genres"] if str(g).strip()]
    parsed["liked_authors"] = [str(a).strip() for a in parsed["liked_authors"] if str(a).strip()]
    parsed["disliked_authors"] = [str(a).strip() for a in parsed["disliked_authors"] if str(a).strip()]

    # Handle num_recommendations (default to 10 if missing/invalid)
    num_recs = parsed.get("num_recommendations", 10)
    try:
        num_recs = int(round(float(num_recs)))
    except Exception:
        num_recs = 10

    # Clamp to a reasonable range
    if num_recs < 1:
        num_recs = 1
    if num_recs > 50:
        num_recs = 50

    parsed["num_recommendations"] = num_recs

    return parsed


def run_planner_and_teacher(user_message: str, default_top: int = 10):
    """
    Full pipeline:
      user text -> planner (Llama 2 via Ollama) -> query dict -> DescEmbeddingTeacher
    """
    print("=== [PLANNER] Parsing user request ===")
    query = call_ollama_planner(user_message)
    print(json.dumps(query, indent=2))

    num_recs = query.get("num_recommendations", default_top)
    print(f"\n[info] num_recommendations interpreted as: {num_recs}")

    # Decide how many raw candidates to pull (larger pool → room for filtering)
    top_filtered = num_recs
    top_raw = max(num_recs * 3, num_recs + 5)

    print(f"[info] Using top_raw={top_raw}, top_filtered={top_filtered}")

    print("\n=== [TEACHER] Generating recommendations ===")
    teacher = DescEmbeddingTeacher()
    result = teacher.recommend_from_query(
        query,
        top_raw=top_raw,
        top_filtered=top_filtered,
        alpha=1.0,
    )

    print("\n--- RAW RECOMMENDATIONS ---")
    for b in result["raw"]:
        print(
            f"{b['book_id']:8d} | {b['title'][:70]:70s} | "
            f"rating={b['average_rating']:.2f} | "
            f"genres={b['genres']} | authors={b['authors']} | score={b['score']:.4f}"
        )

    print("\n--- FILTERED RECOMMENDATIONS ---")
    for b in result["filtered"]:
        print(
            f"{b['book_id']:8d} | {b['title'][:70]:70s} | "
            f"rating={b['average_rating']:.2f} | "
            f"genres={b['genres']} | authors={b['authors']} | score={b['score']:.4f}"
        )


if __name__ == "__main__":
    # quick manual test
    demo_request = (
        "I absolutely loved 'The Hobbit' and 'Mistborn'."
        "I hated 'Twilight'. I don't want any romance or YA. "
        "Please exclude 'The Hobbit' because I've already read it."
        "I don't like Stephen King. Give me 7 recommendations."
    )
    run_planner_and_teacher(demo_request)
