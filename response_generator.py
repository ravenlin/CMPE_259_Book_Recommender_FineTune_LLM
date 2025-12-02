# response_generator.py
#
# Full pipeline:
#   user text
#     → planner (Llama 2 via Ollama) → structured JSON preferences
#     → teacher (DescEmbeddingTeacher) → raw + filtered recs (+ unmatched_titles)
#     → responder (Llama 2 via Ollama) → final list-style answer for the user
#
# Requirements:
#   - Ollama running locally
#   - planner.py and pseudo_user_teacher.py present in the same directory
#
# Run a quick demo:
#   /usr/local/bin/python3 response_generator.py

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests  # pip install requests

from planner import call_ollama_planner, OLLAMA_URL, OLLAMA_MODEL
from pseudo_user_teacher import DescEmbeddingTeacher


# ---------------------------------
# System prompt for the RESPONDER
# ---------------------------------

RESPONDER_SYSTEM_PROMPT = """You are the final user-facing assistant for a book recommender system.

You receive:
1) The original user request (free-form text).
2) A JSON object "preferences" that describes what the planner extracted:
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
3) A JSON object "teacher_result" with:
   {
     "raw": [
       {
         "book_id": int,
         "title": string,
         "average_rating": float,
         "genres": [string],
         "authors": [string],
         "score": float
       },
       ...
     ],
     "filtered": [
       { same structure as above },
       ...
     ],
     "unmatched_titles": {
       "liked_books":    [string],
       "disliked_books": [string],
       "excluded_books": [string]
     }
   }

Your job:
- Produce a STRICT, concise recommendation message for the user.
- Use the "filtered" list as the main source of recommendations.
- If "filtered" is empty, fall back to "raw".
- Recommend exactly "num_recommendations" books if that many are available;
  otherwise list all available candidates from the chosen list.
- If any titles appear in teacher_result["unmatched_titles"], you MUST state
  them clearly at the top as "not found in the catalog".

CRITICAL FORMAT AND CONSTRAINTS (DO NOT VIOLATE):
- DO NOT describe the books.
- DO NOT justify or explain why they were chosen.
- DO NOT explain anything about the internal system (planner/teacher/embeddings).
- DO NOT invent or mention any books that are NOT present in the chosen candidate list
  (filtered if non-empty, otherwise raw).
- Every book you output must come directly from that candidate list.
- Your entire answer must follow this structure, in this exact order:

  1) If there are any unmatched titles, output a single short line:
       "These titles were not found in the catalog: <title1>, <title2>, ..."
     If there are none, skip this line entirely.

  2) Then output a line:
       "Here are your recommendations:"

  3) Then output a numbered list. Each item MUST be on a single line and MUST
     have this exact format:
       "<index>. <title> — <author1>, <author2>, ..."
     - <index> is 1, 2, 3, ...
     - <title> is the book "title" field.
     - The authors are taken from the "authors" field for that book.
       If there are no authors, omit the " — ..." part entirely.

  4) Finally, output one last line:
       "If you'd like to adjust this list, you can give more feedback (for example: different likes/dislikes or additional titles), and I will generate a new set of recommendations."

- DO NOT output anything else.
- DO NOT add summaries, descriptions, reasoning, apologies, or extra commentary.
- DO NOT output JSON.
"""


def call_ollama_responder(
    user_message: str,
    preferences: Dict[str, Any],
    teacher_result: Dict[str, Any],
) -> str:
    """
    Call Ollama/Llama2 to turn preferences + teacher_result into a
    strictly formatted, list-style answer for the user.
    """

    # Bundle context into one user message
    user_content = {
        "original_user_message": user_message,
        "preferences": preferences,
        "teacher_result": teacher_result,
    }

    prompt_text = (
        "Here is the context for generating a response:\n\n"
        + json.dumps(user_content, indent=2)
        + "\n\nFollow the system instructions exactly and write the final recommendation message for the user."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": RESPONDER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    content = data["message"]["content"]
    return content.strip()


def run_full_pipeline(user_message: str):
    """
    End-to-end:
      - planner parses user_message into preferences JSON
      - teacher generates raw + filtered recs (+ unmatched_titles)
      - responder generates final user-facing output (strict list)
    """

    # 1) Planner: parse request
    print("=== [PLANNER] Parsing user request ===")
    preferences = call_ollama_planner(user_message)
    print(json.dumps(preferences, indent=2))

    num_recs = preferences.get("num_recommendations", 10)
    try:
        num_recs = int(num_recs)
    except Exception:
        num_recs = 10
    if num_recs < 1:
        num_recs = 1
    if num_recs > 50:
        num_recs = 50

    # 2) Teacher: generate recommendations
    print(f"\n[info] planner requested ~{num_recs} recommendations.")
    top_filtered = num_recs
    top_raw = max(num_recs * 3, num_recs + 5)
    print(f"[info] Using top_raw={top_raw}, top_filtered={top_filtered}")

    print("\n=== [TEACHER] Generating recommendations ===")
    teacher = DescEmbeddingTeacher()
    teacher_result = teacher.recommend_from_query(
        preferences,
        top_raw=top_raw,
        top_filtered=top_filtered,
        alpha=1.0,
    )

    # 3) Responder: turn it all into a strict, list-only reply
    print("\n=== [RESPONDER] Generating final answer ===")
    reply = call_ollama_responder(
        user_message=user_message,
        preferences=preferences,
        teacher_result=teacher_result,
    )

    print("\n========= FINAL RESPONSE TO USER =========\n")
    print(reply)
    print("\n==========================================")


if __name__ == "__main__":
    # Example user request for quick testing
    demo_request = (
        "I absolutely loved 'The Hobbit' and 'Mistborn'. "
        "I hated 'Twilight'. I don't want any romance or YA, "
        "and please avoid Stephen King. "
        "Give me around 5 recommendations."
    )
    run_full_pipeline(demo_request)
