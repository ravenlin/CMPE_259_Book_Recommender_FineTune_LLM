# planner.py
#
# Refactored planner using prompt chaining with simple extractors.
#
# For EACH turn, we call several small LLM extractors:
#   - liked_books
#   - disliked_books
#   - excluded_books
#   - liked_genres
#   - disliked_genres
#   - liked_authors
#   - disliked_authors
#   - num_recommendations
#
# FIRST TURN:
#   context = { "current_user_message": <user text> }
#
# FOLLOW-UP TURN:
#   context = {
#       "current_user_message": <user text>,
#       "previous_preferences": <JSON from prior turn>
#   }
#
# Each extractor has a meta instruction:
#   - DO NOT hallucinate
#   - Only include books/authors/genres that appear either:
#       - in current_user_message, OR
#       - in previous_preferences JSON
#
# PLUS: we add a deterministic Python post-filter that enforces this rule,
# and a merge step that preserves previous preferences across turns.
#
# Public API:
#   call_ollama_planner(user_message: str, previous_preferences: dict | None = None) -> dict
#
#   - If previous_preferences is None → first turn
#   - Else → follow-up turn

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

import requests  # pip install requests

# -----------------------------
# Ollama config
# -----------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"

# -----------------------------
# Normalization helpers
# -----------------------------

REQUIRED_PREF_KEYS = {
    "liked_books",
    "disliked_books",
    "excluded_books",
    "liked_genres",
    "disliked_genres",
    "liked_authors",
    "disliked_authors",
    "num_recommendations",
}


def _safe_float(x, default: float = 1.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _normalize_preferences(prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw preferences dict into the exact schema we need.

    Ensures:
      - all required keys present
      - liked/disliked/excluded book lists are lists of dicts
      - genre/author lists are lists of strings
      - num_recommendations is a positive int, default 10
    """
    if not isinstance(prefs, dict):
        prefs = {}

    liked_books_raw = prefs.get("liked_books", []) or []
    disliked_books_raw = prefs.get("disliked_books", []) or []
    excluded_books_raw = prefs.get("excluded_books", []) or []

    def _norm_book_list(raw) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                # try to interpret as title string
                title = str(item)
                out.append({"title": title, "rating": 5.0})
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            rating = _safe_float(item.get("rating", 5.0), default=5.0)
            out.append({"title": title, "rating": rating})
        return out

    liked_books = _norm_book_list(liked_books_raw)
    disliked_books = _norm_book_list(disliked_books_raw)

    def _norm_excluded_list(raw) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    if title:
                        out.append({"title": title})
                else:
                    title = str(item).strip()
                    if title:
                        out.append({"title": title})
        else:
            if raw:
                title = str(raw).strip()
                if title:
                    out.append({"title": title})
        return out

    excluded_books = _norm_excluded_list(excluded_books_raw)

    def _norm_str_list(raw) -> List[str]:
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for v in raw:
            if isinstance(v, dict) and "author" in v:
                s = str(v["author"]).strip()
            else:
                s = str(v).strip()
            if s:
                out.append(s)
        return out

    liked_genres = _norm_str_list(prefs.get("liked_genres", []) or [])
    disliked_genres = _norm_str_list(prefs.get("disliked_genres", []) or [])
    liked_authors = _norm_str_list(prefs.get("liked_authors", []) or [])
    disliked_authors = _norm_str_list(prefs.get("disliked_authors", []) or [])

    num_recs_raw = prefs.get("num_recommendations", 10)
    try:
        num_recs = int(num_recs_raw)
    except Exception:
        num_recs = 10
    if num_recs < 1:
        num_recs = 1
    if num_recs > 50:
        num_recs = 50

    return {
        "liked_books": liked_books,
        "disliked_books": disliked_books,
        "excluded_books": excluded_books,
        "liked_genres": liked_genres,
        "disliked_genres": disliked_genres,
        "liked_authors": liked_authors,
        "disliked_authors": disliked_authors,
        "num_recommendations": num_recs,
    }


# -----------------------------
# Shared base meta-prompt
# -----------------------------

BASE_META_PROMPT = """
You are an INFORMATION EXTRACTION assistant for a book recommender system.

You will receive a JSON object that ALWAYS contains:
- "current_user_message": the latest natural-language message from the user.
- Optionally "previous_preferences": the JSON preferences from earlier turns, with keys:
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

The current_user_message may also contain a printed list of recommendations
in the following format:

  number=<index> |
   <BOOK_TITLE>                            |
   -----------

Users may refer to these items by position (e.g., "#2", "#4") or by title.

CRITICAL ANTI-HALLUCINATION RULES:
- You MUST NOT invent or hallucinate any new book titles, authors, or genres.
- You may ONLY use book titles, author names, and genres that appear:
    - explicitly in "current_user_message", OR
    - explicitly in "previous_preferences" (in any field).
- DO NOT infer or guess authors from book titles.
- DO NOT infer or guess genres from world knowledge if they are not written.
- If something is not present in those inputs, DO NOT include it.
- If there is no information for your field, return the appropriate empty structure.

OUTPUT RULES:
- You MUST respond with valid, strict JSON ONLY, with the exact structure requested
  for each specific task.
- DO NOT include explanations, comments, prose, or extra keys.
"""


# -----------------------------
# Low-level field extractor
# -----------------------------

def _call_field_extractor(field_prompt: str, context_obj: Dict[str, Any]) -> Any:
    """
    Generic helper to call Ollama with:
      - a global meta prompt (anti-hallucination, schema rules)
      - a small field-specific instruction (e.g., "What books were liked?")
    """
    system_prompt = BASE_META_PROMPT + "\n\n" + field_prompt.strip()

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                # We pass the context as JSON so the model can only pick
                # from current_user_message + previous_preferences.
                "content": json.dumps(context_obj, indent=2),
            },
        ],
        "stream": False,
        # Ask Ollama to produce JSON only
        "format": "json",
        "options": {"temperature": 0.0},
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    # print("RAW FIELD RESPONSE:", content)
    try:
        return json.loads(content)
    except JSONDecodeError:
        print("JSON parse error in field extractor; returning None")
        return None


# -----------------------------
# Field-specific prompts
# -----------------------------

LIKED_BOOKS_PROMPT = """
TASK: Answer ONLY this question:

What books were liked?

Rules:
- Consider positive expressions such as "liked", "loved", "enjoyed", "favorite"
  in current_user_message.
- The current_user_message may include a numbered list of recommendations like:
    number=0 |
     SOME_BOOK_TITLE |
     -----------
  The user may say "I liked #2 and #4" to refer to those items by index.
- If previous_preferences is provided, include any existing liked_books
  unless the user clearly contradicts them.
- Do NOT add any book that does NOT appear either in:
    - current_user_message, OR
    - previous_preferences.*books lists.
- Use rating ≈ 5.0 for "loved" or "favorite" books.
- Use rating 3–4 if they just "kinda liked" them or were neutral-positive.

JSON RESPONSE FORMAT:
Return a JSON LIST (not an object). Example of structure (placeholder values):

[
  {"title": "BOOK_TITLE_1", "rating": 5.0},
  {"title": "BOOK_TITLE_2", "rating": 4.0}
]

If there are no liked books, return: []
"""

DISLIKED_BOOKS_PROMPT = """
TASK: Answer ONLY this question:

What books were disliked?

Rules:
- Include any books (by title) the user explicitly says they disliked or hated,
  or refers to negatively (e.g., "I really didn't like #1").
- The current_user_message may include a numbered list of recommendations like:
    number=0 |
     SOME_BOOK_TITLE |
     -----------
  The user may reference items by index (e.g., "#1", "#3").
- If previous_preferences is provided, include existing disliked_books
  unless the user clearly changes their mind.
- Do NOT add any book that does NOT appear either in:
    - current_user_message, OR
    - previous_preferences.*books lists.
- Use low ratings such as 1.0 or 2.0 for disliked books.

JSON RESPONSE FORMAT:
Return a JSON LIST. Example of structure (placeholder values):

[
  {"title": "BOOK_TITLE_1", "rating": 1.0}
]

If there are no disliked books, return: []
"""

EXCLUDED_BOOKS_PROMPT = """
TASK: Answer ONLY this question:

What books does the user want to EXCLUDE from future recommendations?

Rules:
- These are books the user explicitly does NOT want recommended again,
  e.g., "don't recommend this again", "no more of that one",
  "I've read this already, don't show it again". 
- If the user states, "I dislike" or any other sentiment of dislike, they should not be included in the 
  exclusions.
- Exclusions may refer either to titles in previous_preferences or titles
  in the numbered recommendation list shown in current_user_message.
- If previous_preferences is provided, include existing excluded_books
  unless the user clearly changes that preference.
- Do NOT automatically exclude liked or disliked books unless the user
  explicitly indicates they should be excluded from future suggestions.
- Do NOT add any book that does NOT appear either in:
    - current_user_message, OR
    - previous_preferences.*books lists.

JSON RESPONSE FORMAT:
Return a JSON LIST. Example of structure (placeholder values):

[
  {"title": "BOOK_TITLE_1"},
  {"title": "BOOK_TITLE_2"}
]

If there are no excluded books, return: []
"""

LIKED_GENRES_PROMPT = """
TASK: Answer ONLY this question:

What genres does the user LIKE?

Rules:
- Include genres that are explicitly mentioned positively in current_user_message
  (e.g., "I like <GENRE>", "I enjoy <GENRE>").
- Include previously stored liked_genres from previous_preferences.liked_genres
  unless the user now says they dislike them.
- DO NOT include genres that were only mentioned negatively
  (e.g., "no <GENRE>", "I don't want <GENRE>").
- DO NOT invent new genres that do not appear in the text or previous_preferences.

JSON RESPONSE FORMAT:
Return a JSON LIST of strings. Example structure (placeholder values):

["GENRE_1", "GENRE_2"]

If there are no liked genres, return: []
"""

DISLIKED_GENRES_PROMPT = """
TASK: Answer ONLY this question:

What genres does the user DISLIKE or want to AVOID?

Rules:
- Look for explicit negative phrases such as:
  "no <GENRE>", "I don't want <GENRE>", "avoid <GENRE>", "not into <GENRE>".
- Include those genres as disliked.
- Also include any previously stored disliked_genres from
  previous_preferences.disliked_genres unless the user reverses them.
- DO NOT invent genres that are not mentioned in current_user_message
  or previous_preferences.

JSON RESPONSE FORMAT:
Return a JSON LIST of strings. Example structure (placeholder values):

["GENRE_1", "GENRE_2"]

If there are no disliked genres, return: []
"""

LIKED_AUTHORS_PROMPT = """
TASK: Answer ONLY this question:

What authors does the user LIKE?

STRICT RULES:
- ONLY include authors whose NAMES appear literally in:
    - current_user_message, OR
    - previous_preferences.liked_authors.
- DO NOT infer or guess authors from book titles.
- Keep previously liked authors from previous_preferences.liked_authors
  unless the user clearly reverses their preference.
- If no author names are explicitly written anywhere, return [].

JSON RESPONSE FORMAT:
Return a JSON LIST of strings. Example structure (placeholder values):

["AUTHOR_NAME_1", "AUTHOR_NAME_2"]

If there are no liked authors, return: []
"""

DISLIKED_AUTHORS_PROMPT = """
TASK: Answer ONLY this question:

What authors does the user DISLIKE or want to AVOID?

STRICT RULES:
- ONLY include authors whose NAMES appear literally in:
    - current_user_message, OR
    - previous_preferences.disliked_authors, OR
    - author-name fields in previous_preferences (if any).
- DO NOT infer or guess authors from book titles.
- Keep previously disliked authors from previous_preferences.disliked_authors
  unless the user clearly changes their mind.
- If no author names are explicitly written anywhere, return [].

JSON RESPONSE FORMAT:
Return a JSON LIST of strings. Example structure (placeholder values):

["AUTHOR_NAME_1"]

If there are no disliked authors, return: []
"""

NUM_RECS_PROMPT = """
TASK: Answer ONLY this question:

How many recommendations does the user want in THIS TURN?

Rules:
- If the user clearly says "give me N recommendations" or "around N", use N.
- If they say "a couple", use 2.
- If they say "a few", use a number between 3 and 5 (you choose).
- Otherwise:
    - If previous_preferences is provided, you MAY reuse its
      num_recommendations if the user does not specify anything new.
    - If there is no previous_preferences or no explicit number, default to 10.

JSON RESPONSE FORMAT:
Return a JSON OBJECT with exactly one key, for example:

{ "num_recommendations": 5 }
"""


# -----------------------------
# Deterministic anti-hallucination filter
# -----------------------------

def _build_context_blob(context_obj: Dict[str, Any], include_previous: bool = True) -> str:
    """
    Build a lowercase text blob of what the model is allowed to reference.

    If include_previous=True:
        - current_user_message + previous_preferences
    Else:
        - only current_user_message
    """
    data: Dict[str, Any] = {
        "current_user_message": context_obj.get("current_user_message", "")
    }
    if include_previous and "previous_preferences" in context_obj:
        data["previous_preferences"] = context_obj["previous_preferences"]
    return json.dumps(data, ensure_ascii=False).lower()


def _string_is_allowed(value: str, ctx_blob: str) -> bool:
    """
    Return True iff the string appears (case-insensitive) in the context blob.
    """
    return value.strip().lower() in ctx_blob


def _filter_prefs_against_context(
    prefs_raw: Dict[str, Any],
    context_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Enforce the anti-hallucination rule in code:

    - Only keep titles/authors/genres whose strings appear in:
      - current_user_message, OR
      - previous_preferences (for books/genres),
      with extra safeguards for exclusions and authors.
    """
    ctx_all = _build_context_blob(context_obj, include_previous=True)
    ctx_user = _build_context_blob(context_obj, include_previous=False)

    prev = context_obj.get("previous_preferences") or {}
    prev_excluded_titles = {
        str(b.get("title", "")).strip().lower()
        for b in prev.get("excluded_books", [])
        if isinstance(b, dict)
    }
    prev_liked_authors = {
        str(a).strip().lower() for a in prev.get("liked_authors", [])
    }
    prev_disliked_authors = {
        str(a).strip().lower() for a in prev.get("disliked_authors", [])
    }

    def _filter_book_list_general(raw_list: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(raw_list, list):
            return out
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            if _string_is_allowed(title, ctx_all):
                out.append(item)
        return out

    def _filter_excluded_list(raw_list: Any) -> List[Dict[str, Any]]:
        """
        Exclusions are stricter: a title can be excluded if either:
          - it was previously excluded, OR
          - it appears (by string) in the current_user_message.
        """
        out: List[Dict[str, Any]] = []
        if not isinstance(raw_list, list):
            return out
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            t_low = title.lower()
            if t_low in prev_excluded_titles or _string_is_allowed(title, ctx_user):
                out.append(item)
        return out

    def _filter_genre_list(raw_list: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(raw_list, list):
            return out
        for s in raw_list:
            s_str = str(s).strip()
            if not s_str:
                continue
            if _string_is_allowed(s_str, ctx_all):
                out.append(s_str)
        return out

    def _filter_liked_authors_list(raw_list: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(raw_list, list):
            return out
        for s in raw_list:
            s_str = str(s).strip()
            if not s_str:
                continue
            s_low = s_str.lower()
            # Allow if user literally typed the name, or it was previously liked.
            if _string_is_allowed(s_str, ctx_user) or s_low in prev_liked_authors:
                out.append(s_str)
        return out

    def _filter_disliked_authors_list(raw_list: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(raw_list, list):
            return out
        for s in raw_list:
            s_str = str(s).strip()
            if not s_str:
                continue
            s_low = s_str.lower()
            # Allow if user literally typed the name, or it was previously disliked.
            if _string_is_allowed(s_str, ctx_user) or s_low in prev_disliked_authors:
                out.append(s_str)
        return out

    return {
        "liked_books": _filter_book_list_general(prefs_raw.get("liked_books", [])),
        "disliked_books": _filter_book_list_general(prefs_raw.get("disliked_books", [])),
        "excluded_books": _filter_excluded_list(prefs_raw.get("excluded_books", [])),
        "liked_genres": _filter_genre_list(prefs_raw.get("liked_genres", [])),
        "disliked_genres": _filter_genre_list(prefs_raw.get("disliked_genres", [])),
        "liked_authors": _filter_liked_authors_list(prefs_raw.get("liked_authors", [])),
        "disliked_authors": _filter_disliked_authors_list(
            prefs_raw.get("disliked_authors", [])
        ),
        "num_recommendations": prefs_raw.get("num_recommendations", 10),
    }


# -----------------------------
# Merge with previous preferences
# -----------------------------

def _merge_with_previous(
    previous_prefs: Dict[str, Any],
    new_prefs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge previous normalized preferences with new filtered preferences.

    Goal:
    - Preserve previous likes/dislikes/exclusions unless the new turn
      explicitly overrides them.
    """
    prev = _normalize_preferences(previous_prefs)
    cur = _normalize_preferences(new_prefs)

    def _index_books(lst: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            str(b.get("title", "")).strip(): b
            for b in lst
            if isinstance(b, dict) and b.get("title")
        }

    liked = _index_books(prev["liked_books"])
    disliked = _index_books(prev["disliked_books"])
    excluded = _index_books(prev["excluded_books"])

    # Apply current on top (new information has precedence).
    for b in cur["liked_books"]:
        liked[str(b["title"]).strip()] = b
    for b in cur["disliked_books"]:
        disliked[str(b["title"]).strip()] = b
    for b in cur["excluded_books"]:
        excluded[str(b["title"]).strip()] = b

    # Conflict resolution:
    # 1) Titles newly liked: remove from disliked/excluded.
    for title in list(liked.keys()):
        if title in disliked:
            del disliked[title]
        if title in excluded:
            del excluded[title]

    # 2) Titles newly excluded: remove from liked/disliked.
    for title in list(excluded.keys()):
        if title in liked:
            del liked[title]
        if title in disliked:
            del disliked[title]

    # 3) Titles newly disliked: remove from liked.
    for title in list(disliked.keys()):
        if title in liked:
            del liked[title]

    # Merge string lists (keep order, remove duplicates).
    def _merge_str_lists(prev_list: List[str], cur_list: List[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for v in prev_list + cur_list:
            s = str(v).strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                merged.append(s)
        return merged

    merged = {
        "liked_books": list(liked.values()),
        "disliked_books": list(disliked.values()),
        "excluded_books": list(excluded.values()),
        "liked_genres": _merge_str_lists(prev["liked_genres"], cur["liked_genres"]),
        "disliked_genres": _merge_str_lists(prev["disliked_genres"], cur["disliked_genres"]),
        "liked_authors": _merge_str_lists(prev["liked_authors"], cur["liked_authors"]),
        "disliked_authors": _merge_str_lists(
            prev["disliked_authors"], cur["disliked_authors"]
        ),
        "num_recommendations": (
            cur["num_recommendations"] or prev["num_recommendations"]
        ),
    }

    return merged


# -----------------------------
# Field-specific extractor wrappers
# -----------------------------

def _extract_liked_books(context_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = _call_field_extractor(LIKED_BOOKS_PROMPT, context_obj)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "liked_books" in data:
        v = data["liked_books"]
        return v if isinstance(v, list) else []
    return []


def _extract_disliked_books(context_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = _call_field_extractor(DISLIKED_BOOKS_PROMPT, context_obj)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "disliked_books" in data:
        v = data["disliked_books"]
        return v if isinstance(v, list) else []
    return []


def _extract_excluded_books(context_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = _call_field_extractor(EXCLUDED_BOOKS_PROMPT, context_obj)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "excluded_books" in data:
        v = data["excluded_books"]
        return v if isinstance(v, list) else []
    return []


def _extract_liked_genres(context_obj: Dict[str, Any]) -> List[str]:
    data = _call_field_extractor(LIKED_GENRES_PROMPT, context_obj)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "liked_genres" in data:
        v = data["liked_genres"]
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def _extract_disliked_genres(context_obj: Dict[str, Any]) -> List[str]:
    data = _call_field_extractor(DISLIKED_GENRES_PROMPT, context_obj)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "disliked_genres" in data:
        v = data["disliked_genres"]
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def _extract_liked_authors(context_obj: Dict[str, Any]) -> List[str]:
    data = _call_field_extractor(LIKED_AUTHORS_PROMPT, context_obj)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "liked_authors" in data:
        v = data["liked_authors"]
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def _extract_disliked_authors(context_obj: Dict[str, Any]) -> List[str]:
    data = _call_field_extractor(DISLIKED_AUTHORS_PROMPT, context_obj)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "disliked_authors" in data:
        v = data["disliked_authors"]
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def _extract_num_recs(context_obj: Dict[str, Any]) -> int:
    data = _call_field_extractor(NUM_RECS_PROMPT, context_obj)
    if isinstance(data, dict) and "num_recommendations" in data:
        try:
            n = int(data["num_recommendations"])
            if n < 1:
                n = 1
            if n > 50:
                n = 50
            return n
        except Exception:
            pass
    # fallback default
    return 10


# -----------------------------
# Public planner API
# -----------------------------

def call_ollama_planner(
    user_message: str,
    previous_preferences: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main planner function.

    FIRST TURN:
      call_ollama_planner(user_message)
        -> only the current message is used.

    FOLLOW-UP TURN:
      call_ollama_planner(user_message, previous_preferences)
        -> previous_preferences + current user message are used.
    """
    context: Dict[str, Any] = {"current_user_message": user_message}

    prev_norm: Optional[Dict[str, Any]] = None
    if previous_preferences is not None:
        prev_norm = _normalize_preferences(previous_preferences)
        context["previous_preferences"] = prev_norm

    # Raw extraction from the model (may hallucinate)
    liked_books = _extract_liked_books(context)
    disliked_books = _extract_disliked_books(context)
    excluded_books = _extract_excluded_books(context)
    liked_genres = _extract_liked_genres(context)
    disliked_genres = _extract_disliked_genres(context)
    liked_authors = _extract_liked_authors(context)
    disliked_authors = _extract_disliked_authors(context)
    num_recommendations = _extract_num_recs(context)

    prefs_raw = {
        "liked_books": liked_books,
        "disliked_books": disliked_books,
        "excluded_books": excluded_books,
        "liked_genres": liked_genres,
        "disliked_genres": disliked_genres,
        "liked_authors": liked_authors,
        "disliked_authors": disliked_authors,
        "num_recommendations": num_recommendations,
    }

    # Deterministic self-check: drop anything that doesn't literally appear
    # in the allowed context.
    prefs_filtered = _filter_prefs_against_context(prefs_raw, context)

    # Merge with previous preferences so we don't "forget" earlier info.
    if prev_norm is not None:
        prefs_merged = _merge_with_previous(prev_norm, prefs_filtered)
    else:
        prefs_merged = prefs_filtered

    # Final normalization into the schema used by the rest of the system.
    
    final = _normalize_preferences(prefs_merged)
    return final


# -----------------------------
# Demo / manual test
# -----------------------------

def _format_recs_for_prompt(recs):
    """
    Format a list of recommendation dicts into the same
    text style as your recommender module prints, so the
    planner can see the list in the follow-up turn.
    """
    lines = []
    for idx, b in enumerate(recs):
        lines.append(
            f" number={idx} |\n"
            f" {b['title'][:70]:70s} |\n"
            f" -----------"
        )
    return "\n".join(lines)


def _demo():
    # First turn
    first_message = (
        "I absolutely loved 'The Hobbit' and 'The Lord Of The Rings'. "
        "I disliked 'Twilight'. I don't want romance or young adult. "
        "Please give me around 5 recommendations."
    )

    first_prefs = call_ollama_planner(first_message)
    print("=== FINAL FIRST-TURN PREFS ===")
    print(json.dumps(first_prefs, indent=2))
    print()

    # Dummy recommendation list simulating another module's output
    dummy_recs = [
        {
            "book_id": 101,
            "title": "The Name of the Wind",
            "average_rating": 4.51,
            "genres": ["fantasy", "epic"],
            "authors": ["Patrick Rothfuss"],
            "score": 0.9234,
        },
        {
            "book_id": 102,
            "title": "The Way of Kings",
            "average_rating": 4.65,
            "genres": ["fantasy", "epic"],
            "authors": ["Brandon Sanderson"],
            "score": 0.9172,
        },
        {
            "book_id": 103,
            "title": "The Lies of Locke Lamora",
            "average_rating": 4.29,
            "genres": ["fantasy", "heist"],
            "authors": ["Scott Lynch"],
            "score": 0.9021,
        },
        {
            "book_id": 104,
            "title": "Good Omens",
            "average_rating": 4.25,
            "genres": ["fantasy", "humor"],
            "authors": ["Neil Gaiman", "Terry Pratchett"],
            "score": 0.8877,
        },
    ]


    rec_list_text = _format_recs_for_prompt(dummy_recs)

    # Follow-up turn: include the list + user commentary
    followup_message = (
        "Here is the last recommendation list:\n\n"
        f"{rec_list_text}\n\n"
        "From that list, I liked #1 and #3, but I really didn't like #0. "
        "Also, I think I enjoy fantasy."
    )

    print(followup_message)
    follow_prefs = call_ollama_planner(
        user_message=followup_message,
        previous_preferences=first_prefs,
    )

    print("=== FINAL FOLLOW-UP PREFS ===")
    print(json.dumps(follow_prefs, indent=2))
    print()


if __name__ == "__main__":
    _demo()
