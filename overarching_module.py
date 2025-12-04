# overarching_module.py
#
# High-level orchestrator:
#   - Turn 1: user preference prompt → planner → teacher → fixed-format rec list
#   - Follow-up turns: user feedback → planner (with previous prefs + rec list) → teacher → new rec list
#
# This module does NOT call any LLM for the final response; it just formats
# the teacher's "filtered" recommendations into a fixed, deterministic string.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from planner import call_ollama_planner  # :contentReference[oaicite:3]{index=3}
from pseudo_user_teacher import DescEmbeddingTeacher  # :contentReference[oaicite:4]{index=4}
import json


def _format_recs_for_planner(recs: List[Dict[str, Any]]) -> str:
    """
    Minimal list format specifically for the planner.

    Matches the style documented in planner.py:

      number=<index> |
       <BOOK_TITLE>                            |
       -----------
    """
    lines: List[str] = []
    for idx, b in enumerate(recs):
        title = str(b.get("title", "")).strip()
        if not title:
            continue
        lines.append(
            f" number={idx} |\n"
            f" {title[:70]:70s} |\n"
            f" -----------"
        )
    return "\n".join(lines)


def _format_recs_for_user(recs: List[Dict[str, Any]]) -> str:
    """
    Fixed-format response shown to the user.

    This does NOT call any LLM. It just prints the filtered list
    with adjacent information in a deterministic layout.
    """
    if not recs:
        return (
            "I couldn't find any recommendations that match your current "
            "preferences. Try adjusting your likes/dislikes or genres."
        )

    lines: List[str] = []
    lines.append("Here are your current book recommendations:\n")

    for idx, b in enumerate(recs):
        title = str(b.get("title", "")).strip()
        rating = b.get("average_rating", None)
        genres = b.get("genres") or []
        authors = b.get("authors") or []
        score = b.get("score", None)

        lines.append(f"number={idx} |")
        lines.append(f" title: {title}")
        if rating is not None:
            try:
                lines.append(f" average_rating: {float(rating):.2f}")
            except Exception:
                lines.append(f" average_rating: {rating}")
        if genres:
            lines.append(" genres: " + ", ".join(str(g) for g in genres))
        if authors:
            lines.append(" authors: " + ", ".join(str(a) for a in authors))
        if score is not None:
            try:
                lines.append(f" score: {float(score):.4f}")
            except Exception:
                lines.append(f" score: {score}")
        lines.append("-----------")

    return "\n".join(lines)


class BookRecSession:
    """
    Stateful session that ties together:
      - planner.call_ollama_planner (preference extraction)
      - DescEmbeddingTeacher (pseudo-user teacher recommender)
      - fixed-format user responses

    Typical usage:

        from overarching_module import BookRecSession

        session = BookRecSession()

        # FIRST TURN
        first_prompt = "I loved The Hobbit and Mistborn, hated Twilight..."
        result1 = session.handle_first_turn(first_prompt)
        print(result1["response_text"])

        # User sees list and replies:
        feedback = "I really liked #0 and #2 from that list, but #1 felt too YA."
        result2 = session.handle_followup(feedback)
        print(result2["response_text"])
    """

    def __init__(self, teacher: Optional[DescEmbeddingTeacher] = None):
        # Teacher is relatively heavy to load, so we reuse it across turns.
        self.teacher: DescEmbeddingTeacher = teacher or DescEmbeddingTeacher()
        self.previous_preferences: Optional[Dict[str, Any]] = None
        self.last_filtered_recs: List[Dict[str, Any]] = []

    # -----------------------
    # Turn 1 (no history yet)
    # -----------------------

    def handle_first_turn(self, user_message: str) -> Dict[str, Any]:
        """
        First-turn entry point.

        - user_message: free-form preference description ("I liked X, hated Y...")
        - Returns a dict with:
            {
              "preferences": <planner JSON>,
              "teacher_result": <teacher raw/filtered/unmatched>,
              "response_text": <fixed-format string for the user>,
              "rec_list_for_feedback": <string block to reuse in follow-ups>
            }
        """
        # 1) Planner interprets the raw text into preference JSON
        prefs = call_ollama_planner(user_message)
        self.previous_preferences = prefs

        num_recs = int(prefs.get("num_recommendations", 10))
        if num_recs < 1:
            num_recs = 1

        # 2) Teacher generates recommendations from the planner query
        teacher_result = self.teacher.recommend_from_query(
            query=prefs,
            top_raw=max(num_recs * 2, 20),
            top_filtered=num_recs,
        )
        filtered = teacher_result.get("filtered", [])
        self.last_filtered_recs = filtered

        # 3) Build a fixed-format user response
        response_text = _format_recs_for_user(filtered)

        # 4) Build the text snippet that will be fed back into the planner
        #    on follow-up turns for "#2", "#3" style references.
        rec_list_for_feedback = _format_recs_for_planner(filtered)

        return {
            "preferences": prefs,
            "teacher_result": teacher_result,
            "response_text": response_text,
            "rec_list_for_feedback": rec_list_for_feedback,
        }

    # -----------------------
    # Follow-up turns
    # -----------------------

    def handle_followup(self, user_feedback_message: str) -> Dict[str, Any]:
        """
        Follow-up turn entry point.

        - user_feedback_message: the user's natural-language feedback about the
          last recommendation list (e.g., "I liked #0 and #2, but disliked #1.")

        This method:
          1) Injects the previous rec list into the planner prompt, like the demo.
          2) Calls the planner with `previous_preferences`.
          3) Calls the teacher again with the updated prefs.
          4) Returns a new fixed-format response and updated metadata.
        """
        if self.previous_preferences is None or not self.last_filtered_recs:
            raise RuntimeError(
                "No previous recommendations found. "
                "Call handle_first_turn() before handle_followup()."
            )

        # Reuse the planner-style list format so the planner can parse "#0", "#1", etc.
        rec_list_text = _format_recs_for_planner(self.last_filtered_recs)

        # Build combined message for the planner, similar to planner._demo()
        planner_message = (
            "Here is the last recommendation list:\n\n"
            f"{rec_list_text}\n\n"
            f"{user_feedback_message}"
        )

        # 1) Planner updates preferences using both previous prefs + new feedback
        prefs = call_ollama_planner(
            user_message=planner_message,
            previous_preferences=self.previous_preferences,
        )
        self.previous_preferences = prefs
        
        print("=== FINAL FOLLOW-UP PREFS ===")
        print(json.dumps(prefs, indent=2))
        print()

        num_recs = int(prefs.get("num_recommendations", 10))
        if num_recs < 1:
            num_recs = 1

        # 2) Teacher generates a new set of recs from the updated prefs
        teacher_result = self.teacher.recommend_from_query(
            query=prefs,
            top_raw=max(num_recs * 2, 20),
            top_filtered=num_recs,
        )
        filtered = teacher_result.get("filtered", [])
        self.last_filtered_recs = filtered

        # 3) Fixed-format response
        response_text = _format_recs_for_user(filtered)
        rec_list_for_feedback = _format_recs_for_planner(filtered)

        return {
            "preferences": prefs,
            "teacher_result": teacher_result,
            "response_text": response_text,
            "rec_list_for_feedback": rec_list_for_feedback,
        }


# Optional: quick CLI-ish demo if run directly
if __name__ == "__main__":
    session = BookRecSession()

    print("=== FIRST TURN ===")
    first_msg = (
        "I absolutely loved 'The Hobbit' and 'Mistborn'. "
        "I disliked 'Twilight' and I don't want YA or romance. "
        "Give me around 5 recommendations."
    )
    out1 = session.handle_first_turn(first_msg)
    print(out1["response_text"])

    print("\n=== FOLLOW-UP TURN ===")
    # In a real app, the user would see the response_text and then type feedback
    feedback = (
        "From that list, I liked #0 and #2, but #1 felt too slow. "
        "I think I enjoy epic fantasy and heist stories."
    )
    out2 = session.handle_followup(feedback)
    print(out2["response_text"])
