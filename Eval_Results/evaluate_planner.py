"""
evaluate_planner.py

Evaluation harness for the planner defined in planner.py.
Uses a synthetic dataset of 500 examples and computes:
- JSON validity
- Per-field precision/recall/F1
- Hallucination detection
- Num_recommendations matching

Works directly with the user's existing planner.py (no modifications required).
"""

import random
import json
import csv
from typing import Dict, List, Any, Tuple

from planner import call_ollama_planner  # uses current planner config

random.seed(42)

# ----------------------------------------------------------
# Synthetic data generation helpers
# ----------------------------------------------------------

BOOK_TITLES = [
    "Dune", "Foundation", "The Hobbit", "Mistborn",
    "Ender's Game", "The Way of Kings", "The Name of the Wind",
    "The Lies of Locke Lamora", "Good Omens", "Dracula",
    "Pride and Prejudice", "Twilight", "Neuromancer",
    "Snow Crash", "The Martian", "Ready Player One"
]

GENRES = [
    "fantasy", "sci-fi", "romance", "horror", "mystery",
    "thriller", "adventure", "young adult"
]

AUTHORS = [
    "Brandon Sanderson", "J.R.R. Tolkien", "Patrick Rothfuss",
    "Isaac Asimov", "Frank Herbert", "Neil Gaiman",
    "Terry Pratchett", "Dan Simmons"
]


def random_books(n: int) -> List[str]:
    return random.sample(BOOK_TITLES, k=n)


def random_authors(n: int) -> List[str]:
    return random.sample(AUTHORS, k=n)


def random_genres(n: int) -> List[str]:
    return random.sample(GENRES, k=n)


# ----------------------------------------------------------
# Build a synthetic dataset of ~500 examples
# ----------------------------------------------------------

def generate_synthetic_dataset(num_examples=500) -> List[Dict[str, Any]]:
    dataset = []

    for i in range(num_examples):
        # Random choice: simple / medium / complex structure
        style = random.choice(["simple", "medium", "complex"])

        liked = random_books(random.randint(1, 3))
        disliked = random_books(random.randint(0, 2))
        excluded = random_books(random.randint(0, 1))
        liked_g = random_genres(random.randint(0, 2))
        disliked_g = random_genres(random.randint(0, 1))
        liked_a = random_authors(random.randint(0, 2))
        disliked_a = random_authors(random.randint(0, 1))

        # Build ground truth
        ground = {
            "liked_books": [{"title": b, "rating": 5.0} for b in liked],
            "disliked_books": [{"title": b, "rating": 1.0} for b in disliked],
            "excluded_books": [{"title": b} for b in excluded],
            "liked_genres": list(liked_g),
            "disliked_genres": list(disliked_g),
            "liked_authors": list(liked_a),
            "disliked_authors": list(disliked_a),
            "num_recommendations": random.choice([3, 5, 10]),
        }

        # Construct natural-language prompt
        if style == "simple":
            msg = (
                f"I loved {', '.join(liked)}. "
                f"I disliked {', '.join(disliked)}. "
                f"Don't recommend {', '.join(excluded)} again. "
                f"I like {', '.join(liked_g)}. "
                f"I don't want {', '.join(disliked_g)}. "
                f"My favorite authors are {', '.join(liked_a)}. "
                f"I dislike authors {', '.join(disliked_a)}. "
                f"Please give me {ground['num_recommendations']} recommendations."
            )
        elif style == "medium":
            msg = (
                f"I really enjoyed {liked[0]}. "
                f"I think I hated {disliked[0] if disliked else ''}. "
                f"No more {excluded[0] if excluded else ''} please. "
                f"I enjoy genres like {', '.join(liked_g)}. "
                f"But avoid {', '.join(disliked_g)}. "
                f"Authors I'm into: {', '.join(liked_a)}. "
                f"Please recommend around {ground['num_recommendations']} books."
            )
        else:  # complex + includes fake rec list
            recs = random_books(4)
            formatted_list = "\n".join(
                f" number={idx} |\n {title} |\n -----------"
                for idx, title in enumerate(recs)
            )
            selection = random.choice([0, 1, 2, 3])
            msg = (
                f"Here is a list:\n\n{formatted_list}\n\n"
                f"I liked #{selection} which is {recs[selection]}. "
                f"I enjoy {', '.join(liked_g)} genre. "
                f"Give me {ground['num_recommendations']} recs."
            )

        dataset.append({"id": i, "message": msg, "ground_truth": ground})

    return dataset


# ----------------------------------------------------------
# Metric computation
# ----------------------------------------------------------

def titles_set(lst: List[Dict[str, Any]]) -> set:
    return {str(d["title"]).lower().strip() for d in lst}


def str_set(lst: List[str]) -> set:
    return {str(x).lower().strip() for x in lst}


def compute_f1(pred_set: set, true_set: set):
    # Perfect empty match
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1



# ----------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------

def evaluate_planner(num_examples=500):
    dataset = generate_synthetic_dataset(num_examples)

    results = []

    for example in dataset:
        msg = example["message"]
        truth = example["ground_truth"]

        try:
            pred = call_ollama_planner(msg)
            valid_json = 1
        except Exception as e:
            print(f"JSON ERROR on id={example['id']}: {e}")
            valid_json = 0
            pred = {k: [] for k in truth}
            pred["num_recommendations"] = 0

        # Extract sets
        prec_l, rec_l, f1_l = compute_f1(
            titles_set(pred["liked_books"]),
            titles_set(truth["liked_books"]),
        )
        prec_d, rec_d, f1_d = compute_f1(
            titles_set(pred["disliked_books"]),
            titles_set(truth["disliked_books"]),
        )
        prec_e, rec_e, f1_e = compute_f1(
            titles_set(pred["excluded_books"]),
            titles_set(truth["excluded_books"]),
        )
        prec_lg, rec_lg, f1_lg = compute_f1(
            str_set(pred["liked_genres"]),
            str_set(truth["liked_genres"]),
        )
        prec_dg, rec_dg, f1_dg = compute_f1(
            str_set(pred["disliked_genres"]),
            str_set(truth["disliked_genres"]),
        )
        prec_la, rec_la, f1_la = compute_f1(
            str_set(pred["liked_authors"]),
            str_set(truth["liked_authors"]),
        )
        prec_da, rec_da, f1_da = compute_f1(
            str_set(pred["disliked_authors"]),
            str_set(truth["disliked_authors"]),
        )
        
        # Compute sets for hallucination detection (FP = hallucinations)
        pred_liked = titles_set(pred["liked_books"])
        true_liked = titles_set(truth["liked_books"])
        halluc_liked = len(pred_liked - true_liked)

        pred_disliked = titles_set(pred["disliked_books"])
        true_disliked = titles_set(truth["disliked_books"])
        halluc_disliked = len(pred_disliked - true_disliked)

        pred_excluded = titles_set(pred["excluded_books"])
        true_excluded = titles_set(truth["excluded_books"])
        halluc_excluded = len(pred_excluded - true_excluded)

        pred_liked_genres = str_set(pred["liked_genres"])
        true_liked_genres = str_set(truth["liked_genres"])
        halluc_liked_genres = len(pred_liked_genres - true_liked_genres)

        pred_disliked_genres = str_set(pred["disliked_genres"])
        true_disliked_genres = str_set(truth["disliked_genres"])
        halluc_disliked_genres = len(pred_disliked_genres - true_disliked_genres)

        pred_liked_authors = str_set(pred["liked_authors"])
        true_liked_authors = str_set(truth["liked_authors"])
        halluc_liked_authors = len(pred_liked_authors - true_liked_authors)

        pred_disliked_authors = str_set(pred["disliked_authors"])
        true_disliked_authors = str_set(truth["disliked_authors"])
        halluc_disliked_authors = len(pred_disliked_authors - true_disliked_authors)

        num_match = int(pred["num_recommendations"] == truth["num_recommendations"])

        results.append({
            "id": example["id"],
            "valid_json": valid_json,
            "f1_liked": f1_l,
            "f1_disliked": f1_d,
            "f1_excluded": f1_e,
            "f1_liked_genres": f1_lg,
            "f1_disliked_genres": f1_dg,
            "f1_liked_authors": f1_la,
            "f1_disliked_authors": f1_da,
            "num_match": num_match,
            "halluc_liked": halluc_liked,
            "halluc_disliked": halluc_disliked,
            "halluc_excluded": halluc_excluded,
            "halluc_liked_genres": halluc_liked_genres,
            "halluc_disliked_genres": halluc_disliked_genres,
            "halluc_liked_authors": halluc_liked_authors,
            "halluc_disliked_authors": halluc_disliked_authors,

        })

    # Print summary
    print("=== SUMMARY ===")
    def avg(key): return sum(r[key] for r in results) / len(results)

    print("JSON validity:", avg("valid_json"))
    print("Liked_books F1:", avg("f1_liked"))
    print("Disliked_books F1:", avg("f1_disliked"))
    print("Excluded_books F1:", avg("f1_excluded"))
    print("Liked_genres F1:", avg("f1_liked_genres"))
    print("Disliked_genres F1:", avg("f1_disliked_genres"))
    print("Liked_authors F1:", avg("f1_liked_authors"))
    print("Disliked_authors F1:", avg("f1_disliked_authors"))
    print("Num_recommendations exact match:", avg("num_match"))

    # Save to CSV
    with open("planner_eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\nFull results saved to planner_eval_results.csv")


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------

if __name__ == "__main__":
    evaluate_planner(100)
