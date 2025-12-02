# pseudo_user_teacher.py
#
# Pseudo-user teacher recommender using DESCRIPTION EMBEDDINGS.
#
# Query comes from planner in RAW TEXT:
#   liked_books:    [{"title": "...", "rating": 5.0}, ...]
#   disliked_books: [{"title": "...", "rating": 1.0}, ...]
#   excluded_books: [{"title": "..."}]          # explicitly hide from final recs
#   liked_genres:   ["mystery", ...]
#   disliked_genres:["fantasy", ...]
#   liked_authors:  ["Agatha Christie", ...]
#   disliked_authors:["Stephen King", ...]
#
# This script:
#   1) maps titles/author names -> internal IDs
#   2) builds a pseudo-user embedding from description embeddings of liked/disliked books
#   3) scores items by cosine similarity
#   4) returns:
#        {
#          "raw":      [book_dict, ...],   # top candidates (excluding liked/disliked families only)
#          "filtered": [book_dict, ...],   # after genre/author/excluded-title filtering with replacement
#        }

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable, Set, Any

import ast
import re
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

BOOKS_PATH = PROC_DIR / "books_with_genres.parquet"
AUTHORS_PATH = PROC_DIR / "authors.parquet"
DESC_EMB_PATH = MODEL_DIR / "desc_embeddings.npy"


# -----------------------------
# Low-level helpers
# -----------------------------

def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def canonical_main_title(title: str) -> str:
    """
    Canonicalize a title to a coarse "main title" for de-duplication:

      - lowercase
      - strip punctuation → spaces
      - collapse whitespace
      - drop stopwords: 'the', 'a', 'an', 'of'
      - return the FIRST non-stopword token as key

    So:
      "The Hobbit"
      "The Hobbit, or There and Back Again"
      "The Hobbit: Graphic Novel"
      → all map to "hobbit"
    """
    t = str(title).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    tokens = t.split()
    stopwords = {"the", "a", "an", "of"}
    for tok in tokens:
        if tok and tok not in stopwords:
            return tok

    return tokens[0] if tokens else ""


def parse_authors_value(val) -> List[int]:
    """
    Parse the 'authors' value into a list of integer author IDs.

    Handles:
      - list/tuple/ndarray of ints/strings/dicts
      - dicts with keys like 'author_id' or 'id'
      - stringified lists/dicts via ast.literal_eval
      - single id as int/str
    """

    # If it's a string, try to literal_eval it first
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            val = parsed
        except Exception:
            val = s  # leave as raw string

    def extract_id(obj) -> Optional[int]:
        # int-like
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        # string that might be an int
        if isinstance(obj, str):
            try:
                return int(obj)
            except Exception:
                return None
        # dict containing id
        if isinstance(obj, dict):
            for key in ["author_id", "id"]:
                if key in obj:
                    try:
                        return int(obj[key])
                    except Exception:
                        return None
        return None

    # list-like
    if isinstance(val, (list, tuple, set, np.ndarray)):
        ids: List[int] = []
        for item in val:
            aid = extract_id(item)
            if aid is not None:
                ids.append(aid)
        return ids

    # single object
    aid = extract_id(val)
    return [aid] if aid is not None else []


def load_authors_mapping() -> Dict[int, str]:
    """
    Load authors.parquet (if present) and build a mapping author_id -> author_name.

    We try:
      id_col in ['author_id', 'id']
      name_col in ['name', 'author_name']
    """
    if not AUTHORS_PATH.exists():
        print(f"[warn] {AUTHORS_PATH} not found; author filters will only use raw 'authors' text.")
        return {}

    print(f"[load] Loading authors from {AUTHORS_PATH}")
    df_auth = pd.read_parquet(AUTHORS_PATH)
    print("[info] authors shape:", df_auth.shape)

    cols = list(df_auth.columns)
    id_col = next((c for c in ["author_id", "id"] if c in cols), None)
    name_col = next((c for c in ["name", "author_name"] if c in cols), None)

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
# Teacher class
# -----------------------------

class DescEmbeddingTeacher:
    def __init__(self):
        print("[load] Loading books_with_genres...")
        self.df_books = pd.read_parquet(BOOKS_PATH)
        self.df_books["book_id"] = self.df_books["book_id"].astype(int)
        print("[info] books shape:", self.df_books.shape)

        print(f"[load] Loading description embeddings from {DESC_EMB_PATH}")
        self.item_embs = np.load(DESC_EMB_PATH)
        print("[info] item_embs shape:", self.item_embs.shape)

        if self.item_embs.shape[0] != len(self.df_books):
            raise ValueError(
                f"Row count mismatch: books={len(self.df_books)} vs "
                f"embeddings={self.item_embs.shape[0]}"
            )

        # Pre-normalize for cosine similarity
        self.item_embs_norm = normalize_rows(self.item_embs.astype(np.float32))

        # Normalized title + canonical main_title
        self.df_books["title_norm"] = (
            self.df_books["title"].astype(str).str.strip().str.lower()
        )
        self.df_books["main_title"] = self.df_books["title"].astype(str).apply(canonical_main_title)

        # Build title → indices map
        self.title_to_idxs: Dict[str, List[int]] = {}
        for idx, tnorm in enumerate(self.df_books["title_norm"].tolist()):
            self.title_to_idxs.setdefault(tnorm, []).append(idx)

        # Book id → index map
        self.book_id_to_idx: Dict[int, int] = {
            int(bid): i for i, bid in enumerate(self.df_books["book_id"].tolist())
        }

        # ---- Robust genres normalization ----
        def normalize_genres(v):
            if v is None:
                return []
            if isinstance(v, np.ndarray):
                return [str(x).strip() for x in v.tolist() if str(x).strip()]
            if isinstance(v, (list, tuple, set)):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return []
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple, set, np.ndarray)):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
                return [s]
            try:
                if pd.isna(v):
                    return []
            except Exception:
                pass
            return [str(v).strip()]

        if "genres" in self.df_books.columns:
            self.df_books["genres"] = self.df_books["genres"].apply(normalize_genres)
        else:
            self.df_books["genres"] = [[] for _ in range(len(self.df_books))]

        # ---- Authors: parse IDs from books['authors'] ----
        if "authors" in self.df_books.columns:
            self.df_books["authors_parsed"] = self.df_books["authors"].apply(parse_authors_value)
        else:
            self.df_books["authors_parsed"] = [[] for _ in range(len(self.df_books))]

        # ---- Map author_ids → names using authors.parquet if available ----
        author_map = load_authors_mapping()
        if author_map:
            def ids_to_names(ids: List[int]) -> List[str]:
                names: List[str] = []
                for aid in ids:
                    name = author_map.get(aid)
                    if name:
                        s = str(name).strip()
                        if s and s not in names:
                            names.append(s)
                return names

            self.df_books["author_names"] = self.df_books["authors_parsed"].apply(ids_to_names)
            total_names = int(self.df_books["author_names"].apply(len).sum())
            print(f"[authors] author_names filled from authors.parquet, total names={total_names}")
        else:
            self.df_books["author_names"] = [[] for _ in range(len(self.df_books))]

        # If still empty everywhere, fall back to textual authors column
        if int(self.df_books["author_names"].apply(len).sum()) == 0 and "authors" in self.df_books.columns:
            print("[authors] author_names empty; falling back to books['authors'] text.")

            def authors_text_to_names(v) -> List[str]:
                if v is None:
                    return []
                if isinstance(v, (list, tuple, set)):
                    return [str(x).strip() for x in v if str(x).strip()]
                s = str(v).strip()
                if not s:
                    return []
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple, set, np.ndarray)):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
                parts = [p.strip() for p in s.split(",") if p.strip()]
                return parts or [s]

            self.df_books["author_names"] = self.df_books["authors"].apply(authors_text_to_names)
            total_names = int(self.df_books["author_names"].apply(len).sum())
            print(f"[authors] filled author_names from books['authors'] text, total names={total_names}")

        print("[init] Teacher initialized with titles, genres, authors, and embeddings.")

    # --------- lookup / aggregation helpers ---------

    def _find_title_indices(self, title: str) -> List[int]:
        key = title.strip().lower()
        return self.title_to_idxs.get(key, [])

    def _expand_indices_and_weights(
        self,
        books: List[Dict[str, Any]],  # [{"title": str, "rating": float}, ...]
    ) -> Tuple[List[int], List[float], Set[str]]:
        """
        For each book dict:
          - resolve title → one or more indices (editions)
          - repeat its rating weight for each index
        Returns:
          indices, weights, main_titles_set (for filtering later)
        """
        all_indices: List[int] = []
        all_weights: List[float] = []
        main_titles: Set[str] = set()

        for b in books:
            title = str(b.get("title", "")).strip()
            if not title:
                continue
            rating = float(b.get("rating", 1.0))
            idxs = self._find_title_indices(title)
            if not idxs:
                print(f"[warn] title from query not found in corpus: '{title}'")
                continue

            # Expand indices & weights
            all_indices.extend(idxs)
            all_weights.extend([rating] * len(idxs))

            # Track main_title for family-level filtering
            mt = canonical_main_title(title)
            if mt:
                main_titles.add(mt)

        # Deduplicate indices but preserve sum of weights for duplicates
        if not all_indices:
            return [], [], main_titles

        # Aggregate weights per index
        weight_map: Dict[int, float] = {}
        for idx, w in zip(all_indices, all_weights):
            weight_map[idx] = weight_map.get(idx, 0.0) + w

        indices = sorted(weight_map.keys())
        weights = [weight_map[i] for i in indices]
        return indices, weights, main_titles

    def _expand_indices_titles_only(
        self,
        books: List[Dict[str, Any]],  # [{"title": str, ...}, ...]
    ) -> Tuple[List[int], Set[str]]:
        """
        Resolve books (by 'title') to:
          - indices in df_books
          - set of main_title keys for family-level exclusion.

        Used for excluded_books: they don't affect embedding, just filtering.
        """
        indices: List[int] = []
        main_titles: Set[str] = set()

        for b in books:
            title = str(b.get("title", "")).strip()
            if not title:
                continue
            idxs = self._find_title_indices(title)
            if not idxs:
                print(f"[warn] excluded title not found in corpus: '{title}'")
                continue
            indices.extend(idxs)
            mt = canonical_main_title(title)
            if mt:
                main_titles.add(mt)

        indices = sorted(set(indices))
        return indices, main_titles

    def _build_user_embedding_from_query(
        self,
        liked_books: List[Dict[str, Any]],
        disliked_books: List[Dict[str, Any]],
        alpha: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], List[int], List[int], Set[str], Set[str]]:
        """
        Build user embedding:

          u_raw = weighted_mean(liked_embs) - alpha * weighted_mean(disliked_embs)
          u = normalize(u_raw)

        Weights come from the 'rating' field provided by the planner.

        Returns:
          user_vec,
          liked_indices,
          disliked_indices,
          liked_main_titles_set,
          disliked_main_titles_set
        """
        liked_idx, liked_w, liked_mts = self._expand_indices_and_weights(liked_books)
        disliked_idx, disliked_w, disliked_mts = self._expand_indices_and_weights(disliked_books)

        if not liked_idx and not disliked_idx:
            print("[warn] No liked or disliked books matched; cannot build user embedding.")
            return None, [], [], liked_mts, disliked_mts

        def weighted_mean(indices: List[int], weights: List[float]) -> Optional[np.ndarray]:
            if not indices:
                return None
            embs = self.item_embs_norm[indices]  # (n, d)
            w = np.asarray(weights, dtype=np.float32)
            if w.sum() <= 0:
                return embs.mean(axis=0)
            w = w / w.sum()
            return (w[:, None] * embs).sum(axis=0)

        like_vec = weighted_mean(liked_idx, liked_w)
        dislike_vec = weighted_mean(disliked_idx, disliked_w)

        if like_vec is None and dislike_vec is not None:
            u_raw = -alpha * dislike_vec
        elif like_vec is not None and dislike_vec is None:
            u_raw = like_vec
        else:
            u_raw = like_vec - alpha * dislike_vec  # liked – alpha * disliked

        u_raw = u_raw.reshape(1, -1)
        u = normalize_rows(u_raw)[0]
        print(
            f"[user] built user embedding from query: "
            f"{len(liked_idx)} liked idx (|liked_books|={len(liked_books)}), "
            f"{len(disliked_idx)} disliked idx (|disliked_books|={len(disliked_books)}), "
            f"alpha={alpha}"
        )
        return u, liked_idx, disliked_idx, liked_mts, disliked_mts

    # --------- filtering helpers ---------

    def _build_disliked_genre_mask(self, disliked_genres: Iterable[str]) -> np.ndarray:
        if not disliked_genres:
            return np.zeros(len(self.df_books), dtype=bool)

        disliked_set: Set[str] = {g.strip().lower() for g in disliked_genres if g.strip()}
        if not disliked_set:
            return np.zeros(len(self.df_books), dtype=bool)

        def has_disliked(gen_list) -> bool:
            for g in gen_list:
                try:
                    if str(g).strip().lower() in disliked_set:
                        return True
                except Exception:
                    continue
            return False

        mask = self.df_books["genres"].apply(has_disliked).to_numpy(dtype=bool)
        return mask

    def _build_disliked_author_mask(self, disliked_authors: Iterable[str]) -> np.ndarray:
        """
        disliked_authors: iterable of author NAMES (strings).
        """
        if not disliked_authors:
            return np.zeros(len(self.df_books), dtype=bool)

        disliked_set: Set[str] = {a.strip().lower() for a in disliked_authors if a.strip()}
        if not disliked_set:
            return np.zeros(len(self.df_books), dtype=bool)

        def has_disliked_author(names: List[str]) -> bool:
            for n in names:
                if str(n).strip().lower() in disliked_set:
                    return True
            return False

        mask = self.df_books["author_names"].apply(has_disliked_author).to_numpy(dtype=bool)
        return mask

    # --------- public: title search helper ---------

    def search_titles(self, query: str, top_n: int = 20) -> pd.DataFrame:
        q = query.strip().lower()
        mask = self.df_books["title"].astype(str).str.lower().str.contains(q)
        res = self.df_books[mask].copy()
        if res.empty:
            print(f"[search] No titles matched query: '{query}'")
            return res

        res = res[["book_id", "title", "average_rating"]][:top_n]
        print(f"[search] Found {len(res)} match(es) for '{query}' (showing up to {top_n}):")
        for _, row in res.iterrows():
            print(f"  {row['book_id']:8d} | {row['title'][:80]:80s} | rating={row['average_rating']:.2f}")
        return res

    # --------- main: planner-style query API ---------

    def recommend_from_query(
        self,
        query: Dict[str, Any],
        top_raw: int = 50,
        top_filtered: int = 20,
        alpha: float = 1.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        query dict structure:

        {
          "liked_books":     [{"title": str, "rating": float}, ...],
          "disliked_books":  [{"title": str, "rating": float}, ...],
          "excluded_books":  [{"title": str}, ...],      # hard exclusion, no effect on embedding
          "liked_genres":    [str, ...],
          "disliked_genres": [str, ...],
          "liked_authors":   [str, ...],
          "disliked_authors":[str, ...],
        }

        Returns:
          {
            "raw":      [book_dict, ...],  # top_raw (no excluded filter here)
            "filtered": [book_dict, ...],  # after genre/author/excluded filters, up to top_filtered
          }
        """
        liked_books = query.get("liked_books", []) or []
        disliked_books = query.get("disliked_books", []) or []
        excluded_books = query.get("excluded_books", []) or []

        liked_genres = query.get("liked_genres", []) or []
        disliked_genres = query.get("disliked_genres", []) or []
        liked_authors = query.get("liked_authors", []) or []
        disliked_authors = query.get("disliked_authors", []) or []

        # Build user embedding from positive/negative examples
        user_vec, liked_idx, disliked_idx, liked_mts, disliked_mts = self._build_user_embedding_from_query(
            liked_books=liked_books,
            disliked_books=disliked_books,
            alpha=alpha,
        )
        if user_vec is None:
            return {"raw": [], "filtered": []}

        # Excluded books: indices + main_title families (for *filtering* only)
        excluded_idx, excluded_mts = self._expand_indices_titles_only(excluded_books)
        if excluded_idx:
            print(f"[exclude] {len(excluded_idx)} indices derived from {len(excluded_books)} excluded_books entries.")

        # Score all items (cosine similarity on normalized vectors)
        scores = self.item_embs_norm @ user_vec  # (num_items,)

        # Exclude explicit liked/disliked book indices for raw pool
        mask_exclude_liked_idx = np.zeros_like(scores, dtype=bool)
        if liked_idx:
            mask_exclude_liked_idx[liked_idx] = True

        mask_exclude_disliked_idx = np.zeros_like(scores, dtype=bool)
        if disliked_idx:
            mask_exclude_disliked_idx[disliked_idx] = True

        # Exclude entire title families by main_title *only* for liked/disliked books
        raw_excluded_main_titles = set()
        raw_excluded_main_titles.update(liked_mts)
        raw_excluded_main_titles.update(disliked_mts)
        mask_exclude_main = self.df_books["main_title"].isin(raw_excluded_main_titles).to_numpy(bool)

        # Genre/author masks for filtering step
        mask_disliked_genres = self._build_disliked_genre_mask(disliked_genres)
        mask_disliked_authors = self._build_disliked_author_mask(disliked_authors)

        # Overall hard exclusion for candidate *pool* (no exact liked/disliked families)
        # NOTE: excluded_books are NOT removed here; they will be removed only in filtered step.
        mask_exclude_raw = (
            mask_exclude_liked_idx
            | mask_exclude_disliked_idx
            | mask_exclude_main
        )

        # Sort all candidates by score descending
        candidate_indices = np.argsort(-scores)

        # Decide how deep to look for candidates to allow replacement after filtering
        pool_size = max(top_raw * 3, top_filtered * 5)

        # Collect candidate pool with de-dup by main_title
        raw_selected: List[int] = []
        seen_main_titles: Set[str] = set()

        for idx in candidate_indices:
            if len(raw_selected) >= pool_size:
                break
            if mask_exclude_raw[idx]:
                continue
            mt = self.df_books.iloc[idx]["main_title"]
            if mt in seen_main_titles:
                continue
            seen_main_titles.add(mt)
            raw_selected.append(idx)

        # Precompute exclusion info for filtering step
        mask_excluded_idx = np.zeros_like(scores, dtype=bool)
        if excluded_idx:
            mask_excluded_idx[excluded_idx] = True
        excluded_main_titles = set(excluded_mts)

        def row_to_dict(row, score_val: float) -> Dict[str, Any]:
            return {
                "book_id": int(row["book_id"]),
                "title": str(row["title"]),
                "average_rating": float(row.get("average_rating", np.nan)),
                "genres": list(row.get("genres", [])),
                "authors": list(row.get("author_names", [])),
                "score": float(score_val),
            }

        # RAW list = first top_raw from candidate pool (excluded_books are allowed here)
        raw_selected_for_output = raw_selected[:top_raw]
        raw_list: List[Dict[str, Any]] = []
        for idx in raw_selected_for_output:
            row = self.df_books.iloc[idx]
            raw_list.append(row_to_dict(row, scores[idx]))

        # FILTERED list = from the same pool but enforce disliked_genres/authors + excluded_books
        filtered_list: List[Dict[str, Any]] = []
        for idx in raw_selected:
            if len(filtered_list) >= top_filtered:
                break

            # Exclude genres/authors
            if mask_disliked_genres[idx] or mask_disliked_authors[idx]:
                continue

            # Exclude explicitly excluded indices & their main_title families
            if mask_excluded_idx[idx]:
                continue
            mt = self.df_books.iloc[idx]["main_title"]
            if mt in excluded_main_titles:
                continue

            row = self.df_books.iloc[idx]
            filtered_list.append(row_to_dict(row, scores[idx]))

        return {
            "raw": raw_list,
            "filtered": filtered_list,
        }


# -----------------------------
# Quick sanity test when run directly
# -----------------------------

def main():
    teacher = DescEmbeddingTeacher()

    # Example "planner" query
    query = {
        "liked_books": [
            {"title": "The Hobbit", "rating": 5.0},
        ],
        "disliked_books": [],
        "excluded_books": [
            {"title": "A Tolkien Bestiary"},
        ],
        "liked_genres": [],
        "disliked_genres": ["young adult"],
        "liked_authors": [],
        "disliked_authors": ["J.R.R. Tolkien"],
    }

    result = teacher.recommend_from_query(query, top_raw=15, top_filtered=20, alpha=1.0)

    print("\n================= RAW (top 15) =================")
    for b in result["raw"]:
        print(
            f"{b['book_id']:8d} |\n {b['title'][:70]:70s} |\n "
            f"rating={b['average_rating']:.2f} |\n "
            f"genres={b['genres']} |\n authors={b['authors']} |\n score={b['score']:.4f} \n -----------"
        )

    print("\n================= FILTERED (top 20) =================")
    for b in result["filtered"]:
        print(
            f"{b['book_id']:8d} |\n {b['title'][:70]:70s} |\n "
            f"rating={b['average_rating']:.2f} |\n "
            f"genres={b['genres']} |\n authors={b['authors']} |\n score={b['score']:.4f} \n -----------"
        )

    print("\n[done] pseudo_user_teacher sanity test.")


if __name__ == "__main__":
    main()
