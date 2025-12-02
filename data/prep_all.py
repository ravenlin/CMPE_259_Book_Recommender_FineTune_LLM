# prep_all.py
#
# Run the full data preparation pipeline in one go:
#   1) data_prep.py              → builds books.parquet + interactions.parquet
#   2) build_authors_parquet.py  → builds authors.parquet
#   3) enrich_books_genres.py    → builds books_with_genres.parquet
#
# So you just run:
#     python prep_all.py
#
# and all the processed files are ready for training + teacher.

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable  # uses the same Python you're running this with

def run_step(cmd, desc):
    print(f"\n=== [STEP] {desc} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {desc} finished successfully.")

def main():
    root = Path(__file__).parent

    # 1) Base data prep: books + interactions (your existing script)
    run_step(
        [PYTHON, str(root / "data_prep.py")],
        "Base data prep (books + interactions)",
    )

    # 2) Build authors.parquet (from earlier build_authors_parquet.py)
    run_step(
        [PYTHON, str(root / "build_authors.py")],
        "Build authors.parquet",
    )

    # 3) Enrich books with genres (your existing enrich_books_genres.py)
    run_step(
        [PYTHON, str(root / "enrich_books_genres.py")],
        "Enrich books_with_genres.parquet",
    )

    print("\n🎉 All data prep steps completed. You should now have:")
    print("  - data/processed/books.parquet (or similar)")
    print("  - data/processed/interactions.parquet")
    print("  - data/processed/authors.parquet")
    print("  - data/processed/books_with_genres.parquet")

if __name__ == "__main__":
    main()
