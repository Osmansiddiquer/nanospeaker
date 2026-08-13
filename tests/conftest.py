import sys
from pathlib import Path

# pytest only puts tests/ on sys.path; add the repo root so `src.model...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
