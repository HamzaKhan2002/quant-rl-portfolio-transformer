"""conftest.py — add repo root to sys.path so `src.*` imports work in tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
