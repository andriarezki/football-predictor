"""python live.py <out.html>  -> refresh data, predict upcoming fixtures, write the live page."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
from src import live

live.run(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "live.html"))
print("ok")
