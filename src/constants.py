from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_CV = 6
DEFAULT_BINS = 25

ALL_METRICS = (
    "ECE",
    "cwECE",
)
