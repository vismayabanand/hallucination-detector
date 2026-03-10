from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT_DIR / "results"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15