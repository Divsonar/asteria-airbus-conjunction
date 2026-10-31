"""Train the conjunction triage model and export all supplied test predictions."""
import argparse
from pathlib import Path

from src.pipeline import ARTIFACTS, DATA_DIR, train_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train_data.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_data.csv")
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    train_pipeline(args.train, args.test, args.output)
