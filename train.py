"""Train the updated Airbus final-tier forecast and export test predictions."""
import argparse
from pathlib import Path

from src.pipeline import ARTIFACTS, DATA_DIR, LABELS_PATH, train_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train_data.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_data.csv")
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args()
    train_pipeline(args.train, args.test, args.output, labels_path=args.labels)
