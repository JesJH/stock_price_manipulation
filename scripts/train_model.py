#!/usr/bin/env python3
"""
CLI: run the full training pipeline (LOO-CV + final model fit).

Usage
-----
    python scripts/train_model.py \\
        --tp-features   data/processed/tp_features.parquet \\
        --tn-features   data/processed/tn_features.parquet \\
        --output        outputs/models/fraud_classifier.joblib \\
        --method        distance_ranked \\
        --tn-tp-ratio   5 \\
        --max-depth     4

Prerequisites
-------------
Run scripts/download_data.py first to populate data/raw/, then run
feature engineering (notebook section 2) to produce the feature parquets.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_transformer import FeatureTransformer
from src.pipeline.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the pump-and-dump detection pipeline"
    )
    parser.add_argument(
        "--tp-features",
        default="data/processed/tp_features.parquet",
        help="Parquet file with TP feature matrix",
    )
    parser.add_argument(
        "--tn-features",
        default="data/processed/tn_features.parquet",
        help="Parquet file with TN candidate feature matrix",
    )
    parser.add_argument(
        "--output",
        default="outputs/models/fraud_classifier.joblib",
        help="Where to save the trained pipeline",
    )
    parser.add_argument(
        "--method",
        choices=["random", "distance_ranked", "stratified"],
        default="distance_ranked",
        help="CentroidSelector TN sampling method",
    )
    parser.add_argument("--tn-tp-ratio", type=int,   default=5)
    parser.add_argument("--max-depth",   type=int,   default=4)
    parser.add_argument("--min-leaf",    type=int,   default=2)
    parser.add_argument("--fp-cost",     type=float, default=1.0)
    parser.add_argument("--fn-cost",     type=float, default=5.0)
    parser.add_argument("--random-state", type=int,  default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tp_path = Path(args.tp_features)
    tn_path = Path(args.tn_features)

    if not tp_path.exists():
        print(f"ERROR: TP features not found at {tp_path}")
        print("  Run feature engineering first (notebook section 2).")
        sys.exit(1)

    if not tn_path.exists():
        print(f"ERROR: TN features not found at {tn_path}")
        print("  Run scripts/download_data.py --source tn then feature engineering.")
        sys.exit(1)

    print(f"Loading TP features from  {tp_path}")
    tp_features = pd.read_parquet(tp_path)
    print(f"Loading TN features from  {tn_path}")
    tn_features = pd.read_parquet(tn_path)

    print(f"  TP rows: {len(tp_features)}   TN candidate rows: {len(tn_features)}")

    # Determine feature columns: all numeric columns except identifiers / label
    exclude = {"ticker", "d_date", "label"}
    transformer = FeatureTransformer()
    feature_cols = [
        c for c in transformer.feature_names
        if c in tp_features.columns and c not in exclude
    ]
    print(f"  Feature columns: {len(feature_cols)}")

    pipeline = TrainingPipeline(
        centroid_method=args.method,
        tn_tp_ratio=args.tn_tp_ratio,
        dt_max_depth=args.max_depth,
        dt_min_samples_leaf=args.min_leaf,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        random_state=args.random_state,
    )
    pipeline.fit(tp_features, tn_features, feature_cols)

    # Save
    out_path = Path(args.output)
    pipeline.save(out_path)

    # Print top features
    if pipeline.final_classifier_ is not None:
        print("\nTop 10 features by importance:")
        print(pipeline.final_classifier_.top_features(10).to_string(index=False))

    # Save OOF predictions
    if pipeline.oof_predictions_ is not None:
        oof_path = out_path.parent / "oof_predictions.csv"
        pipeline.oof_predictions_.to_csv(oof_path, index=False)
        print(f"\nOOF predictions saved to {oof_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
