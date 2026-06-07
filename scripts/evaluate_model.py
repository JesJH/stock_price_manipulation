#!/usr/bin/env python3
"""
CLI: evaluate a trained pipeline on a holdout feature set.

Usage
-----
    python scripts/evaluate_model.py \\
        --model     outputs/models/fraud_classifier.joblib \\
        --features  data/processed/holdout_features.parquet \\
        --threshold auto

Options for --threshold:
  auto        : use the cost-optimal threshold stored in the saved pipeline
  <float>     : use a fixed threshold, e.g. 0.5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.pipeline.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved TrainingPipeline on holdout data"
    )
    parser.add_argument(
        "--model",
        default="outputs/models/fraud_classifier.joblib",
        help="Path to saved TrainingPipeline (.joblib)",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Parquet file with holdout feature matrix (must include 'label' column)",
    )
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Classification threshold: 'auto' (pipeline default) or a float",
    )
    parser.add_argument(
        "--fp-cost",  type=float, default=1.0, help="FP cost weight for threshold optimisation"
    )
    parser.add_argument(
        "--fn-cost",  type=float, default=5.0, help="FN cost weight for threshold optimisation"
    )
    parser.add_argument(
        "--save-report",
        default=None,
        help="Optional path to save the threshold summary CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    feat_path  = Path(args.features)

    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)
    if not feat_path.exists():
        print(f"ERROR: features not found at {feat_path}")
        sys.exit(1)

    print(f"Loading model from   {model_path}")
    pipeline = TrainingPipeline.load(model_path)

    print(f"Loading features from {feat_path}")
    holdout = pd.read_parquet(feat_path)

    if "label" not in holdout.columns:
        print("ERROR: holdout features must include a 'label' column (1=TP, 0=TN)")
        sys.exit(1)

    print(f"  Holdout rows: {len(holdout)} "
          f"({int(holdout['label'].sum())} TP, "
          f"{int((1-holdout['label']).sum())} TN)")

    probas   = pipeline.predict_proba(holdout)
    y_true   = holdout["label"].values
    y_proba  = probas.values

    # Determine threshold
    if args.threshold == "auto":
        threshold = pipeline.best_threshold_
        print(f"\nUsing pipeline's stored threshold: {threshold:.4f}")
    else:
        threshold = float(args.threshold)
        print(f"\nUsing fixed threshold: {threshold:.4f}")

    # Metrics at chosen threshold
    metrics = EvaluationMetrics.compute(y_true, y_proba, threshold, args.fp_cost, args.fn_cost)
    print("\n=== Holdout Evaluation ===")
    print(f"  Threshold:   {metrics['threshold']}")
    print(f"  Precision:   {metrics['precision']}")
    print(f"  Recall:      {metrics['recall']}")
    print(f"  F1:          {metrics['f1']}")
    print(f"  Specificity: {metrics['specificity']}")
    print(f"  ROC-AUC:     {metrics['roc_auc']}")
    print(f"  PR-AUC:      {metrics['pr_auc']}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")
    print(f"  Weighted cost: {metrics['cost']:.1f}")

    # Full threshold sweep
    print("\n=== Threshold Optimisation on Holdout ===")
    opt = ThresholdOptimizer(fp_cost=args.fp_cost, fn_cost=args.fn_cost)
    opt.fit(y_true, y_proba)
    print(opt.summary())

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        opt.results_df_.to_csv(report_path, index=False)
        print(f"\nThreshold report saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
