"""
End-to-end training pipeline with LOO-CV and temporal leakage prevention.

Pipeline per fold
-----------------
1. CentroidSelector  — select TNs from the training pool
2. AnomalyScorer     — fit Isolation Forest on training TNs; score all rows
3. FraudClassifier   — fit Decision Tree on training set (with IF score)
4. Evaluate          — predict on held-out TP + its matched TN window

Leakage prevention
------------------
TN candidates are tagged with the TP D-date of the window they were computed
in (see downloader.download_tn_batch). In each LOO-CV fold:

  - Held-out TP (TP_i, d_date = D_i) is excluded from training.
  - All TN rows with d_date == D_i are also excluded from training and form
    the negative test set alongside TP_i.

This ensures no temporal signal from TP_i's calendar window leaks into the
training fold via its matched TN candidates.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.models.centroid_selector import CentroidSelector
from src.models.classifier import FraudClassifier
from src.models.isolation_forest import AnomalyScorer


class TrainingPipeline:
    """
    LOO-CV pipeline: centroid TN selection → Isolation Forest → Decision Tree.

    Parameters
    ----------
    centroid_method   : TN sampling strategy ('random', 'distance_ranked', 'stratified')
    tn_tp_ratio       : target TNs per TP centroid
    dt_max_depth      : Decision Tree max depth
    dt_min_samples_leaf: Decision Tree minimum samples per leaf
    if_n_estimators   : number of trees in the Isolation Forest
    fp_cost           : relative cost of a false positive for threshold optimisation
    fn_cost           : relative cost of a false negative for threshold optimisation
    random_state      : reproducibility seed
    """

    def __init__(
        self,
        centroid_method: str = "distance_ranked",
        tn_tp_ratio: int = 5,
        dt_max_depth: int | None = 4,
        dt_min_samples_leaf: int = 2,
        if_n_estimators: int = 100,
        fp_cost: float = 1.0,
        fn_cost: float = 5.0,
        random_state: int = 42,
    ):
        self.centroid_method      = centroid_method
        self.tn_tp_ratio          = tn_tp_ratio
        self.dt_max_depth         = dt_max_depth
        self.dt_min_samples_leaf  = dt_min_samples_leaf
        self.if_n_estimators      = if_n_estimators
        self.fp_cost              = fp_cost
        self.fn_cost              = fn_cost
        self.random_state         = random_state

        # Populated after fit()
        self.oof_predictions_: pd.DataFrame | None = None
        self.cv_metrics_: dict | None = None
        self.best_threshold_: float = 0.5
        self.final_classifier_: FraudClassifier | None = None
        self.final_centroid_selector_: CentroidSelector | None = None
        self.final_anomaly_scorer_: AnomalyScorer | None = None
        self.feature_cols_: list[str] = []
        self.dt_feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        tp_features: pd.DataFrame,
        tn_candidate_features: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label",
    ) -> "TrainingPipeline":
        """
        Run LOO-CV to evaluate the pipeline, then fit a final model on all data.

        Parameters
        ----------
        tp_features            : feature matrix for confirmed TP securities.
                                 Must have columns: ticker, d_date, label (=1),
                                 plus all columns in feature_cols.
        tn_candidate_features  : feature matrix for TN candidates.
                                 Must have columns: ticker, d_date, label (=0),
                                 plus all columns in feature_cols.
                                 d_date == TP's D-date the window was computed for.
        feature_cols           : numeric feature columns to pass to models
        label_col              : target column name

        After fitting:
          oof_predictions_       — OOF predicted probabilities + true labels
          cv_metrics_            — summary metrics from LOO-CV
          best_threshold_        — cost-optimal threshold from OOF probabilities
          final_classifier_      — Decision Tree fit on all data
          final_centroid_selector_, final_anomaly_scorer_ — fit on all data
        """
        self.feature_cols_ = feature_cols

        print("=== LOO-CV ===")
        self.oof_predictions_ = self._run_loo_cv(
            tp_features, tn_candidate_features, feature_cols, label_col
        )

        # Evaluate OOF predictions
        oof_tp_rows = self.oof_predictions_[self.oof_predictions_["true_label"] == 1]
        if len(oof_tp_rows) > 0 and self.oof_predictions_["true_label"].nunique() > 1:
            opt = ThresholdOptimizer(fp_cost=self.fp_cost, fn_cost=self.fn_cost)
            opt.fit(
                self.oof_predictions_["true_label"].values,
                self.oof_predictions_["pred_proba"].values,
            )
            self.best_threshold_ = opt.best_threshold_
            self.cv_metrics_     = opt.optimal_metrics_
            print(f"\n  OOF {opt.summary()}")
        else:
            warnings.warn("Not enough label diversity in OOF predictions for metric computation.")

        print("\n=== Fitting final model on all data ===")
        self._fit_final_model(tp_features, tn_candidate_features, feature_cols, label_col)
        print("  Done.")
        return self

    def predict_proba(
        self,
        features_df: pd.DataFrame,
        use_threshold: bool = False,
    ) -> pd.Series:
        """
        Score new securities using the fitted final model.

        Returns
        -------
        Series of fraud probabilities (index preserved from features_df).
        If use_threshold=True, returns binary 0/1 predictions.
        """
        self._check_fitted()
        scored = self.final_anomaly_scorer_.add_score_column(features_df)
        probas = self.final_classifier_.predict_proba(scored)[: , 1]
        result = pd.Series(probas, index=features_df.index, name="fraud_proba")
        if use_threshold:
            return (result >= self.best_threshold_).astype(int).rename("fraud_flag")
        return result

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  TrainingPipeline saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingPipeline":
        return joblib.load(Path(path))

    # ------------------------------------------------------------------
    # LOO-CV
    # ------------------------------------------------------------------

    def _run_loo_cv(
        self,
        tp_features: pd.DataFrame,
        tn_candidate_features: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
    ) -> pd.DataFrame:
        """
        Leave-one-out cross-validation over TP observations.

        Returns DataFrame with columns:
          ticker, d_date, true_label, pred_proba, fold_held_out_ticker
        """
        records: list[dict] = []
        tp_groups = tp_features[["ticker", "d_date"]].drop_duplicates()
        n_folds   = len(tp_groups)

        for fold_i, (_, tp_row) in enumerate(tp_groups.iterrows(), 1):
            held_ticker = tp_row["ticker"]
            held_d_date = pd.Timestamp(tp_row["d_date"])

            print(f"  Fold {fold_i}/{n_folds}: holding out {held_ticker} (D={held_d_date.date()})")

            # --- Split by temporal group ---
            tp_train = tp_features[tp_features["d_date"] != held_d_date].copy()
            tp_test  = tp_features[tp_features["d_date"] == held_d_date].copy()

            # All TN candidates from held-out TP's window are excluded from training
            tn_train = tn_candidate_features[
                tn_candidate_features["d_date"] != held_d_date
            ].copy()
            tn_test = tn_candidate_features[
                tn_candidate_features["d_date"] == held_d_date
            ].copy()

            if tp_train.empty or tn_train.empty:
                print(f"    Skipping: insufficient training data in fold {fold_i}")
                continue

            # --- Step 1: CentroidSelector ---
            selector = CentroidSelector(
                method=self.centroid_method,
                tn_tp_ratio=self.tn_tp_ratio,
                random_state=self.random_state,
            )
            selected_tn_train = selector.fit_select(tp_train, tn_train, feature_cols)
            if selected_tn_train.empty:
                print(f"    Skipping: CentroidSelector returned no TNs in fold {fold_i}")
                continue

            train_set = pd.concat([tp_train, selected_tn_train], ignore_index=True)

            # --- Step 2: AnomalyScorer (fit on training TNs, score all) ---
            scorer = AnomalyScorer(
                n_estimators=self.if_n_estimators,
                random_state=self.random_state,
            )
            train_scores = scorer.fit_score(selected_tn_train, train_set, feature_cols)
            train_set    = scorer.add_score_column(train_set, train_scores)

            test_set = pd.concat([tp_test, tn_test], ignore_index=True)
            if test_set.empty:
                continue
            test_set = scorer.add_score_column(test_set)

            dt_feature_cols = feature_cols + [scorer.score_col]

            # --- Step 3: FraudClassifier ---
            clf = FraudClassifier(
                max_depth=self.dt_max_depth,
                min_samples_leaf=self.dt_min_samples_leaf,
                random_state=self.random_state,
            )
            clf.fit(train_set, train_set[label_col], dt_feature_cols)

            probas = clf.predict_proba(test_set)[: , 1]

            for idx, (_, row) in enumerate(test_set.iterrows()):
                records.append({
                    "ticker":              row["ticker"],
                    "d_date":              row["d_date"],
                    "true_label":          int(row[label_col]),
                    "pred_proba":          float(probas[idx]),
                    "fold_held_out_ticker": held_ticker,
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Final model
    # ------------------------------------------------------------------

    def _fit_final_model(
        self,
        tp_features: pd.DataFrame,
        tn_candidate_features: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
    ) -> None:
        """Fit final model on all available data (no held-out fold)."""

        # CentroidSelector on all TPs
        self.final_centroid_selector_ = CentroidSelector(
            method=self.centroid_method,
            tn_tp_ratio=self.tn_tp_ratio,
            random_state=self.random_state,
        )
        selected_tn = self.final_centroid_selector_.fit_select(
            tp_features, tn_candidate_features, feature_cols
        )
        print(f"  CentroidSelector selected {len(selected_tn)} TNs from "
              f"{len(tn_candidate_features)} candidates")

        all_data = pd.concat([tp_features, selected_tn], ignore_index=True)

        # AnomalyScorer on all selected TNs
        self.final_anomaly_scorer_ = AnomalyScorer(
            n_estimators=self.if_n_estimators,
            random_state=self.random_state,
        )
        scores   = self.final_anomaly_scorer_.fit_score(selected_tn, all_data, feature_cols)
        all_data = self.final_anomaly_scorer_.add_score_column(all_data, scores)

        self.dt_feature_cols_ = feature_cols + [self.final_anomaly_scorer_.score_col]

        # FraudClassifier
        self.final_classifier_ = FraudClassifier(
            max_depth=self.dt_max_depth,
            min_samples_leaf=self.dt_min_samples_leaf,
            random_state=self.random_state,
        )
        self.final_classifier_.fit(all_data, all_data[label_col], self.dt_feature_cols_)
        print(f"  Decision Tree trained on {len(all_data)} rows "
              f"({all_data[label_col].sum():.0f} TP, "
              f"{(1 - all_data[label_col]).sum():.0f} TN)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.final_classifier_ is None:
            raise RuntimeError("TrainingPipeline must be fit() before predicting.")
