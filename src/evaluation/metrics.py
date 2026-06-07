import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


class EvaluationMetrics:
    """
    Evaluation metrics for the fraud classifier with asymmetric cost weighting.

    Cost model
    ----------
    FP (false positive): legitimate security flagged and blocked.
        Cost = revenue opportunity loss + customer friction.
    FN (false negative): fraudulent security missed, fraud executes.
        Cost = direct financial loss + reputational damage.

    A cost ratio of fn_cost = 5 × fp_cost means we are willing to accept
    five false alarms to avoid one missed fraud case.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        fp_cost: float = 1.0,
        fn_cost: float = 5.0,
    ) -> dict:
        """
        Compute a full suite of metrics at a single threshold.

        Returns a dict with: precision, recall, f1, specificity,
        tp/fp/tn/fn counts, roc_auc, pr_auc, and cost metrics.
        """
        y_true  = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_pred  = (y_proba >= threshold).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        has_both_classes = len(np.unique(y_true)) > 1
        roc_auc = float(roc_auc_score(y_true, y_proba)) if has_both_classes else float("nan")
        pr_auc  = float(average_precision_score(y_true, y_proba)) if has_both_classes else float("nan")

        cost    = fp * fp_cost + fn * fn_cost
        n_total = len(y_true)

        return {
            "threshold":       round(threshold, 4),
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "f1":              round(f1, 4),
            "specificity":     round(specificity, 4),
            "tp":              tp,
            "fp":              fp,
            "tn":              tn,
            "fn":              fn,
            "roc_auc":         round(roc_auc, 4) if not np.isnan(roc_auc) else None,
            "pr_auc":          round(pr_auc, 4)  if not np.isnan(pr_auc)  else None,
            "cost":            round(cost, 4),
            "cost_per_case":   round(cost / n_total, 4) if n_total > 0 else None,
            "n_positive":      int(y_true.sum()),
            "n_negative":      int((1 - y_true).sum()),
            "n_total":         n_total,
        }

    @staticmethod
    def summary_table(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: list[float] | None = None,
        fp_cost: float = 1.0,
        fn_cost: float = 5.0,
    ) -> pd.DataFrame:
        """
        Compute metrics at multiple thresholds.

        Returns a DataFrame sorted by threshold, useful for plotting
        the precision/recall/cost tradeoff curve.
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19).tolist()
        rows = [
            EvaluationMetrics.compute(y_true, y_proba, t, fp_cost, fn_cost)
            for t in thresholds
        ]
        return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
