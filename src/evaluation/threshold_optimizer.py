import numpy as np
import pandas as pd

from src.evaluation.metrics import EvaluationMetrics


class ThresholdOptimizer:
    """
    Grid search over classification thresholds to find the operating point
    that minimises the FP/FN cost-weighted total error.

    Usage
    -----
    After LOO-CV produces out-of-fold predicted probabilities:

        opt = ThresholdOptimizer(fp_cost=1.0, fn_cost=5.0)
        opt.fit(oof_labels, oof_probas)
        print(opt.best_threshold_)
        print(opt.optimal_metrics_)
        opt.plot_cost_curve()

    Parameters
    ----------
    fp_cost      : relative cost of one false positive (default 1.0)
    fn_cost      : relative cost of one false negative (default 5.0)
    n_thresholds : number of threshold candidates in [0.01, 0.99]
    """

    def __init__(
        self,
        fp_cost: float = 1.0,
        fn_cost: float = 5.0,
        n_thresholds: int = 99,
    ):
        self.fp_cost      = fp_cost
        self.fn_cost      = fn_cost
        self.n_thresholds = n_thresholds

        self.results_df_:      pd.DataFrame | None = None
        self.best_threshold_:  float | None = None
        self.optimal_metrics_: dict | None  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> "ThresholdOptimizer":
        """
        Evaluate all threshold candidates and identify the cost-optimal one.

        After calling fit():
          best_threshold_  — threshold minimising total weighted cost
          results_df_      — full DataFrame of threshold vs. all metrics
          optimal_metrics_ — metric dict at best_threshold_
        """
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds).tolist()
        rows = [
            EvaluationMetrics.compute(y_true, y_proba, t, self.fp_cost, self.fn_cost)
            for t in thresholds
        ]
        self.results_df_ = pd.DataFrame(rows)

        best_idx             = self.results_df_["cost"].idxmin()
        self.best_threshold_ = float(self.results_df_.loc[best_idx, "threshold"])
        self.optimal_metrics_ = self.results_df_.loc[best_idx].to_dict()
        return self

    def plot_cost_curve(self, ax=None) -> None:
        """Plot total cost vs. threshold, marking the optimal point."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(8, 4))

        df = self.results_df_
        ax.plot(df["threshold"], df["cost"], label="Total cost", color="steelblue")
        ax.axvline(
            self.best_threshold_, color="tomato", linestyle="--",
            label=f"Optimal threshold = {self.best_threshold_:.2f}"
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Weighted cost (FP·fp_cost + FN·fn_cost)")
        ax.set_title("Cost curve: threshold optimisation")
        ax.legend()

        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_pr_curve(self, ax=None) -> None:
        """Plot precision and recall vs. threshold."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(8, 4))

        df = self.results_df_
        ax.plot(df["threshold"], df["precision"], label="Precision", color="steelblue")
        ax.plot(df["threshold"], df["recall"],    label="Recall",    color="darkorange")
        ax.axvline(
            self.best_threshold_, color="tomato", linestyle="--",
            label=f"Optimal threshold = {self.best_threshold_:.2f}"
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall vs. threshold")
        ax.set_ylim(0, 1.05)
        ax.legend()

        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_precision_recall_tradeoff(self, ax=None) -> None:
        """Precision-Recall space (x=recall, y=precision)."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(6, 5))

        df = self.results_df_
        sc = ax.scatter(
            df["recall"], df["precision"],
            c=df["threshold"], cmap="viridis", s=20,
        )
        opt = self.optimal_metrics_
        ax.scatter(
            opt["recall"], opt["precision"],
            marker="*", s=200, color="tomato", zorder=5,
            label=f"Optimal (t={self.best_threshold_:.2f})"
        )
        plt.colorbar(sc, ax=ax, label="Threshold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall tradeoff")
        ax.legend()

        if fig is not None:
            plt.tight_layout()
            plt.show()

    def summary(self) -> str:
        self._check_fitted()
        m = self.optimal_metrics_
        return (
            f"Optimal threshold: {self.best_threshold_:.4f}\n"
            f"  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  "
            f"F1: {m['f1']:.4f}\n"
            f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}\n"
            f"  ROC-AUC: {m['roc_auc']}  PR-AUC: {m['pr_auc']}\n"
            f"  Weighted cost: {m['cost']:.1f}  "
            f"(fp_cost={self.fp_cost}, fn_cost={self.fn_cost})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.results_df_ is None:
            raise RuntimeError("Call fit() before accessing results.")
