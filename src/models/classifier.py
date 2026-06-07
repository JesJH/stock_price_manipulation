import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


class FraudClassifier:
    """
    Decision Tree classifier for pump-and-dump security detection.

    Decision trees were chosen to preserve interpretability: fraud reviewers
    need to trace exactly which behavioural conditions triggered a flag.
    Features are passed in unscaled — scaling would change split thresholds
    and make the tree harder to read in human terms.

    Parameters
    ----------
    max_depth         : maximum tree depth; shallow trees generalise better
                        with small TP sets (default 4)
    min_samples_leaf  : minimum samples per leaf; prevents micro-splits
    class_weight      : 'balanced' reweights by inverse class frequency,
                        compensating for TN:TP imbalance
    criterion         : 'gini' (default) or 'entropy'
    random_state      : reproducibility seed
    """

    def __init__(
        self,
        max_depth: int | None = 4,
        min_samples_leaf: int = 2,
        class_weight: str | dict | None = "balanced",
        criterion: str = "gini",
        random_state: int = 42,
    ):
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.class_weight      = class_weight
        self.criterion         = criterion
        self.random_state      = random_state

        self._tree: DecisionTreeClassifier | None = None
        self.feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features_df: pd.DataFrame,
        y: pd.Series | np.ndarray,
        feature_cols: list[str],
    ) -> "FraudClassifier":
        """
        Fit the decision tree on labelled feature rows.

        Parameters
        ----------
        features_df  : DataFrame containing at least the columns in feature_cols
        y            : binary labels (1 = TP, 0 = TN)
        feature_cols : which columns to use as model inputs
        """
        self.feature_cols_ = feature_cols
        X = self._prepare(features_df[feature_cols])
        self._tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            criterion=self.criterion,
            random_state=self.random_state,
        )
        self._tree.fit(X, y)
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = self._prepare(features_df[self.feature_cols_])
        return self._tree.predict(X)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Returns shape (n, 2); column 1 is P(fraud)."""
        self._check_fitted()
        X = self._prepare(features_df[self.feature_cols_])
        return self._tree.predict_proba(X)

    @property
    def feature_importances(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self._tree.feature_importances_,
            index=self.feature_cols_,
            name="importance",
        ).sort_values(ascending=False)

    def top_features(self, n: int = 10) -> pd.DataFrame:
        imp = self.feature_importances.head(n).reset_index()
        imp.columns = ["feature", "importance"]
        return imp

    def tree_rules(self) -> str:
        """Human-readable text representation of the decision tree."""
        self._check_fitted()
        return export_text(self._tree, feature_names=self.feature_cols_)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  FraudClassifier saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FraudClassifier":
        return joblib.load(Path(path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy().astype(float)
        return X.fillna(X.median()).values

    def _check_fitted(self) -> None:
        if self._tree is None:
            raise RuntimeError("FraudClassifier must be fit() before predicting.")
