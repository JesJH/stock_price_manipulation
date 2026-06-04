import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class AnomalyScorer:
    """
    Wraps sklearn's IsolationForest to produce a per-security anomaly score
    that is appended as an additional feature column before Decision Tree training.

    Role in the pipeline
    --------------------
    The IsolationForest is trained on the full TN candidate pool (the broadest
    available sample of "normal" securities). It then scores every security —
    both TP and TN — on how anomalous its behaviour looks relative to that
    normal baseline.

    That score is added as a single feature column ('if_anomaly_score') to the
    feature matrix. The Decision Tree can then use it as one input among many,
    giving it an unsupervised signal without replacing the interpretable
    rule-based output.

    Sklearn's IsolationForest returns a score where more negative = more
    anomalous. We negate it so that higher = more anomalous, which is more
    intuitive in the fraud context.

    Scaling note
    ------------
    A RobustScaler is fit on the training set inside fit() to normalise features
    before the IsolationForest sees them. As with CentroidSelector, the scaler
    must be re-fit on training-fold data only inside each CV iteration to
    prevent leakage.

    Parameters
    ----------
    n_estimators    : number of trees in the forest (default 100)
    contamination   : expected fraction of anomalies; 'auto' lets sklearn decide
    max_features    : fraction of features used per tree
    random_state    : reproducibility seed
    score_col       : name of the output score column added to the feature matrix
    """

    SCORE_COL = "if_anomaly_score"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        score_col: str = SCORE_COL,
    ):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.max_features  = max_features
        self.random_state  = random_state
        self.score_col     = score_col

        self._forest: IsolationForest | None = None
        self._scaler: RobustScaler | None = None
        self.feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_features: pd.DataFrame, feature_cols: list[str]) -> "AnomalyScorer":
        """
        Fit the IsolationForest on training-fold securities.
        Typically called with the TN candidate pool as the 'normal' baseline.

        Must be re-called from scratch in each CV fold (use fit_score()).
        """
        self.feature_cols_ = feature_cols
        X = self._prepare(train_features[feature_cols])

        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._forest.fit(X_scaled)
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        Return anomaly scores for the given features.
        Higher score = more anomalous.
        """
        self._check_fitted()
        X = self._prepare(features[self.feature_cols_])
        X_scaled = self._scaler.transform(X)
        # sklearn returns negative scores; negate so higher = more anomalous
        raw = self._forest.score_samples(X_scaled)
        return pd.Series(-raw, index=features.index, name=self.score_col)

    def fit_score(
        self,
        train_features: pd.DataFrame,
        score_features: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.Series:
        """
        Fit on train_features, then score score_features.
        Convenience wrapper for use inside CV loops.

        Typical usage inside a LOO-CV fold:
            scorer.fit_score(tn_train_features, all_features, feature_cols)
        """
        self.fit(train_features, feature_cols)
        return self.score(score_features)

    def add_score_column(
        self,
        features_df: pd.DataFrame,
        scores: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Append the anomaly score as a new column to a feature DataFrame.
        If scores is None, computes them from features_df directly.
        """
        if scores is None:
            scores = self.score(features_df)
        df = features_df.copy()
        df[self.score_col] = scores.values
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy().astype(float)
        return X.fillna(X.median()).values

    def _check_fitted(self) -> None:
        if self._forest is None:
            raise RuntimeError("AnomalyScorer must be fit() before calling score().")
