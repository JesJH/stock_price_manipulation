import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler


class CentroidSelector:
    """
    Model 1: centroid-based true negative downsampling.

    Each confirmed TP security acts as a centroid. TN candidates are assigned
    to their nearest TP centroid by Euclidean distance in robust-scaled feature
    space, then downsampled so the classifier must learn genuine behavioural
    distinctions rather than trivial ones between unrelated securities.

    Three downsampling methods
    --------------------------
    random
        For each TP centroid, draw N TNs at random from its assigned pool.
        Baseline — makes no assumption about match quality.

    distance_ranked
        Select the N closest TNs to each centroid. Maximises match quality:
        the classifier sees the hardest cases (most similar to the TP) and
        must learn subtle distinctions.

    stratified
        Bin each centroid's pool into Q equal-count quantile bands by distance,
        then sample from each band. Balances hard near-boundary cases with
        clearly distinct easy cases, producing a more robust decision boundary.

    Scaling note
    ------------
    A RobustScaler is fit on TP features inside fit() and applied to TN
    candidates in select(). The scaler must be re-fit from scratch in each
    CV fold (call fit_select() per fold) to prevent test data leaking into
    the distance computation.

    Decision-tree features are NOT scaled here — that is handled separately
    by the downstream classifier which receives raw unscaled feature values.

    Parameters
    ----------
    method          : 'random' | 'distance_ranked' | 'stratified'
    tn_tp_ratio     : target number of TNs per TP centroid
    n_bins          : number of quantile bins for stratified method
    bin_weights     : 'uniform' | 'closer_weighted' — how to sample within bins
    max_distance_pct: if set, drop TNs beyond this percentile of distance before
                      sampling (removes outlier TNs too unlike their centroid)
    random_state    : reproducibility seed
    """

    VALID_METHODS = ("random", "distance_ranked", "stratified")

    def __init__(
        self,
        method: str = "distance_ranked",
        tn_tp_ratio: int = 5,
        n_bins: int = 5,
        bin_weights: str = "uniform",
        max_distance_pct: float | None = None,
        random_state: int = 42,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")
        if bin_weights not in ("uniform", "closer_weighted"):
            raise ValueError("bin_weights must be 'uniform' or 'closer_weighted'")

        self.method           = method
        self.tn_tp_ratio      = tn_tp_ratio
        self.n_bins           = n_bins
        self.bin_weights      = bin_weights
        self.max_distance_pct = max_distance_pct
        self.random_state     = random_state

        self.scaler_: RobustScaler | None = None
        self.tp_scaled_: np.ndarray | None = None
        self.feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, tp_features: pd.DataFrame, feature_cols: list[str]) -> "CentroidSelector":
        """
        Fit the robust scaler on TP features and store scaled TP centroids.
        Must be called on training-fold TPs only inside each CV iteration.
        """
        self.feature_cols_ = feature_cols
        X_tp = self._prepare(tp_features[feature_cols])
        self.scaler_ = RobustScaler()
        self.tp_scaled_ = self.scaler_.fit_transform(X_tp)
        return self

    def select(self, tn_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Assign TN candidates to TP centroids and downsample.

        Returns a DataFrame of selected TNs with metadata columns stripped
        (_centroid_idx, _distance). Call build_training_set() to combine
        with TPs into a labelled training set.
        """
        self._check_fitted()

        X_tn = self._prepare(tn_candidates[self.feature_cols_])
        X_tn_scaled = self.scaler_.transform(X_tn)

        # Distance matrix: shape (n_tn, n_tp)
        dist_matrix = cdist(X_tn_scaled, self.tp_scaled_, metric="euclidean")

        assignments  = dist_matrix.argmin(axis=1)   # nearest TP centroid index
        min_dists    = dist_matrix.min(axis=1)

        working = tn_candidates.copy()
        working["_centroid_idx"] = assignments
        working["_distance"]     = min_dists

        # Optional: drop TNs too far from any TP centroid
        if self.max_distance_pct is not None:
            cutoff = np.percentile(min_dists, self.max_distance_pct)
            before = len(working)
            working = working[working["_distance"] <= cutoff].reset_index(drop=True)
            dropped = before - len(working)
            if dropped > 0:
                print(f"  CentroidSelector: dropped {dropped} TNs beyond "
                      f"{self.max_distance_pct}th-percentile distance cutoff")

        selected_parts: list[pd.DataFrame] = []
        n_tp = len(self.tp_scaled_)

        for centroid_idx in range(n_tp):
            pool = working[working["_centroid_idx"] == centroid_idx]
            if pool.empty:
                continue

            if self.method == "random":
                chunk = self._random_sample(pool, self.tn_tp_ratio)
            elif self.method == "distance_ranked":
                chunk = self._distance_ranked_sample(pool, self.tn_tp_ratio)
            else:
                chunk = self._stratified_sample(pool, self.tn_tp_ratio)

            selected_parts.append(chunk)

        if not selected_parts:
            return pd.DataFrame()

        result = pd.concat(selected_parts, ignore_index=True)
        return result.drop(columns=["_centroid_idx", "_distance"], errors="ignore")

    def fit_select(
        self,
        tp_features: pd.DataFrame,
        tn_candidates: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Fit on TP features then select TNs. Convenience wrapper for CV loops."""
        return self.fit(tp_features, feature_cols).select(tn_candidates)

    def build_training_set(
        self,
        tp_features: pd.DataFrame,
        selected_tn: pd.DataFrame,
        label_col: str = "label",
    ) -> pd.DataFrame:
        """
        Combine TP rows (label=1) with selected TN rows (label=0) into one
        training DataFrame. Both inputs must already have a label column.
        """
        return pd.concat([tp_features, selected_tn], ignore_index=True).sample(
            frac=1, random_state=self.random_state
        )

    def centroid_pool_sizes(self, tn_candidates: pd.DataFrame) -> pd.Series:
        """
        Return the number of TN candidates assigned to each TP centroid.
        Useful for diagnosing uneven cluster sizes before sampling.
        """
        self._check_fitted()
        X_tn     = self._prepare(tn_candidates[self.feature_cols_])
        scaled   = self.scaler_.transform(X_tn)
        dist_mat = cdist(scaled, self.tp_scaled_, metric="euclidean")
        assignments = dist_mat.argmin(axis=1)
        return pd.Series(assignments).value_counts().sort_index().rename("pool_size")

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def _random_sample(self, pool: pd.DataFrame, n: int) -> pd.DataFrame:
        replace = n > len(pool)
        return pool.sample(n=min(n, len(pool)), replace=replace, random_state=self.random_state)

    def _distance_ranked_sample(self, pool: pd.DataFrame, n: int) -> pd.DataFrame:
        return pool.nsmallest(min(n, len(pool)), "_distance")

    def _stratified_sample(self, pool: pd.DataFrame, n: int) -> pd.DataFrame:
        # Fall back to distance-ranked when pool is too small to bin
        if len(pool) < self.n_bins * 2:
            return self._distance_ranked_sample(pool, n)

        pool = pool.copy()
        pool["_bin"] = pd.qcut(
            pool["_distance"], q=self.n_bins, labels=False, duplicates="drop"
        )
        actual_bins  = pool["_bin"].nunique()
        n_per_bin    = max(1, n // actual_bins)

        parts: list[pd.DataFrame] = []
        for _, bin_pool in pool.groupby("_bin", sort=True):
            take = min(n_per_bin, len(bin_pool))
            if self.bin_weights == "closer_weighted":
                w = 1.0 / (bin_pool["_distance"] + 1e-9)
                w = w / w.sum()
                chunk = bin_pool.sample(n=take, weights=w, random_state=self.random_state)
            else:
                chunk = bin_pool.sample(n=take, random_state=self.random_state)
            parts.append(chunk)

        result = pd.concat(parts, ignore_index=True)

        # Top up to n if bins were small — take the next-closest unused TNs
        if len(result) < n:
            used = result.index
            leftover = pool[~pool.index.isin(used)].nsmallest(n - len(result), "_distance")
            result = pd.concat([result, leftover], ignore_index=True)

        return result.drop(columns=["_bin"], errors="ignore")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        """Median-impute NaNs and return a float numpy array."""
        X = X.copy().astype(float)
        col_medians = X.median()
        return X.fillna(col_medians).values

    def _check_fitted(self) -> None:
        if self.scaler_ is None or self.tp_scaled_ is None:
            raise RuntimeError("CentroidSelector must be fit() before calling select().")
