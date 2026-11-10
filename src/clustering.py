"""K-Means market segmentation.

The cluster label produced here is appended to the classifier's feature set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config


@dataclass
class ClusterSearch:
    k_values: list[int]
    inertias: list[float]
    silhouettes: list[float]
    optimal_k: int = field(init=False)
    best_silhouette: float = field(init=False)

    def __post_init__(self) -> None:
        best = int(np.argmax(self.silhouettes))
        self.optimal_k = self.k_values[best]
        self.best_silhouette = self.silhouettes[best]


def select_k(X_scaled: np.ndarray, verbose: bool = True) -> ClusterSearch:
    """Choose the number of clusters by silhouette score on a random sample."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = min(config.SILHOUETTE_SAMPLE_SIZE, X_scaled.shape[0])
    idx = rng.choice(X_scaled.shape[0], size=n, replace=False)
    sample = X_scaled[idx]

    k_values, inertias, silhouettes = [], [], []
    for k in config.K_RANGE:
        km = KMeans(
            n_clusters=k,
            random_state=config.RANDOM_STATE,
            n_init=config.KMEANS_SEARCH_N_INIT,
            max_iter=config.KMEANS_SEARCH_MAX_ITER,
        )
        km.fit(sample)
        sil = silhouette_score(sample, km.labels_)
        k_values.append(k)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(sil))
        if verbose:
            print(f"  k={k:<2}  silhouette={sil:.4f}  inertia={km.inertia_:,.0f}")

    return ClusterSearch(k_values, inertias, silhouettes)


def fit_final(X_scaled: np.ndarray, k: int) -> KMeans:
    """Fit the final K-Means model with the selected k."""
    km = KMeans(
        n_clusters=k,
        random_state=config.RANDOM_STATE,
        n_init=config.KMEANS_FINAL_N_INIT,
        max_iter=config.KMEANS_FINAL_MAX_ITER,
    )
    km.fit(X_scaled)
    return km
