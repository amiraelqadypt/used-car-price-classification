"""Metrics, McNemar significance testing and figures."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy plus weighted precision, recall and F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Tidy model-by-metric table, sorted by accuracy."""
    return (
        pd.DataFrame(results)
        .T.rename(columns=str.title)
        .sort_values("Accuracy", ascending=False)
        .round(4)
    )


# --------------------------------------------------------------------------
# McNemar's test
# --------------------------------------------------------------------------
def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """McNemar's test on the paired correct/incorrect outcomes of two models.

    Returns the discordant counts, the chi-square statistic with Yates'
    continuity correction, and a p-value. For small discordant totals the exact
    binomial p-value is reported instead, which is the standard recommendation.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c

    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "test": "identical"}

    if n < 25:
        p = binomtest(b, n, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p),
                "test": "exact binomial"}

    stat = (abs(b - c) - 1) ** 2 / n
    p = float(chi2.sf(stat, df=1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p,
            "test": "chi-square (Yates)"}


def pairwise_mcnemar(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Run McNemar's test for every pair of models."""
    rows = []
    for a, b in combinations(predictions, 2):
        res = mcnemar(y_true, predictions[a], predictions[b])
        rows.append({
            "Model A": a,
            "Model B": b,
            "A right / B wrong": res["b"],
            "A wrong / B right": res["c"],
            "Statistic": round(res["statistic"], 4),
            "p-value": res["p_value"],
            "Test": res["test"],
            "Significant (a=0.01)": res["p_value"] < 0.01,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(df["sellingprice"], bins=50, edgecolor="black", alpha=0.7, color="skyblue")
    axes[0].axvline(df["sellingprice"].mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: ${df['sellingprice'].mean():,.0f}")
    axes[0].axvline(df["sellingprice"].median(), color="green", linestyle="--", linewidth=2,
                    label=f"Median: ${df['sellingprice'].median():,.0f}")
    axes[0].set_xlabel("Selling Price ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Selling Prices", fontweight="bold")
    axes[0].legend()
    axes[1].boxplot(df["sellingprice"], vert=True, patch_artist=True)
    axes[1].set_ylabel("Selling Price ($)")
    axes[1].set_title("Box Plot of Selling Prices", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "price_distribution.png")


def plot_cluster_search(search, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(search.k_values, search.inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[1].plot(search.k_values, search.silhouettes, "ro-", linewidth=2, markersize=8)
    axes[1].axvline(search.optimal_k, color="green", linestyle="--", linewidth=2,
                    label=f"Optimal k={search.optimal_k}")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis", fontweight="bold")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, out_dir, "cluster_optimization.png")


def plot_cluster_profile(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    specs = [
        ("sellingprice", "Average Selling Price ($)", "skyblue", axes[0, 0]),
        ("car_age", "Average Car Age (years)", "lightcoral", axes[0, 1]),
        ("odometer", "Average Odometer (miles)", "lightgreen", axes[1, 0]),
    ]
    for col, label, colour, ax in specs:
        means = df.groupby("Cluster")[col].mean().sort_values()
        ax.barh(means.index.astype(str), means.values, color=colour)
        ax.set_xlabel(label)
        ax.set_ylabel("Cluster")
        ax.set_title(f"{label} by Cluster", fontweight="bold")
    sizes = df["Cluster"].value_counts().sort_index()
    axes[1, 1].bar(sizes.index.astype(str), sizes.values, color="plum")
    axes[1, 1].set_xlabel("Cluster")
    axes[1, 1].set_ylabel("Number of Cars")
    axes[1, 1].set_title("Cluster Size Distribution", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "cluster_analysis.png")


def plot_confusion(y_true, y_pred, labels, title: str, out_dir: Path, name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Count"})
    plt.xlabel("Predicted Price Range", fontweight="bold")
    plt.ylabel("Actual Price Range", fontweight="bold")
    plt.title(title, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save(fig, out_dir, name)


def plot_feature_importance(importances, feature_names, title: str,
                            out_dir: Path, name: str, top_n: int = 15) -> tuple[Path, pd.DataFrame]:
    table = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    table["Percentage"] = (table["Importance"] / table["Importance"].sum() * 100).round(2)
    top = table.head(min(top_n, len(table)))
    fig = plt.figure(figsize=(12, 8))
    colours = plt.cm.viridis(np.linspace(0, 1, len(top)))
    plt.barh(range(len(top)), top["Importance"].values, color=colours)
    plt.yticks(range(len(top)), top["Feature"].values)
    plt.xlabel("Importance Score", fontweight="bold")
    plt.ylabel("Features", fontweight="bold")
    plt.title(title, fontweight="bold", pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return _save(fig, out_dir, name), table


def text_report(y_true, y_pred, labels) -> str:
    return classification_report(y_true, y_pred, target_names=labels, digits=4,
                                 zero_division=0)
