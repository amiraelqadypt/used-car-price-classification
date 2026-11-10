"""End-to-end pipeline: clean, cluster, classify, evaluate.

Usage
-----
    python -m src.main --mode thesis
    python -m src.main --mode strict
    python -m src.main --mode both --data /path/to/car_prices.csv
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import clustering, config, data_loader, evaluation, models, preprocessing


def set_seeds(seed: int = config.RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_mode(df_clean: pd.DataFrame, mode: str, out_dir: Path) -> dict:
    """Run the full pipeline for one configuration and return its results."""
    settings = config.MODES[mode]
    include_price_derived = settings["include_price_derived"]
    cluster_uses_price_derived = settings["cluster_uses_price_derived"]
    fit_on_train_only = settings["fit_on_train_only"]

    print("\n" + "=" * 78)
    print(f"MODE: {mode}")
    print(f"  price-derived features for the classifier : {include_price_derived}")
    print(f"  price-derived features for K-Means        : {cluster_uses_price_derived}")
    print(f"  scaler / K-Means fitted on train only     : {fit_on_train_only}")
    print("=" * 78)

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seeds()

    df = df_clean.copy()
    feature_cols = preprocessing.feature_columns(include_price_derived)
    cluster_cols = preprocessing.clustering_columns(cluster_uses_price_derived)

    # ---- target -------------------------------------------------------
    y_labels, bins = preprocessing.build_target(df)
    df["Price_Range"] = y_labels
    # Encode in price order, not alphabetically: a LabelEncoder would sort the
    # labels to Budget, Luxury, Mid-Range, Premium, which scrambles adjacency
    # in the confusion matrix and the classification report.
    class_names = list(config.PRICE_CLASS_LABELS)
    y = df["Price_Range"].map({name: i for i, name in enumerate(class_names)}).to_numpy()

    print("\nPrice range boundaries:")
    for i, name in enumerate(config.PRICE_CLASS_LABELS):
        print(f"  {name:<10} ${bins[i]:>10,.0f} - ${bins[i + 1]:>10,.0f}")

    # ---- split first, so nothing is fitted on the test rows ------------
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )

    # ---- scaling (for clustering only; the tree models use raw values) ---
    scaler = StandardScaler()
    X_cluster_raw = df[cluster_cols].to_numpy(dtype=float)
    if fit_on_train_only:
        scaler.fit(X_cluster_raw[idx_train])
    else:
        scaler.fit(X_cluster_raw)
    X_scaled = scaler.transform(X_cluster_raw)
    print(f"\nK-Means input: {len(cluster_cols)} scaled vehicle attributes "
          f"({', '.join(cluster_cols)})")

    # ---- clustering ----------------------------------------------------
    print("\nSelecting number of clusters:")
    search_input = X_scaled[idx_train] if fit_on_train_only else X_scaled
    search = clustering.select_k(search_input)
    print(f"  -> optimal k = {search.optimal_k} "
          f"(silhouette {search.best_silhouette:.4f})")

    kmeans = clustering.fit_final(search_input, search.optimal_k)
    df["Cluster"] = kmeans.predict(X_scaled)

    dist = df["Cluster"].value_counts().sort_index()
    print("\nCluster distribution:")
    for cid, count in dist.items():
        print(f"  cluster {cid}: {count:,} ({count / len(df) * 100:.2f}%)")

    # ---- classification ------------------------------------------------
    classification_features = feature_cols + ["Cluster"]
    X = df[classification_features]
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    print(f"\nTraining set: {len(X_train):,}   Test set: {len(X_test):,}")

    print("\nTraining models:")
    t0 = time.perf_counter()
    fitted = models.train_all(models.build_models(), X_train, y_train)
    train_seconds = time.perf_counter() - t0
    print(f"  total training time: {train_seconds:.1f} s")

    predictions = {name: m.predict(X_test) for name, m in fitted.items()}
    predictions["Soft Voting Ensemble"] = models.soft_vote_predict(fitted, X_test)

    results = {name: evaluation.score(y_test, pred) for name, pred in predictions.items()}
    table = evaluation.comparison_table(results)
    print("\nModel comparison:")
    print(table.to_string())

    best_name = table.index[0]
    print(f"\nBest model: {best_name} (accuracy {table.loc[best_name, 'Accuracy']:.4f})")

    # ---- significance testing ------------------------------------------
    mcnemar_table = evaluation.pairwise_mcnemar(y_test, predictions)
    print("\nMcNemar pairwise tests:")
    print(mcnemar_table.to_string(index=False))

    # ---- figures and artefacts -----------------------------------------
    evaluation.plot_cluster_search(search, out_dir)
    evaluation.plot_cluster_profile(df, out_dir)
    evaluation.plot_confusion(
        y_test, predictions[best_name], class_names,
        f"Confusion Matrix - {best_name}\nAccuracy: "
        f"{table.loc[best_name, 'Accuracy']:.2%}",
        out_dir, "confusion_matrix.png",
    )

    importance_table = None
    source = best_name if hasattr(fitted.get(best_name), "feature_importances_") else "XGBoost"
    if hasattr(fitted[source], "feature_importances_"):
        _, importance_table = evaluation.plot_feature_importance(
            fitted[source].feature_importances_, classification_features,
            f"Feature Importances - {source}", out_dir, "feature_importance.png",
        )
        print(f"\nFeature importance ({source}):")
        print(importance_table.head(10).to_string(index=False))

    print("\nClassification report (best model):")
    report = evaluation.text_report(y_test, predictions[best_name], class_names)
    print(report)

    table.to_csv(out_dir / "model_comparison.csv")
    mcnemar_table.to_csv(out_dir / "mcnemar_tests.csv", index=False)
    if importance_table is not None:
        importance_table.to_csv(out_dir / "feature_importance.csv", index=False)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "mode": mode,
        "settings": settings,
        "random_state": config.RANDOM_STATE,
        "records": int(len(df)),
        "clustering_features": cluster_cols,
        "classifier_features": classification_features,
        "optimal_k": int(search.optimal_k),
        "silhouette": round(search.best_silhouette, 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "training_seconds": round(train_seconds, 1),
        "metrics": {k: {m: round(v, 4) for m, v in s.items()} for k, s in results.items()},
        "best_model": best_name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nArtefacts written to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="path to car_prices.csv")
    parser.add_argument("--mode", default="both", choices=["thesis", "strict", "both"],
                        help="which configuration to run (default: both)")
    parser.add_argument("--results", default=None, help="output directory")
    args = parser.parse_args()

    set_seeds()
    results_root = Path(args.results) if args.results else config.RESULTS_DIR

    print("Loading dataset ...")
    df_raw = data_loader.load_raw(args.data)
    print("\nCleaning:")
    df_clean = preprocessing.engineer_features(preprocessing.clean(df_raw))
    df_clean, _ = preprocessing.encode_categoricals(df_clean)

    evaluation.plot_price_distribution(df_clean, results_root)

    modes = ["thesis", "strict"] if args.mode == "both" else [args.mode]
    summaries = [run_mode(df_clean, m, results_root / m) for m in modes]

    if len(summaries) > 1:
        print("\n" + "=" * 78)
        print("HEADLINE COMPARISON")
        print("=" * 78)
        for s in summaries:
            acc = s["metrics"][s["best_model"]]["accuracy"]
            print(f"  {s['mode']:<8} best={s['best_model']:<22} accuracy={acc:.4f}")
        print("\nThe difference between the two is the contribution of the\n"
              "price-derived classifier features. See README, 'Two run modes'.")


if __name__ == "__main__":
    main()
