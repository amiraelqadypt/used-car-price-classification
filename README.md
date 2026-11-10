# Hybrid Clustering-Enhanced Ensemble Learning for Used-Car Price Range Classification

Experimental code for the M.Sc. thesis *A Machine Learning Based Predictive Model for Used Cars Prices*, Department of Information Systems and Computers, Faculty of Business, Alexandria University.

**Author:** Amira Mahmoud Salem Mahmoud El Qady
**Supervisors:** Prof. Ghada A. El Khayat · Prof. Abeer Abd El-Hamid Amer

**Related publication:** A. El Qady, G. Elkhayat and A. A. Amer, "Hybrid Clustering-Enhanced Ensemble Learning for Large-Scale Used-Car Price Range Classification," *2026 International Conference on Machine Intelligence and Smart Innovation (ICMISI 2026)*, Alexandria, Egypt, 9–12 May 2026, pp. 371–378.

---

## What this does

The framework predicts which of four price ranges a used vehicle falls into — Budget, Mid-Range, Premium or Luxury — from its attributes and auction reference data.

It runs in two stages. First, K-Means segments the vehicles into market clusters, with the number of clusters chosen by silhouette score over k = 2…10. The resulting cluster label is then appended to the feature set of four supervised classifiers: XGBoost, Random Forest, a linear SVM, and a soft-voting ensemble averaging the probability outputs of the first two. Models are compared on accuracy, weighted precision, recall and F1, and the differences between them are tested for significance with McNemar's test on the paired test-set outcomes.

## Two run modes

K-Means is fitted on vehicle attributes only — year, condition, odometer, mmr, car_age and the six encoded categorical columns. No feature derived from the selling price is supplied to the clustering step, so the cluster label carries no target information into the classifier.

The classifier feature set differs between the two modes:

| | `thesis` | `strict` |
|---|---|---|
| `price_ratio` as a classifier input | yes | no |
| `price_ratio` supplied to K-Means | no | no |
| Scaler and cluster centroids fitted on | the full dataset | the training partition only |

`price_ratio` is `sellingprice / (mmr + 1)`. It describes the relationship between the transaction price and the market reference value, and it is only available where the transaction price is already known — retrospective valuation auditing, for instance, rather than forecasting an unknown price. Because `mmr` is also a classifier input, a model given both can recover the selling price arithmetically, and the accuracy reported in `thesis` mode should be read in that light.

`strict` mode drops the price-derived features entirely and fits the scaler and K-Means on the training partition only. It gives the figure that applies when the price is genuinely unknown at prediction time.

Run both and compare:

```bash
python -m src.main --mode both
```

The headline accuracy of each mode is printed side by side at the end.

## Reproduced results

The `thesis` configuration was run on the full dataset and reproduces the
figures published in the thesis. `random_state = 42`, silhouette sample 50,000.

**Pipeline**

| | This run | Thesis |
|---|---|---|
| Records after cleaning | 547,921 | 547,921 |
| Optimal k | 3 | 3 |
| Silhouette score | 0.2574 | 0.2560 |

**Accuracy (Table 4.3)**

| Model | This run | Thesis | Difference |
|---|---|---|---|
| XGBoost | 0.9983 | 0.9981 | +0.0002 |
| Random Forest | 0.9721 | 0.9710 | +0.0011 |
| SVM | 0.8372 | 0.8372 | 0.0000 |
| Soft Voting Ensemble | 0.9982 | 0.9983 | −0.0001 |

**Feature importance, top five (Table 4.4)**

| Feature | This run | Thesis | Difference |
|---|---|---|---|
| Cluster | 0.5120 | 0.5897 | −0.0777 |
| mmr | 0.3127 | 0.2803 | +0.0324 |
| price_ratio | 0.0673 | 0.0567 | +0.0106 |
| year | 0.0326 | 0.0068 | +0.0258 |
| body_encoded | 0.0178 | 0.0035 | +0.0143 |

Cluster remains the dominant feature even though K-Means is fitted on vehicle
attributes alone, which confirms that the segmentation carries genuine market
structure rather than target information. Small differences across all tables
are the expected run-to-run variation from library versions and thread counts.

Figures for `strict` mode are not listed here because that configuration has
not yet been run on the full dataset. Run it with `--mode strict`.

## Dataset

Not included in this repository. Download `car_prices.csv` (roughly 550,000 US auction records, 2014–2015) from either source and place it in `data/`:

- https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data
- https://www.kaggle.com/datasets/tunguz/used-car-auction-prices

See `data/README.md`.

## Installation

```bash
git clone https://github.com/amiraelqadypt/used-car-price-classification.git
cd used-car-price-classification
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## Usage

```bash
python -m src.main --mode both                       # both configurations
python -m src.main --mode thesis                     # as reported in the thesis
python -m src.main --mode strict                     # leakage-free
python -m src.main --data /path/to/car_prices.csv    # dataset elsewhere
python -m src.main --results out/                    # alternative output directory
```

Each run writes to `results/<mode>/`: `model_comparison.csv`, `mcnemar_tests.csv`, `feature_importance.csv`, `classification_report.txt`, `summary.json`, and the figures (`cluster_optimization.png`, `cluster_analysis.png`, `confusion_matrix.png`, `feature_importance.png`).

## Pipeline

1. **Cleaning** — median imputation for numeric columns, mode for categorical, duplicate removal, type coercion, and removal of selling prices outside the 1st–99th percentile.
2. **Feature engineering** — `car_age` (reference year 2024), `price_diff`, `price_ratio`.
3. **Encoding** — label encoding of `make`, `body`, `transmission`, `state`, `color`, `interior`. Standard scaling is applied to the clustering inputs; the tree-based classifiers use the raw values.
4. **Target** — `sellingprice` discretised into four equal-frequency classes by quartile.
5. **Clustering** — K-Means on the scaled vehicle attributes, with no access to the target or to any feature derived from it. k is chosen by silhouette score over k = 2…10 on a 50,000-row sample, with `n_init` 10 during the search and 20 for the final fit. The resulting label is appended to the classifier feature set.
6. **Classification** — XGBoost, Random Forest, LinearSVC, plus the soft-voting ensemble, on an 80/20 stratified split. `LinearSVC` is the default; it reproduces the reported SVM accuracy exactly, while an RBF kernel at this scale is O(N²)–O(N³) as Section 3.14 notes.
7. **Evaluation** — metrics, confusion matrix, feature importance, and pairwise McNemar tests.

## Reproducibility

`random_state = 42` throughout, and `random.seed` / `numpy.random.seed` are set at entry. The silhouette sample uses a seeded `numpy.random.Generator`. Exact figures can still shift slightly across BLAS builds, library versions and thread counts; the ordering of models is stable.

Runtime is roughly 15–40 minutes per mode on the full dataset, dominated by the silhouette search over k = 2…10 and the SVM. The silhouette calculation needs a few GB of free memory; on a constrained machine, reduce scikit-learn's chunk size with `sklearn.set_config(working_memory=512)` rather than shrinking the 50,000-row sample, which Section 3.7.2 specifies.

## Layout

```
notebooks/            Colab / Jupyter reproduction notebook
src/config.py         parameters, seeds, run-mode definitions
src/data_loader.py    CSV loading and schema check
src/preprocessing.py  cleaning, feature engineering, encoding, target
src/clustering.py     k selection by silhouette, final K-Means fit
src/models.py         classifiers and the soft-voting ensemble
src/evaluation.py     metrics, McNemar's test, figures
src/main.py           pipeline entry point
```

## Citation

```bibtex
@inproceedings{elqady2026hybrid,
  title     = {Hybrid Clustering-Enhanced Ensemble Learning for Large-Scale
               Used-Car Price Range Classification},
  author    = {El Qady, Amira and Elkhayat, Ghada and Amer, Abeer A.},
  booktitle = {2026 International Conference on Machine Intelligence and
               Smart Innovation (ICMISI)},
  address   = {Alexandria, Egypt},
  pages     = {371--378},
  year      = {2026}
}
```

## Licence

MIT — see `LICENSE`. The dataset is not covered by this licence; it remains subject to its own terms on Kaggle.
