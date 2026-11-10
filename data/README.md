# Dataset

The raw data is not stored in this repository: the file is around 80 MB, above
GitHub's comfortable limit, and it is redistributed under its own terms on
Kaggle rather than ours.

## Download

Either source provides the same US used-car auction data (2014-2015,
approximately 558,000 records, 16 attributes):

- https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data
- https://www.kaggle.com/datasets/tunguz/used-car-auction-prices

Place `car_prices.csv` in this directory:

```
data/car_prices.csv
```

Or point the pipeline somewhere else:

```bash
python -m src.main --data /path/to/car_prices.csv
```

## Expected columns

`year`, `make`, `model`, `trim`, `body`, `transmission`, `vin`, `state`,
`condition`, `odometer`, `color`, `interior`, `seller`, `mmr`, `sellingprice`,
`saledate`

The loader checks for the columns the pipeline needs and fails with a clear
message if any are absent.
