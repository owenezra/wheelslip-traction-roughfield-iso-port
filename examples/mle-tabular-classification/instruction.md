# Synthetic Tabular Regression + Classification

You have a small tabular dataset with three continuous input features
and three prediction targets. Train on the provided training rows, then
produce predictions for the held-out test rows.

## Data

The public data files live under `/data/`:

- `/data/train.parquet`: 500 rows, columns `x1`, `x2`, `x3`, `t1`, `t2`,
  `label`. The first three are features, and the last three are targets.
- `/data/test.parquet`: 200 rows, columns `x1`, `x2`, `x3` only. No
  targets are exposed for the test set.
- `/data/column_mapping.json`: high-level descriptions of each column.
  It contains no formulas and no hints about relationships between
  columns.

The train and test sets differ in the range of `x1`: training rows have
`x1` inside a narrower sub-range than test rows. The distributions of
`x2` and `x3` are identical across splits. Models must generalize over
`x1` to score well on the test set.

## What To Produce

Write predictions for every test row to:

```text
/tmp/output/submission.csv
```

The file must have one row per input row in the same order as
`/data/test.parquet`.

Required columns, with a header row:

- `t1`: regression prediction, real-valued.
- `t2`: regression prediction, real-valued.
- `label`: classification prediction as integer `0` or `1`.

Example submission header:

```text
t1,t2,label
```

## Scoring

Each target is scored against its own metric and anchored between a
naive baseline and the theoretical optimum:

- `t1`: SRE, standardized RMSE (`RMSE / std(true)`); lower is better.
  Perfect is `0`.
- `t2`: SRE; lower is better. Perfect is `0`.
- `label`: binary F1; higher is better. Perfect is `1`.

The per-target scores are combined into a single continuous score in
`[0, 1]` by a weighted aggregate followed by an exponential mapping
through three anchor points: baseline, reference, and perfect.

The relationship between inputs and targets cannot be expressed in
closed form. Direct computation or formula-based approaches will not
achieve competitive accuracy.

For long-running training, use `tmux` and check progress later instead
of waiting inside one long bash command.
