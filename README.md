<p align="center">
  <img src="https://raw.githubusercontent.com/salim-studio/bandas/main/assets/banner.svg" alt="bandas banner" width="100%"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/bandas-df/"><img src="https://img.shields.io/pypi/v/bandas-df.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/bandas-df/"><img src="https://img.shields.io/pypi/pyversions/bandas-df.svg" alt="Python versions"/></a>
  <a href="https://github.com/salim-studio/bandas/blob/main/LICENSE"><img src="https://img.shields.io/github/license/salim-studio/bandas" alt="License"/></a>
  <a href="https://github.com/salim-studio/bandas"><img src="https://img.shields.io/github/stars/salim-studio/bandas" alt="GitHub stars"/></a>
  <a href="https://github.com/salim-studio/bandas/actions/workflows/ci.yml"><img src="https://github.com/salim-studio/bandas/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
</p>

# bandas — pandas-compatible DataFrames, at full speed ⚡

**bandas** is a pandas-compatible DataFrame library (same API names:
`DataFrame`, `Series`, `read_csv`, `concat`, `merge`, …) with a faster
execution engine — plus a complete data platform for databases, time series,
feature preprocessing, machine learning, deep learning and visualization.
Built for developers, data analysts, data scientists and ML engineers.

## Why bandas?

1. **Pure columnar NumPy storage** — no BlockManager overhead, no per-op index alignment.
2. **Numeric fast paths** — `sum/mean/min/max` straight through NumPy, `bincount`-powered groupby.
3. **Optional Numba + PyArrow acceleration** — compiled groupby kernels, 3–5x faster CSV reads.
4. **Real SQL built in** — `df.sql("SELECT …")` with zero extra dependencies (embedded SQLite), plus a unified `Database` connector for SQLite / DuckDB / Postgres / MySQL / SQLAlchemy.
5. **ML/DL ready** — `train_test_split`, `KFold`, metrics, `StandardScaler`, `OneHotEncoder`, `to_torch`, `to_tensorflow`, dataset loaders.
6. **Advanced analytics** — `rolling/expanding/ewm`, `shift/diff/pct_change`, `pivot_table/melt/crosstab`, `str`/`dt` accessors, and `LazyFrame` for out-of-core files.

## Install

One command installs everything (speed + databases + ML + plotting + Excel):

```bash
pip install bandas-df
```

From source:

```bash
git clone https://github.com/salim-studio/bandas.git
cd bandas
pip install -e .
```

Deep-learning frameworks only (`torch` and `tensorflow` are gigabytes in size
and have no wheels for the newest Python versions):

```bash
pip install "bandas-df[dl]"
```

## Quickstart

```python
import bandas as bd

df = bd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
print(df.head())
print(df["a"].sum())

df2 = bd.read_csv("data.csv")  # uses PyArrow when available
g = df2.groupby("city")["price"].mean()
print(g.to_pandas())

# Full pandas interop whenever you need it
pdf = df.to_pandas()
df3 = bd.from_pandas(pdf)
```

On 500k rows bandas is faster than pandas across the board
(`sum` ~1.9x, `mean` ~2.6x, `groupby` ~1.4x — see `bench/bench.py`).

## Databases / SQL

```python
# Zero-setup SQL — always works (embedded SQLite)
top = df.sql("SELECT city, AVG(price) AS m FROM df GROUP BY city ORDER BY m DESC")

# Query across several frames
from bandas import sql_query
out = sql_query("SELECT * FROM a JOIN b ON a.k = b.k WHERE a.v > 10", {"a": a, "b": b})

# Unified connector
from bandas.sql import Database
db = Database.sqlite("app.db")  # or ":memory:"
# db = Database.duckdb("data.duckdb")        # needs duckdb
# db = Database.postgres(user=..., password=..., host=..., db=...)
db.write(df, "sales")
print(db.query("SELECT COUNT(*) AS n FROM sales").to_pandas())
print(db.tables())

# pandas-style API
df.to_sql("sales", "sqlite:///app.db")
df2 = bd.read_sql("SELECT * FROM sales", "sqlite:///app.db")
```

## Time series & windows

```python
s = df["price"]
s.rolling(7).mean()    # moving average
s.expanding().sum()
s.ewm(span=10).mean()
s.shift(1); s.diff(); s.pct_change(); s.cumsum()
df.rolling(30).mean(); df.ewm(alpha=0.3).mean()

bd.to_datetime(["2024-01-01", "2024-02-01"])
bd.date_range("2024-01-01", periods=12, freq="ME")
df["d"].dt.year / .month / .day / .dayofweek
```

## Strings, categories & reshaping

```python
df["name"].str.lower().str.strip().str.contains("ali")
df["name"].str.replace("a", "@").str.split(" ", expand=True)
bd.get_dummies(df, columns=["city"])
bd.melt(df, id_vars=["city"])
bd.pivot_table(df, values="price", index="city", columns="year", aggfunc="mean")
bd.crosstab(df["city"], df["year"])
bd.cut(df["age"], bins=[0, 18, 60, 100])
bd.qcut(df["price"], 4)
```

## Preprocessing

```python
bd.SimpleImputer(strategy="mean").fit_transform(df)
bd.StandardScaler().fit_transform(df)
bd.MinMaxScaler().fit_transform(df)
bd.RobustScaler().fit_transform(df)
bd.LabelEncoder().fit_transform(df["city"])
bd.OneHotEncoder(columns=["city"]).fit_transform(df)
```

## Machine learning

```python
X_train, X_test, y_train, y_test = bd.train_test_split(X, y, test_size=0.2, random_state=0)
list(bd.KFold(n_splits=5).split(X))
bd.accuracy_score(y, pred); bd.f1_score(y, pred)
bd.mean_squared_error(y, pred); bd.r2_score(y, pred)

# Drop straight into scikit-learn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier().fit(X_train_numpy, y_train_numpy)
```

## Deep learning

```python
X_t, y_t = df.to_torch(target="label")     # torch.Tensor
loader = bd.dl.torch_dataset(df, target="label", batch_size=64)
ds = bd.dl.tf_dataset(df, target="label")  # tf.data.Dataset
```

## Large files (lazy / out-of-core)

```python
lf = bd.read_csv_chunked("big.csv", chunksize=100_000)
out = lf.filter(lambda d: d[d["x"] > 0]).select(["x", "y"]).collect()
print(lf.sum("x"), lf.mean("x"), lf.count())
```

## Quick plotting

```python
df.plot(kind="line", x="date", y="price")
df.plot(kind="bar", x="city", y="sales")
df["price"].hist(bins=50)
bd.viz.scatter(df, "a", "b"); bd.viz.heatmap_corr(df)
```

## Full IO coverage

`read_csv` / `read_parquet` / `read_json` / `read_excel` / `read_html` /
`read_feather` / `read_orc` / `read_sql` +
`to_csv` / `to_parquet` / `to_json` / `to_excel` / `to_sql` —
all with pandas-compatible signatures.

## Project layout

```
bandas/
  __init__.py      # public API + concat/merge/options/show_versions
  series.py        # Series + str/dt + rolling/ewm + cum*/shift/diff
  dataframe.py     # DataFrame + SQL + stats + reshape + viz + torch
  groupby.py       # fast GroupBy + std/var/median/first/last/nunique
  io.py            # every file reader + chunked
  sql.py           # sql_query + read_sql/to_sql + unified Database
  reshape.py       # get_dummies/melt/pivot/crosstab/cut/qcut
  window.py        # Rolling/Expanding/EWM
  strings.py       # Series.str
  datetimes.py     # Series.dt + to_datetime/date_range
  preprocessing.py # Imputer/Scaler/Encoder
  ml.py            # split/CV/metrics/sklearn bridge
  dl.py            # torch/tensorflow bridges
  stats.py         # corr/cov/outliers/summary
  lazy.py          # out-of-core LazyFrame
  viz.py           # quick charts
  ops.py / utils.py
assets/            # logo + banner
tests/             # 20 tests
bench/bench.py     # bandas vs pandas benchmark
```

## Test & benchmark

```bash
pytest -q
python bench/bench.py
python -c "import bandas; bandas.show_versions()"
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
