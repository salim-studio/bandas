# bandas ⚡ — pandas-compatible, faster + Universal Data Platform

> أُعيدت تسمية المكتبة من `fandas` إلى `bandas` — المستودع الرسمي الآن: https://github.com/salim-studio/bandas
> للترحيل: `pip uninstall fandas` ثم `pip install bandas-df`، واستبدل `import fandas` بـ `import bandas`.

`bandas` واجهة مطابقة لـ `pandas` (نفس الأسماء: `DataFrame`, `Series`, `read_csv`, `concat`, `merge`...) لكن **أسرع**، ومع **منظومة متكاملة** لقواعد البيانات، السلاسل الزمنية، تجهيز البيانات، تعلم الآلة، التعلم العميق، والتصوير — لتصبح خيار المطورين ومحللي البيانات وعلماء البيانات.

## لماذا bandas؟

1. **تخزين عمودي خالص بـ NumPy** بدون Overhead الـ BlockManager.
2. **Fast-paths رقمية**: `sum/mean/min/max` عبر numpy مباشرة + `bincount` للـ groupby.
3. **تسريع اختياري بـ Numba** + **قراءة CSV بـ PyArrow** (أسرع 3-5x).
4. **SQL حقيقي**: `df.sql("SELECT ...")` بدون أي اعتماد إضافي (sqlite مدمج)، وموصل موحد `Database` لـ sqlite / duckdb / postgres / mysql / sqlalchemy.
5. **ML/DL جاهز**: `train_test_split`, `KFold`, مقاييس، `StandardScaler`, `OneHotEncoder`, `to_torch`, `to_tensorflow`, `torch_dataset`.
6. **تحليل متقدم**: `rolling/expanding/ewm`, `shift/diff/pct_change`, `pivot_table/melt/crosstab`, `str/dt accessors`, `LazyFrame` للملفات الضخمة خارج الذاكرة.

## تنصيب

تنصيب واحد يثبّت كل شيء (سرعة + قواعد بيانات + تعلم آلة + تصوير + إكسل):

```bash
pip install bandas-df
```

من المصدر:

```bash
pip install -e .
```

إضافي فقط للتعلم العميق (torch و tensorflow ضخمان — جيجابايتات — وغير متوفرين لكل نسخ Python):

```bash
pip install "bandas-df[dl]"
# أو من المصدر:
pip install -e ".[dl]"
```

## 1) أساسيات (أسرع من pandas)

```python
import bandas as bd

df = bd.DataFrame({"a": [1,2,3], "b": [4.0,5.0,6.0]})
print(df.head())
print(df["a"].sum())

df2 = bd.read_csv("data.csv")          # pyarrow إن وجد
g = df2.groupby("city")["price"].mean()
print(g.to_pandas())

pdf = df.to_pandas()
df3 = bd.from_pandas(pdf)
```

النتيجة على 500k صف: `sum 1.9x`, `mean 1.5x`, `groupby 1.3x` أسرع من pandas (انظر `bench/bench.py`).

## 2) قواعد البيانات / SQL

```python
# SQL فوري بدون إعداد — يعمل دائماً (sqlite مدمج)
top = df.sql("SELECT city, AVG(price) AS m FROM df GROUP BY city ORDER BY m DESC")

# استعلام عبر عدة جداول
from bandas import sql_query
out = sql_query("SELECT * FROM a JOIN b ON a.k = b.k WHERE a.v > 10", {"a": a, "b": b})

# موصل موحد
from bandas.sql import Database
db = Database.sqlite("app.db")          # أو :memory:
# db = Database.duckdb("data.duckdb")   # يحتاج duckdb
# db = Database.postgres(user=..., password=..., host=..., db=...)
db.write(df, "sales")
print(db.query("SELECT COUNT(*) AS n FROM sales").to_pandas())
print(db.tables())

# pandas API
df.to_sql("sales", "sqlite:///app.db")
df2 = bd.read_sql("SELECT * FROM sales", "sqlite:///app.db")
```

## 3) سلاسل زمنية ونوافذ

```python
s = df["price"]
s.rolling(7).mean()        # متوسط متحرك
s.expanding().sum()
s.ewm(span=10).mean()
s.shift(1); s.diff(); s.pct_change(); s.cumsum()
df.rolling(30).mean(); df.ewm(alpha=0.3).mean()

bd.to_datetime(["2024-01-01", "2024-02-01"])
bd.date_range("2024-01-01", periods=12, freq="ME")
df["d"].dt.year / .month / .day / .dayofweek
```

## 4) نصوص وفئات وإعادة تشكيل

```python
df["name"].str.lower().str.strip().str.contains("ali")
df["name"].str.replace("a", "@").str.split(" ", expand=True)
bd.get_dummies(df, columns=["city"])
bd.melt(df, id_vars=["city"])
bd.pivot_table(df, values="price", index="city", columns="year", aggfunc="mean")
bd.crosstab(df["city"], df["year"])
bd.cut(df["age"], bins=[0,18,60,100])
bd.qcut(df["price"], 4)
```

## 5) تجهيز البيانات (Preprocessing)

```python
bd.SimpleImputer(strategy="mean").fit_transform(df)
bd.StandardScaler().fit_transform(df)
bd.MinMaxScaler().fit_transform(df)
bd.RobustScaler().fit_transform(df)
bd.LabelEncoder().fit_transform(df["city"])
bd.OneHotEncoder(columns=["city"]).fit_transform(df)
```

## 6) تعلم الآلة

```python
X_train, X_test, y_train, y_test = bd.train_test_split(X, y, test_size=0.2, random_state=0)
list(bd.KFold(n_splits=5).split(X))
bd.accuracy_score(y, pred); bd.f1_score(y, pred)
bd.mean_squared_error(y, pred); bd.r2_score(y, pred)

# يعمل مباشرة مع sklearn
from sklearn.ensemble import RandomForestClassifier
X, y = bd.to_sklearn_Xy if False else (df.drop(columns=["target"]).to_numpy(), df["target"].to_numpy())
clf = RandomForestClassifier().fit(X_train_numpy, y_train_numpy)
```

## 7) التعلم العميق

```python
X_t, y_t = df.to_torch(target="label")          # torch.Tensor
loader = bd.dl.torch_dataset(df, target="label", batch_size=64)
ds = bd.dl.tf_dataset(df, target="label")       # tf.data.Dataset
```

## 8) ملفات ضخمة (Lazy / out-of-core)

```python
lf = bd.read_csv_chunked("big.csv", chunksize=100_000)
out = lf.filter(lambda d: d[d["x"] > 0]).select(["x","y"]).collect()
print(lf.sum("x"), lf.mean("x"), lf.count())
```

## 9) تصوير سريع

```python
df.plot(kind="line", x="date", y="price")
df.plot(kind="bar", x="city", y="sales")
df["price"].hist(bins=50)
bd.viz.scatter(df, "a", "b"); bd.viz.heatmap_corr(df)
```

## 10) IO شامل

`read_csv / read_parquet / read_json / read_excel / read_html / read_feather / read_orc / read_sql` +
`to_csv / to_parquet / to_json / to_excel / to_sql` — كلها بنفس توقيع pandas.

## هيكل المشروع

```
bandas/
  __init__.py      # الواجهة العامة + concat/merge/options/show_versions
  series.py        # Series + str/dt + rolling/ewm + cum*/shift/diff + ML helpers
  dataframe.py     # DataFrame + SQL + stats + reshape + viz + torch
  groupby.py       # GroupBy السريع + std/var/median/first/last/nunique
  io.py            # كل قارئات الملفات + chunked
  sql.py           # sql_query + read_sql/to_sql + Database الموحد
  reshape.py       # get_dummies/melt/pivot/crosstab/cut/qcut
  window.py        # Rolling/Expanding/EWM
  strings.py       # Series.str
  datetimes.py     # Series.dt + to_datetime/date_range
  preprocessing.py # Imputer/Scaler/Encoder
  ml.py            # split/CV/metrics/sklearn bridge
  dl.py            # torch/tensorflow bridges
  stats.py         # corr/cov/outliers/summary
  lazy.py          # LazyFrame خارج الذاكرة
  viz.py           # رسوم سريعة
  ops.py / utils.py
tests/test_basic.py + test_advanced.py (20 اختباراً)
```

## اختبار و Benchmark

```bash
pytest -q
python bench/bench.py
python -c "import bandas; bandas.show_versions()"
```
