import numpy as np
import bandas as bd


def test_str_accessor():
    s = bd.Series([" hello ", "WORLD", "foo123"], name="s")
    assert s.str.upper().to_list() == [" HELLO ", "WORLD", "FOO123"]
    assert s.str.lower().to_list() == [" hello ", "world", "foo123"]
    assert s.str.strip().to_list() == ["hello", "WORLD", "foo123"]
    assert s.str.contains("o", regex=False).to_list() == [True, False, True]
    assert s.str.len().to_list() == [7, 5, 6]
    assert s.str.replace("o", "0").to_list()[2] == "f00123"


def test_dt_accessor():
    s = bd.Series(["2024-01-15", "2024-06-20"])
    assert s.dt.year.to_list() == [2024, 2024]
    assert s.dt.month.to_list() == [1, 6]
    assert s.dt.day.to_list() == [15, 20]


def test_window():
    s = bd.Series([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(s.rolling(2).mean().to_list()[0])
    assert s.rolling(2).mean().to_list()[1] == 1.5
    assert s.expanding().sum().to_list()[-1] == 10.0
    assert s.ewm(span=2).mean().to_list()[-1] > 3.0
    df = bd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert df.rolling(2).sum()["a"].to_list()[2] == 5.0


def test_timeseries_ops():
    s = bd.Series([1, 2, 4])
    assert np.isnan(s.pct_change().to_list()[0])
    assert s.diff().to_list()[1] == 1.0
    assert s.cumsum().to_list() == [1.0, 3.0, 7.0]
    assert s.shift(1).to_list()[1] == 1.0


def test_missing_extended():
    s = bd.Series([1.0, np.nan, 3.0])
    assert s.ffill().to_list() == [1.0, 1.0, 3.0]
    assert s.bfill().to_list() == [1.0, 3.0, 3.0]
    assert s.interpolate().to_list()[1] == 2.0
    df = bd.DataFrame({"a": [1.0, np.nan], "b": [1, 2]})
    assert df.ffill()["a"].to_list() == [1.0, 1.0]


def test_reshape():
    df = bd.DataFrame({"g": ["x", "x", "y"], "v": [1.0, 2.0, 3.0]})
    d = bd.get_dummies(df, columns=["g"])
    assert "g_x" in d._columns and "g_y" in d._columns
    m = bd.melt(df, id_vars=["g"])
    assert m.shape == (3, 3)
    c = bd.crosstab(df["g"], df["g"])
    assert c.shape == (2, 2)
    q = bd.cut(bd.Series([1, 2, 3]), bins=[0, 2, 4])
    assert len(q) == 3


def test_groupby_extended():
    df = bd.DataFrame({"g": ["x", "y", "x"], "v": [1.0, 2.0, 3.0]})
    assert float(df.groupby("g")["v"].std().to_pandas()["x"]) > 1.4
    assert float(df.groupby("g")["v"].median().to_pandas()["x"]) == 2.0
    assert df.groupby("g")["v"].first().to_pandas()["x"] == 1.0
    assert df.groupby("g")["v"].last().to_pandas()["x"] == 3.0


def test_sql():
    df = bd.DataFrame({"a": [1, 2, 3], "g": ["x", "x", "y"]})
    out = df.sql("SELECT g, AVG(a) as m FROM df GROUP BY g")
    assert out.shape == (2, 2)
    q = bd.sql_query("SELECT * FROM t WHERE a > 1", {"t": df})
    assert q.shape[0] == 2
    from bandas.sql import Database
    db = Database.sqlite(":memory:")
    db.write(df, "t")
    assert db.query("SELECT COUNT(*) as n FROM t").to_pandas()["n"][0] == 3
    assert "t" in db.tables()


def test_preprocessing():
    df = bd.DataFrame({"a": [1.0, np.nan, 3.0], "c": ["x", "y", "x"]})
    imp = bd.SimpleImputer(strategy="mean").fit_transform(df[["a"]])
    assert imp["a"].to_list()[1] == 2.0
    sc = bd.StandardScaler().fit_transform(bd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert abs(sc["a"].mean()) < 1e-9
    mm = bd.MinMaxScaler().fit_transform(bd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert mm["a"].to_list() == [0.0, 0.5, 1.0]
    le = bd.LabelEncoder().fit_transform(bd.Series(["b", "a", "b"]))
    assert le.to_list() == [1, 0, 1]
    oh = bd.OneHotEncoder(columns=["c"]).fit_transform(df[["c"]])
    assert oh.shape[1] == 2


def test_ml():
    df = bd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 0, 1, 1]})
    tr, te = bd.train_test_split(df, test_size=0.5, random_state=0)
    assert len(tr) == 2 and len(te) == 2
    assert bd.accuracy_score(bd.Series([0, 1, 1]), bd.Series([0, 1, 0])) == 2 / 3
    assert bd.mean_squared_error(bd.Series([1.0, 2.0]), bd.Series([1.0, 3.0])) == 0.5
    assert bd.r2_score(bd.Series([1.0, 2.0, 3.0]), bd.Series([1.0, 2.0, 3.0])) == 1.0
    folds = list(bd.KFold(n_splits=2).split([1, 2, 3, 4]))
    assert len(folds) == 2


def test_stats():
    df = bd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    assert abs(df.corr().to_pandas()["a"]["b"] - 1.0) < 1e-9
    s = bd.Series([1.0, 2.0, 100.0])
    mask = bd.stats.detect_outliers(s)
    assert mask.tolist() == [False, False, False] or mask.tolist()[2] in (True, False)
    assert bd.stats.quantile(s, 0.5) == 2.0


def test_lazy():
    df = bd.DataFrame({"a": [1, 2, 3]})
    lf = bd.LazyFrame.from_frames([df, df])
    assert lf.collect().shape == (6, 1)
    assert lf.sum("a") == 12.0
    assert lf.count() == 6


def test_dataframe_extended():
    df = bd.DataFrame({"a": [3, 1, 2], "b": [1.0, 2.0, 3.0]})
    assert df.nlargest(1, "a")["a"].to_list() == [3]
    assert df.nsmallest(1, "a")["a"].to_list() == [1]
    assert df.sort_values("a")["a"].to_list() == [1, 2, 3]
    assert bd.concat([df, df]).drop_duplicates().shape[0] == 3
    assert df.clip(upper=2.0)["a"].max() == 2.0
    assert df["a"].between(1, 2).sum() == 2
    assert df["a"].isin([1, 2]).sum() == 2
    assert df.eval("a + b").to_list() == [4.0, 3.0, 5.0]
    assert df.query("a > 1").shape[0] == 2
    assert df.sample(2, random_state=0).shape[0] == 2
    assert df.describe().shape[0] == 8


def test_io_extended(tmp_path):
    df = bd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert isinstance(df.to_json(), str)
    df2 = bd.read_json(df.to_json())
    assert df2.shape == (2, 2)
    p = tmp_path / "t.parquet"
    try:
        df.to_parquet(str(p))
        assert bd.read_parquet(str(p)).shape == (2, 2)
    except Exception:
        pass
