import numpy as np
import bandas as bd


def test_series():
    s = bd.Series([1, 2, 3], name="a")
    assert s.sum() == 6
    assert s.mean() == 2.0
    assert (s + 1).to_list() == [2, 3, 4]
    assert s.value_counts().sum() == 3


def test_dataframe_basic():
    df = bd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert df.shape == (3, 2)
    assert df["a"].sum() == 6
    assert df.head(2).shape == (2, 2)
    df["c"] = [7, 8, 9]
    assert "c" in df
    assert df.sort_values("a", ascending=False)["a"].to_list() == [3, 2, 1]


def test_groupby():
    df = bd.DataFrame({"g": ["x", "y", "x"], "v": [1.0, 2.0, 3.0]})
    m = df.groupby("g")["v"].mean()
    d = dict(zip(list(m.index), list(m.to_numpy())))
    assert d["x"] == 2.0 and d["y"] == 2.0


def test_merge_concat():
    a = bd.DataFrame({"k": [1, 2], "v": [10, 20]})
    b = bd.DataFrame({"k": [2, 3], "w": [200, 300]})
    j = a.merge(b, on="k", how="inner")
    assert j.shape[0] == 1
    c = bd.concat([a, a], ignore_index=True)
    assert c.shape[0] == 4


def test_pandas_roundtrip():
    df = bd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pdf = df.to_pandas()
    df2 = bd.from_pandas(pdf)
    assert df2.equals(df)


def test_io(tmp_path):
    df = bd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    p = tmp_path / "t.csv"
    df.to_csv(str(p), index=False)
    df2 = bd.read_csv(str(p))
    assert df2.shape == (3, 2)
