"""Benchmark: bandas vs pandas — same ops, timed."""
import time
import numpy as np

N = 500_000


def timed(fn, n=3):
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    import pandas as pd
    import bandas as bd

    rng = np.random.default_rng(0)
    a = rng.random(N)
    b = rng.integers(0, 1000, N)
    g = rng.integers(0, 100, N).astype(str)

    pdf = pd.DataFrame({"a": a, "b": b, "g": g})
    fdf = bd.DataFrame({"a": a, "b": b, "g": g})

    cases = [
        ("sum(a)", lambda d: d["a"].sum()),
        ("mean(a)", lambda d: d["a"].mean()),
        ("filter b>500", lambda d: d[d["b"] > 500] if hasattr(d, "__getitem__") else None),
        ("groupby g mean", lambda d: d.groupby("g")["a"].mean()),
        ("sort by b", lambda d: d.sort_values("b")),
    ]

    print(f"N={N:,}")
    print(f"{'op':<16} {'pandas':>10} {'bandas':>10} {'speedup':>8}")
    for name, fn in cases:
        tp = timed(lambda: fn(pdf))
        tf = timed(lambda: fn(fdf))
        print(f"{name:<16} {tp:>10.4f}s {tf:>10.4f}s {tp / max(tf, 1e-9):>7.2f}x")


if __name__ == "__main__":
    main()
