# Changelog

All notable changes to **bandas** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.1] - 2026-09-11
### Fixed
- Banner uses an absolute URL so it renders on PyPI as well as GitHub.

## [0.4.0] - 2026-09-11
### Added
- Full English documentation and project README.
- Visual identity: logo, banner, PyPI/GitHub badges.
- GitHub Actions CI (Python 3.11–3.13): tests + benchmark smoke.
- Rich PyPI metadata: authors, classifiers, project URLs.

## [0.3.0] - 2026-09-11
### Changed
- Single-command install: speed, database, ML, visualization and Excel
  dependencies are now installed by default (`pip install bandas-df`).
- Published on PyPI as `bandas-df` (import name stays `bandas`).
### Kept optional
- `torch` / `tensorflow` remain behind the `[dl]` extra (multi-GB size,
  no wheels for the newest Python versions).

## [0.2.0] - 2026-09-11
### Added
- Renamed `fandas` → `bandas` (repo: `salim-studio/bandas`).
- Universal data platform: SQL/`Database` connector, time-series windows
  (`rolling`/`expanding`/`ewm`), `str`/`dt` accessors, reshape
  (`get_dummies`/`melt`/`pivot_table`/`crosstab`/`cut`/`qcut`),
  preprocessing (imputers, scalers, encoders), ML helpers
  (`train_test_split`, `KFold`, metrics), DL bridges (torch/TensorFlow),
  `LazyFrame` out-of-core reader, quick plotting.

## [0.1.x]
- Original `fandas` releases: pandas-compatible core (`DataFrame`,
  `Series`, `groupby`, fast CSV/Parquet IO) with NumPy fast paths and
  optional Numba/PyArrow acceleration.
