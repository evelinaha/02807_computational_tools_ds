
## Graph clustering part

- Code: `src/ctds/graph/` (import as `ctds.graph`; the package was renamed from
  `02807_computational_tools_ds` because module names cannot start with a digit).
- Notebook: `notebooks/graph_clustering.ipynb` (end to end; `CTDS_SAMPLE=1` quick run, `CTDS_SAMPLE=0` full data).
- Data: unzip `books_50pct.zip` into `../data/books_50pct/` next to the repo (or set `CTDS_DATA_DIR`).
  Data and caches are never committed.
- Decisions: `docs/graph_clustering_decisions.md`. Results: `docs/graph_clustering_results.md`.
  Work log: `docs/graph_clustering_worklog.md`.
- Data (how it was pulled, unzipped, inspected and loaded): `docs/data_acquisition.md`;
  Step-0 inspection: `uv run python scripts/inspect_data.py`.
- Project report (draft): `docs/report/project_report.md`.
- Tests: `uv run pytest`.
