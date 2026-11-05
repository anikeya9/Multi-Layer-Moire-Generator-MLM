"""Benchmark: index.csv vs index.sqlite for a large run (default 100 000 rows).

Rows are synthetic (no YAML files needed). Times: writing the index, loading the whole
table, and a typical ``find`` (n_layers=3, max_cumulative_residual=1e-4).

Run from the repository root:  ``python dev/benchmark_index_formats.py [N]``
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from MLM.index import INDEX_COLUMNS, find, load_index, write_index


def synthetic_rows(n_rows: int) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for number in range(n_rows):
        residual = float(10 ** rng.uniform(-6, -2))
        row = dict.fromkeys(INDEX_COLUMNS)
        row.update(id=f"bto_3_x_{number:07d}", level=2, n_layers=int(rng.integers(2, 4)), material="bto,bto,bto",
                   parents="a;b", twist=float(rng.uniform(1, 45)), angle_between_vectors=90.0, residual=residual,
                   cumulative_residual=residual, max_layer_residual=residual * 1.5, max_strain=0.0,
                   cell_area=float(rng.uniform(100, 1e5)), a1_length=30.0, a2_length=30.0, units="angstrom",
                   round=2, passed_filter=bool(residual < 1e-3), path=f"yamls/{number}.yaml")
        rows.append(row)
    return rows


def main(n_rows: int = 100_000) -> None:
    rows = synthetic_rows(n_rows)
    print(f"rows: {n_rows}")
    for index_format in ("csv", "sqlite"):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            started = time.perf_counter()
            path = write_index(run_dir, rows, index_format)
            write_s = time.perf_counter() - started
            started = time.perf_counter()
            load_index(run_dir)
            load_s = time.perf_counter() - started
            started = time.perf_counter()
            hits = find(run_dir, n_layers=3, max_cumulative_residual=1e-4)
            find_s = time.perf_counter() - started
            print(f"{index_format:6s}  size {path.stat().st_size / 1e6:6.1f} MB  write {write_s:5.2f} s  "
                  f"load all {load_s:5.2f} s  find {find_s:5.3f} s  ({len(hits)} hits)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
