"""Benchmark: filtering N candidates by reading N YAML files vs reading index.csv.

Answers the open question "do we need an index table, or can we just glob the YAMLs?".
Creates N synthetic (but schema-valid) trilayer recipes in a temporary run directory,
then times:

1. glob + ``MoireStructure.from_yaml`` on every file, then filter in Python;
2. ``load_index`` (one CSV read) + a pandas filter;
3. ``find`` (the query helper, = 2 plus the material check).

Run from the repository root:  ``python dev/benchmark_yaml_vs_index.py [N]``
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from MLM.index import RunStore, find, load_index
from MLM.structure import MoireStructure


def synthetic_trilayer(rng: np.random.Generator, number: int) -> MoireStructure:
    """A schema-valid level-2 record with random numbers (no search needed)."""
    twists = sorted(rng.uniform(1.0, 45.0, size=2))
    residual = float(10 ** rng.uniform(-6, -2))
    cell = rng.normal(scale=30.0, size=(2, 2))
    return MoireStructure(
        level=2, n_layers=3, material=["bto"] * 3,
        parents=[f"bto_2_{twists[0]:.2f}_parent{number}", "bto_1_root"],
        twist=twists[1], lattice_vectors=cell, residual=residual, cumulative_residual=residual,
        source_poscar=["moire_structures/bto/bto_1-layer.vasp"] * 3,
        twist_sequence=twists, layer_rotations=[0.0, twists[0], twists[1]],
        layer_residuals=[0.0, residual, residual],
        coefficients={"lower": [[1, 2], [-2, 1]], "upper": [[-8, 1], [-1, -8]]},
    )


def main(n_records: int = 2000) -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        store = RunStore(run_dir)
        started = time.perf_counter()
        for number in range(n_records):
            store.add(synthetic_trilayer(rng, number), round_number=2, passed_filter=True)
        store.write_index()
        write_seconds = time.perf_counter() - started

        yaml_files = sorted((run_dir / "yamls").glob("*.yaml"))
        sizes = [path.stat().st_size for path in yaml_files]
        index_size = (run_dir / "index.csv").stat().st_size

        started = time.perf_counter()
        records = [MoireStructure.from_yaml(path) for path in yaml_files]
        hits_yaml = [r.id for r in records if r.cumulative_residual <= 1e-4]
        yaml_seconds = time.perf_counter() - started

        started = time.perf_counter()
        table = load_index(run_dir)
        hits_index = table[table["cumulative_residual"] <= 1e-4]["id"].tolist()
        index_seconds = time.perf_counter() - started

        started = time.perf_counter()
        hits_find = find(run_dir, n_layers=3, max_cumulative_residual=1e-4, material="bto")["id"].tolist()
        find_seconds = time.perf_counter() - started

        assert sorted(hits_yaml) == sorted(hits_index) == sorted(hits_find)
        print(f"records:                       {n_records}")
        print(f"write YAMLs + index:           {write_seconds:.2f} s")
        print(f"YAML file size:                mean {np.mean(sizes):.0f} B, total {sum(sizes) / 1e6:.2f} MB")
        print(f"index.csv size:                {index_size / 1e6:.2f} MB ({index_size / n_records:.0f} B/row)")
        print(f"filter by reading all YAMLs:   {yaml_seconds:.2f} s ({1e3 * yaml_seconds / n_records:.2f} ms/file)")
        print(f"filter via load_index:         {index_seconds * 1e3:.1f} ms")
        print(f"filter via find():             {find_seconds * 1e3:.1f} ms")
        print(f"speed-up index vs YAML:        {yaml_seconds / index_seconds:.0f}x")
        print(f"matches:                       {len(hits_index)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
