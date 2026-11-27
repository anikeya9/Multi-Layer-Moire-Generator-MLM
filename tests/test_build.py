"""Tests for rebuilding atoms from recipes (milestone M4)."""

import numpy as np
import pytest
from ase.io import read

from MLM.build import build_atoms, replicas_needed, rotations_from_lineage, write_structure
from MLM.index import find, get
from MLM.search import Filters, SearchParams, SearchSchedule, build_stack
from MLM.structure import MoireStructure

SCHEDULE = SearchSchedule(
    SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10),
    {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -40, "nmax": 40}},
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("build")
    root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    build_stack([root] * 3, SCHEDULE, Filters(max_residual=1e-3), run_dir=run_dir, log=None)
    ids = {n: find(run_dir, n_layers=n)["id"].iloc[0] for n in (1, 2, 3)}
    return run_dir, ids


def test_rotations_from_lineage_match_stored(run):
    run_dir, ids = run
    record = get(run_dir, ids[3])
    assert rotations_from_lineage(run_dir, ids[3]) == pytest.approx(record.layer_rotations)
    assert rotations_from_lineage(run_dir, ids[1]) == [0.0]


def test_replicas_cover_the_cell():
    square = np.array([[4.0, 0.0], [0.0, 4.0]])
    n = replicas_needed(square, 40.0)
    # The patch -n..n-1 must contain the disc of radius 40 (plus one spare cell),
    # without being wastefully large.
    assert (n - 1) * 4.0 >= 40.0 - 1e-9 and n <= 12


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_build_atoms_counts_and_stacking(run, n_layers):
    run_dir, ids = run
    record = get(run_dir, ids[n_layers])
    built = build_atoms(record)
    assert built.counts_ok, (built.layer_counts, built.expected_counts)
    cells = round(record.cell_area / 15.923125322611096)
    assert built.composition == {"Ba": 2 * cells * n_layers, "O": 4 * cells * n_layers, "Ti": cells * n_layers}
    # Layers are stacked bottom-up with the notebook's 3 Å rule.
    z_max = built.atoms.groupby("layer")["z"].max().to_numpy()
    z_min = built.atoms.groupby("layer")["z"].min().to_numpy()
    assert (z_min[1:] > z_max[:-1]).all()
    np.testing.assert_allclose(built.cell[:2, :2], record.lattice_vectors)


def test_write_vasp_and_lammps(run, tmp_path):
    run_dir, ids = run
    vasp = write_structure(run_dir, ids[3], fmt="vasp")
    assert vasp == run_dir / "structures" / f"{ids[3]}.vasp"
    atoms = read(vasp)
    assert len(atoms) == 1365
    lammps = write_structure(run_dir, ids[3], fmt="lammps", out=tmp_path / "tri.data")
    text = lammps.read_text()
    assert "1365 atoms" in text and "Masses" in text
    with pytest.raises(ValueError):
        write_structure(run_dir, ids[3], fmt="xyz")


def test_write_structure_from_another_directory(run, monkeypatch, tmp_path):
    """POSCAR paths are relative to the repo root; they are found by walking up from the run."""
    run_dir, ids = run
    # The run lives in a temp dir, so point the search at the repository explicitly.
    from conftest import REPO_ROOT

    monkeypatch.chdir(tmp_path)
    out = write_structure(run_dir, ids[2], out=tmp_path / "bi.vasp", search_dirs=[REPO_ROOT])
    assert len(read(out)) == 182
