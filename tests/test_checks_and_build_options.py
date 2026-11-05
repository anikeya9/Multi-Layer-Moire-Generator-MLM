"""Structure checks, lateral layer shifts and the Poisson thickness response."""

import warnings

import numpy as np
import pytest

from MLM.build import build_atoms, check_structure, write_structure
from MLM.checks import check_built, space_group, to_ase
from MLM.index import get
from MLM.search import SearchParams, find_candidates
from MLM.structure import MoireStructure

RUN = "examples/runs/bto_trilayer_demo"
TRILAYER = "bto_3_22.62_36.87_015688"


@pytest.fixture(scope="module")
def trilayer():
    return get(RUN, TRILAYER)


def test_good_structure_passes_all_checks(trilayer):
    report = check_structure(RUN, TRILAYER)
    assert report.ok, str(report)
    assert report.info["n_atoms"] == 1365 and report.info["formula"] == "Ba390O780Ti195"
    assert report.info["min_distance"] > 1.5
    for layer in report.info["layers"]:
        assert layer["min_distance"] == pytest.approx(layer["primitive"], abs=1e-3)
    assert report.info["space_group"]  # spglib is installed in the test environment


def test_layers_too_close_are_flagged(trilayer):
    built = build_atoms(trilayer, interlayer_gap=-3.5)  # pushes layers into each other
    report = check_built(built)
    assert not report.ok and any("too close" in p or "overlap" in p for p in report.problems)


def test_missing_atoms_are_flagged(trilayer):
    built = build_atoms(trilayer)
    built.atoms = built.atoms.drop(index=built.atoms.index[:3])  # lose three atoms
    report = check_built(built, symmetry=False)
    assert any("atoms per layer" in p for p in report.problems)
    assert any("composition" in p for p in report.problems)


def test_write_structure_warns_on_failed_checks(trilayer, tmp_path):
    with pytest.warns(RuntimeWarning, match="FAILED"):
        write_structure(RUN, TRILAYER, out=tmp_path / "bad.vasp", interlayer_gap=-3.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # our check warnings only (ASE emits DeprecationWarnings)
        write_structure(RUN, TRILAYER, out=tmp_path / "good.vasp")


def test_lateral_shift_moves_one_layer_rigidly(trilayer):
    plain = build_atoms(trilayer)
    shifted = build_atoms(trilayer, shifts=[None, (0.5, 0.5), None])
    assert shifted.layer_counts == plain.layer_counts and check_built(shifted, symmetry=False).ok
    # Layer 2 sits at 22.62°: a shift of (a+b)/2 of its own cell, rotated into the lab frame.
    a = 3.9903790951
    angle = np.deg2rad(trilayer.layer_rotations[1])
    expected = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ [a / 2, a / 2]
    cell = trilayer.lattice_vectors.T  # columns A1, A2

    def fractional_set(built, layer):
        xy = built.atoms.loc[built.atoms["layer"] == layer, ["x", "y"]].to_numpy()
        return np.linalg.solve(cell, xy.T).T % 1.0

    moved = fractional_set(plain, 1) + np.linalg.solve(cell, expected)
    target = fractional_set(shifted, 1)
    # every shifted atom sits on a moved plain atom (periodic in the cell). Not exact:
    # layer 2 is commensurate only within its residual (7.6e-5 Å per cell vector), so
    # folding may pick another periodic image, off by up to a few 1e-4 Å.
    for point in target[:20]:
        delta = (moved - point + 0.5) % 1.0 - 0.5
        assert np.min(np.linalg.norm(delta @ cell.T, axis=1)) < 1e-3
    # layer 1 untouched
    np.testing.assert_allclose(np.sort(fractional_set(shifted, 0), axis=0), np.sort(fractional_set(plain, 0), axis=0))


def test_poisson_ratio_changes_thickness_of_strained_layers():
    pto = MoireStructure.from_poscar("moire_structures/PTO_STO/pto_tetra_monolayer.vasp", "pto")
    sto = MoireStructure.from_poscar("moire_structures/PTO_STO/STO_centered_unit_cell_cubic_monolayer.vasp", "sto")
    (cell,) = find_candidates(pto, sto, SearchParams(degmin=-1.0, degmax=1.0, degstep=0.1), strain_limits=(0.0, 1.0))
    eps = cell.strain_tensors()[1][0, 0]  # STO isotropic strain (≈ −0.847 %)
    nu = 0.25

    def thickness(built, layer):
        z = built.atoms.loc[built.atoms["layer"] == layer, "z"]
        return float(z.max() - z.min())

    plain = build_atoms(cell)
    poisson = build_atoms(cell, poisson_ratios=[None, nu])
    expected = 1.0 + (-nu / (1 - nu) * 2 * eps)  # compressed in-plane -> thicker
    assert thickness(poisson, 1) / thickness(plain, 1) == pytest.approx(expected, rel=1e-9)
    assert thickness(poisson, 0) == pytest.approx(thickness(plain, 0))  # unstrained PTO unchanged
    assert check_built(poisson, symmetry=False).ok
    with pytest.raises(ValueError):
        build_atoms(cell, poisson_ratios=[None, 0.7])
    with pytest.raises(ValueError):
        build_atoms(cell, poisson_ratios=[0.2])


def test_space_group_helper(trilayer):
    assert space_group(to_ase(build_atoms(get(RUN, "bto_1_406932"))))  # monolayer
