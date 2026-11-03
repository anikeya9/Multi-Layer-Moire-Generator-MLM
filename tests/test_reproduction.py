"""Reproduce results of the old notebook workflow through the new API.

Reference data (in the repository):

- ``moire_structures/bto/TBLBTO_<angle>_2.dat``: the BTO bilayer structures written by
  the old workflow; their file names list every bilayer twist that was found.
- ``moire_structures/bto/bto_bilayer_candidates.csv`` and
  ``moire_structures/graphene/graphene_bilayer_candidates.csv``: bilayer candidates
  (twist, coefficients n, moiré vectors A1/A2, residual ``vec_del``).
- ``moire_structures/*/..._trilayer_candidates_*.csv``: trilayer candidates.

The old CSVs were produced by the older O(N^4) finder (absolute ``vec_del < dtol``
criterion), the new path uses ``match.py`` (fractional tolerance), so residuals are
compared at the best grid angle and moiré vectors up to an equivalent basis choice.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MLM.search import SearchParams, find_candidates, hermite_normal_form
from MLM.structure import MoireStructure

BTO = "moire_structures/bto/bto_1-layer.vasp"
GRAPHENE = "moire_structures/graphene/graphene_monolayer.vasp"


def parse_vector(text: str) -> np.ndarray:
    """CSV cells look like '[ 6.16822815 -2.13673691  0.        ]'."""
    return np.array([float(x) for x in text.strip("[] ").split()])[:2]


def same_lattice(vectors_a: np.ndarray, vectors_b: np.ndarray, basis: np.ndarray) -> bool:
    """True if both cells span the same lattice (compared via integer coordinates in ``basis``)."""
    coeffs_a = np.rint(vectors_a @ np.linalg.inv(basis)).astype(int)
    coeffs_b = np.rint(vectors_b @ np.linalg.inv(basis)).astype(int)
    return bool((hermite_normal_form(coeffs_a) == hermite_normal_form(coeffs_b)).all())


def old_bto_bilayer_angles(max_angle: float = 45.0) -> list[float]:
    angles = []
    for path in Path("moire_structures/bto").glob("TBLBTO_*_2.dat"):
        angle = float(re.match(r"TBLBTO_([0-9.]+)_2\.dat", path.name).group(1))
        if angle < max_angle:
            angles.append(round(angle, 2))
    return sorted(angles)


@pytest.mark.slow
def test_bto_bilayer_full_window_matches_old_structures():
    """1–45°, n = ±10, goodangle 90° (Bi-layer_structure_finder.ipynb settings), ~7 s."""
    root = MoireStructure.from_poscar(BTO, "bto")
    params = SearchParams(degmin=1.0, degmax=45.0, degstep=0.01, dtol=5e-4, nmin=-10, nmax=10, goodangles=(90.0,))
    candidates = find_candidates(root, root, params)
    new_angles = sorted(round(c.twist, 2) for c in candidates)
    assert new_angles == old_bto_bilayer_angles()  # the 19 angles of the TBLBTO_*.dat files


def test_bto_bilayer_csv_vectors_and_residual():
    reference = pd.read_csv("moire_structures/bto/bto_bilayer_candidates.csv")
    best = reference.loc[reference["vec_del"].idxmin()]
    root = MoireStructure.from_poscar(BTO, "bto")
    params = SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    (bilayer,) = find_candidates(root, root, params)
    assert bilayer.twist == pytest.approx(best["degree"], abs=1e-6)
    old_cell = np.array([parse_vector(best["A1"]), parse_vector(best["A2"])])
    assert same_lattice(bilayer.lattice_vectors, old_cell, root.lattice_vectors)
    assert bilayer.residual == pytest.approx(best["vec_del"], rel=1e-3)


def test_graphene_bilayer_csv_vectors():
    reference = pd.read_csv("moire_structures/graphene/graphene_bilayer_candidates.csv")
    best = reference.loc[reference["vec_del"].idxmin()]
    root = MoireStructure.from_poscar(GRAPHENE, "graphene")
    params = SearchParams(degmin=21.5, degmax=22.0, nmin=-10, nmax=10, goodangles=(60.0,))
    (bilayer,) = find_candidates(root, root, params)
    assert bilayer.twist == pytest.approx(21.79, abs=0.011)
    old_cell = np.array([parse_vector(best["A1"]), parse_vector(best["A2"])])
    assert same_lattice(bilayer.lattice_vectors, old_cell, root.lattice_vectors)
    # Old finder: best at 21.80 with vec_del 1.5e-3 Å; new path's grid point 21.79 is closer.
    assert bilayer.residual <= best["vec_del"]


def test_bto_trilayer_csv_through_tree(tmp_path):
    """bilayer 22.62° + layer 3 vs ``bto_trilayer_candidates_22.62.csv`` (best row: 36.87°)."""
    from MLM.search import Filters, SearchSchedule, build_stack

    reference = pd.read_csv("moire_structures/bto/bto_trilayer_candidates_22.62.csv")
    best = reference.loc[reference["vec_del"].idxmin()]
    root = MoireStructure.from_poscar(BTO, "bto")
    schedule = SearchSchedule(
        SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,)),
        {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -40, "nmax": 40}},
    )
    result = build_stack([root] * 3, schedule, Filters(max_residual=1e-3), run_dir=tmp_path, log=None)
    (trilayer,) = result.final
    assert trilayer.twist == pytest.approx(best["degree"], abs=1e-6)  # 36.87
    assert trilayer.residual == pytest.approx(best["vec_del"], rel=1e-6)  # 5.747e-5 Å
    old_cell = np.array([parse_vector(best["A1"]), parse_vector(best["A2"])])
    assert same_lattice(trilayer.lattice_vectors, old_cell, root.lattice_vectors)


def test_graphene_trilayer_csv_through_tree(tmp_path):
    """graphene bilayer 21.79° + layer 3 vs ``graphene_trilayer_candidates_21.80.csv``.

    The old CSV has layer 3 at 21.75/21.80° (i.e. aligned with layer 2) and a cell of
    3x the bilayer moiré cell. ``match.scan`` skips coefficients with a zero component,
    so the bilayer cell itself (n = (1, 0), (0, 1)) can never be found; the smallest cell
    it can report is the same 3x cell.
    """
    from MLM.search import Filters, SearchSchedule, build_stack

    reference = pd.read_csv("moire_structures/graphene/graphene_trilayer_candidates_21.80.csv")
    best = reference.loc[reference["vec_del"].idxmin()]
    root = MoireStructure.from_poscar(GRAPHENE, "graphene")
    schedule = SearchSchedule(
        SearchParams(degmin=21.5, degmax=22.0, nmin=-10, nmax=10, goodangles=(60.0,)),
        {2: {"degmin": 21.7, "degmax": 21.9, "nmin": -40, "nmax": 40}},
    )
    result = build_stack([root] * 3, schedule, Filters(max_residual=1e-2), run_dir=tmp_path, log=None)
    old_cell = np.array([parse_vector(best["A1"]), parse_vector(best["A2"])])
    matches = [t for t in result.final if same_lattice(t.lattice_vectors, old_cell, root.lattice_vectors)]
    assert matches, [t.twist for t in result.final]
    assert min(abs(t.twist - 21.79) for t in matches) < 0.011
