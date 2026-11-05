"""The vectorised scanner must give exactly what match.py gives (and be faster)."""

import time

import numpy as np
import pytest

import MLM.fastscan as fast
import MLM.match as reference
from MLM.io import read_vasp
from MLM.search import SearchParams, find_candidates
from MLM.structure import MoireStructure


def in_plane(path):
    lattice, _, _ = read_vasp(path)
    return lattice["a"][:2], lattice["b"][:2]


BTO = in_plane("moire_structures/bto/bto_1-layer.vasp")
GRAPHENE = in_plane("moire_structures/graphene/graphene_monolayer.vasp")
PTO = in_plane("moire_structures/PTO_STO/pto_tetra_monolayer.vasp")
STO = in_plane("moire_structures/PTO_STO/STO_centered_unit_cell_cubic_monolayer.vasp")
BILAYER_CELL = (np.array([-11.971137285300001, 7.9807581902]), np.array([-7.9807581902, -11.971137285300001]))

CASES = {
    "bto bilayer": (BTO, BTO, 20.0, 24.0, 0.01, 5e-4, -10, 10),
    "bto trilayer (moiré cell)": (BILAYER_CELL, BTO, 36.5, 37.0, 0.01, 5e-4, -40, 40),
    "graphene": (GRAPHENE, GRAPHENE, 21.0, 22.5, 0.01, 5e-4, -10, 10),
    "pto/sto, dtol 9e-3": (PTO, STO, 25.0, 30.0, 0.01, 9e-3, -10, 10),
    "strain-size tolerance": (PTO, STO, 5.0, 10.0, 0.1, 0.3, -10, 10),
    "window through 0°": (BTO, BTO, -0.5, 0.5, 0.01, 5e-4, -5, 5),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_run_is_identical_to_match(case):
    (a1, a2), (g1, g2), *rest = CASES[case]
    ref = reference.run(a1, a2, g1, g2, *rest)
    new = fast.run(a1, a2, g1, g2, *rest)
    assert ref.columns == new.columns and ref.height == new.height
    for column in ref.columns:
        assert ref[column].to_list() == new[column].to_list(), column


def test_run_and_filter_is_identical():
    (a1, a2), (g1, g2), *rest = CASES["bto bilayer"]
    assert fast.run_and_filter(a1, a2, g1, g2, *rest, [90.0]) == reference.run_and_filter(a1, a2, g1, g2, *rest, [90.0])


def test_fast_scanner_is_faster():
    (a1, a2), (g1, g2), *rest = CASES["bto bilayer"]
    started = time.perf_counter()
    reference.run(a1, a2, g1, g2, *rest)
    slow = time.perf_counter() - started
    started = time.perf_counter()
    fast.run(a1, a2, g1, g2, *rest)
    assert time.perf_counter() - started < slow / 5


def test_include_axes_finds_axis_points():
    (a1, a2), (g1, g2) = BTO, BTO
    without = fast.run(a1, a2, g1, g2, 36.8, 36.9, 0.01, 5e-4, -5, 5)
    with_axes = fast.run(a1, a2, g1, g2, 36.8, 36.9, 0.01, 5e-4, -5, 5, include_axes=True)
    assert with_axes.height > without.height
    assert ((with_axes["n1"] == 0) | (with_axes["n2"] == 0)).any()
    assert not ((without["n1"] == 0) | (without["n2"] == 0)).any()


def test_max_per_angle():
    (a1, a2), (g1, g2) = BILAYER_CELL, BTO
    capped = fast.run(a1, a2, g1, g2, 36.86, 36.88, 0.01, 5e-4, -40, 40)
    unlimited = fast.run(a1, a2, g1, g2, 36.86, 36.88, 0.01, 5e-4, -40, 40, max_per_angle=None)
    assert capped.group_by("angle").len()["len"].max() == 25
    assert unlimited.height > capped.height


def test_search_results_identical_with_both_scanners():
    root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    params = dict(degmin=1.0, degmax=45.0, nmin=-10, nmax=10)
    fast_cells = find_candidates(root, root, SearchParams(**params))
    match_cells = find_candidates(root, root, SearchParams(scanner="match", **params))
    assert [c.id for c in fast_cells] == [c.id for c in match_cells] and len(fast_cells) == 19


def test_axis_option_needs_fast_scanner():
    with pytest.raises(ValueError):
        SearchParams(scanner="match", include_axes=True)
    with pytest.raises(ValueError):
        SearchParams(scanner="shiny")
