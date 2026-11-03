"""Tests for search_pair and its helpers (milestone M2)."""

import numpy as np
import pytest

from MLM.search import (
    CellMatch,
    Filters,
    SearchParams,
    cell_from_pair,
    deduplicate,
    find_candidates,
    hermite_normal_form,
    search_pair,
)
from MLM.structure import MoireStructure

# Narrow window around the known BTO bilayer angle 22.62° (fast: ~0.2 s).
NARROW_BTO = SearchParams(degmin=22.0, degmax=23.0, degstep=0.01, dtol=5e-4, nmin=-10, nmax=10, goodangles=(90.0,))


@pytest.fixture(scope="module")
def bto_bilayers():
    root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    return root, find_candidates(root, root, NARROW_BTO)


def test_search_pair_finds_known_bto_angle(bto_bilayers):
    root, candidates = bto_bilayers
    assert len(candidates) == 1  # 94 raw pairs collapse to one cell
    bilayer = candidates[0]
    assert bilayer.twist == pytest.approx(22.62, abs=1e-9)
    assert bilayer.residual == pytest.approx(3.391e-5, rel=1e-3)
    assert bilayer.angle_between_vectors == pytest.approx(90.0)
    assert bilayer.cell_area == pytest.approx(13 * root.cell_area, rel=1e-9)  # Σ13 cell


def test_child_record_fields(bto_bilayers):
    root, (bilayer,) = bto_bilayers
    assert bilayer.parents == [root.id, root.id]
    assert bilayer.level == 1 and bilayer.n_layers == 2
    assert bilayer.material == ["bto", "bto"]
    assert bilayer.source_poscar == root.source_poscar * 2
    assert bilayer.layer_rotations == [0.0, bilayer.twist]
    assert bilayer.twist_sequence == [bilayer.twist]
    assert bilayer.cumulative_residual == bilayer.residual
    assert bilayer.id.startswith("bto_2_22.62_")
    # Lattice vectors are A @ n in the lower frame, and right-handed.
    n = np.array(bilayer.coefficients["lower"])
    np.testing.assert_allclose(bilayer.lattice_vectors, n @ root.lattice_vectors)
    assert np.linalg.det(bilayer.lattice_vectors) > 0
    # Layer 1 is matched exactly by construction; layer 2 carries the residual.
    assert bilayer.layer_residuals[0] == pytest.approx(0.0, abs=1e-9)
    assert bilayer.layer_residuals[1] == pytest.approx(bilayer.residual, rel=1e-6)


def test_cumulative_residual_is_max_of_lineage(bto_bilayers):
    root, (bilayer,) = bto_bilayers
    params = SearchParams(degmin=36.5, degmax=37.5, nmin=-40, nmax=40, goodangles=(90.0,))
    trilayers = find_candidates(bilayer, root, params)
    assert trilayers
    for trilayer in trilayers:
        assert trilayer.cumulative_residual == max(trilayer.residual, bilayer.cumulative_residual)


def test_filters_drop_but_do_not_change_candidates(bto_bilayers):
    root, (bilayer,) = bto_bilayers
    assert search_pair(root, root, NARROW_BTO, Filters(max_residual=1e-6)) == []
    kept = search_pair(root, root, NARROW_BTO, Filters(max_residual=1e-4))
    assert kept == [bilayer]
    assert Filters(max_residual=1e-6).reasons_rejected(bilayer)
    assert Filters().passes(bilayer)
    assert Filters(max_residual=0.0).passes(root)  # roots always pass
    assert not Filters(max_layer_residual=1e-5).passes(bilayer)
    assert Filters(max_layer_residual=1e-4, max_length_mismatch=0.1).passes(bilayer)


def test_window_relative_to_lower_top_layer(bto_bilayers):
    root, (bilayer,) = bto_bilayers
    params = SearchParams(degmin=1.0, degmax=60.01, window_relative_to="lower_top_layer")
    assert params.window_for(root) == (1.0, 60.01)
    low, high = params.window_for(bilayer)
    assert low == pytest.approx(23.62) and high == pytest.approx(82.63)


def test_dedupe_off_returns_raw_pairs():
    root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    raw = find_candidates(root, root, SearchParams(degmin=22.6, degmax=22.63, nmin=-5, nmax=5, dedupe=False))
    deduped = find_candidates(root, root, SearchParams(degmin=22.6, degmax=22.63, nmin=-5, nmax=5))
    assert len(raw) > len(deduped) == 1


def test_hermite_normal_form_identifies_same_lattice():
    a = np.array([[3, -2], [2, 3]])
    b = np.array([[2, 3], [-3, 2]])  # same lattice, other basis
    c = np.array([[6, -4], [4, 6]])  # supercell
    assert (hermite_normal_form(a) == hermite_normal_form(b)).all()
    assert not (hermite_normal_form(a) == hermite_normal_form(c)).all()
    with pytest.raises(ValueError):
        hermite_normal_form(np.array([[1, 2], [2, 4]]))


def _row(angle, vector, n, m, delvec):
    return {"angle": angle, "matchpoint": vector, "n1": n[0], "n2": n[1], "m1": m[0], "m2": m[1], "delvec": delvec}


def test_cell_from_pair_is_right_handed():
    a = _row(10.0, [0.0, 1.0], [0, 1], [0, 1], 1e-4)
    b = _row(10.0, [1.0, 0.0], [1, 0], [1, 0], 2e-4)
    cell = cell_from_pair(a, b, 90.0)
    np.testing.assert_allclose(cell.vectors, [[1.0, 0.0], [0.0, 1.0]])
    assert cell.residual == 2e-4


def test_deduplicate_merges_neighbouring_grid_angles():
    def cell(twist, residual, scale=1):
        coeffs = np.array([[1, 0], [0, 1]]) * scale
        return CellMatch(twist, 90.0, coeffs.astype(float), coeffs, coeffs, residual)

    cells = [cell(10.00, 3e-3), cell(10.01, 1e-4), cell(10.02, 2e-3), cell(10.01, 2e-4, scale=2), cell(20.0, 5e-4)]
    survivors = deduplicate(cells, keep_supercells=False, cluster_tol=0.015)
    assert [(c.twist, c.residual) for c in survivors] == [(10.01, 1e-4), (20.0, 5e-4)]
    with_super = deduplicate(cells, keep_supercells=True, cluster_tol=0.015)
    assert len(with_super) == 3


def test_estimated_cost():
    assert SearchParams(degmin=1, degmax=2, degstep=0.01, nmin=-10, nmax=10).estimated_cost() == 100 * 19 * 19
