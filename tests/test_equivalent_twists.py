"""Merging twists that are equivalent by the upper layer's lattice symmetry."""

import numpy as np
import pytest

from MLM.search import SearchParams, canonical_twist, find_candidates, lattice_rotation_order
from MLM.structure import MoireStructure


@pytest.fixture(scope="module")
def bto():
    return MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")


def twists(cells):
    return sorted(round(c.twist, 2) for c in cells)


def test_lattice_rotation_orders():
    hexagonal = np.array([[2.4673, 0.0], [-2.4673 / 2, 2.4673 * 3 ** 0.5 / 2]])
    assert lattice_rotation_order(np.eye(2) * 3.99) == 4
    assert lattice_rotation_order(hexagonal) == 6
    assert lattice_rotation_order(np.diag([3.0, 4.0])) == 2
    graphene = MoireStructure.from_poscar("moire_structures/graphene/graphene_monolayer.vasp", "g")
    assert lattice_rotation_order(graphene.lattice_vectors) == 6


def test_canonical_twist():
    assert canonical_twist(112.62, 4, False) == pytest.approx(22.62)
    assert canonical_twist(67.38, 4, False) == pytest.approx(67.38)
    assert canonical_twist(67.38, 4, True) == pytest.approx(22.62)
    assert canonical_twist(-22.62, 4, True) == pytest.approx(22.62)


def test_rotation_merges_twists_one_period_apart(bto):
    window = dict(nmin=-10, nmax=10, degstep=0.01)
    plain = find_candidates(bto, bto, SearchParams(degmin=22.0, degmax=113.0, **window))
    assert {22.62, 112.62} <= set(twists(plain))  # same structure, 90° apart
    merging = SearchParams(degmin=22.0, degmax=113.0, merge_equivalent_twists="rotation", **window)
    merged = find_candidates(bto, bto, merging)
    assert 22.62 in twists(merged) and 112.62 not in twists(merged)
    assert 67.38 in twists(merged)  # = 90 − 22.62: a mirror image, not a rotation


def test_mirror_merges_theta_and_minus_theta(bto):
    window = dict(degmin=22.0, degmax=68.0, nmin=-10, nmax=10)
    plain = find_candidates(bto, bto, SearchParams(**window))
    assert {22.62, 67.38} <= set(twists(plain))
    merged = find_candidates(bto, bto, SearchParams(merge_equivalent_twists="rotation+mirror", **window))
    assert 22.62 in twists(merged) and 67.38 not in twists(merged)
    assert len(merged) < len(plain)


def test_merge_is_off_by_default_and_validated(bto):
    with pytest.raises(ValueError):
        SearchParams(merge_equivalent_twists="all")
    assert SearchParams().merge_equivalent_twists == "none"
