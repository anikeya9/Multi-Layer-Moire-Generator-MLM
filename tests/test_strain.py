"""Tests for heterostrain (branch with-straining).

PTO (a = 3.8796 Å) and STO (a = 3.9127 Å) are square lattices with a 0.85 % mismatch,
which makes them a good test case: at twist 0 the trivial cell needs exactly
a_PTO / a_STO − 1 = −0.847 % strain on STO.
"""

import warnings

import numpy as np
import pytest
import yaml

from MLM.build import build_atoms, write_structure
from MLM.index import find, get, load_index
from MLM.search import Filters, SearchParams, SearchSchedule, build_stack, find_candidates
from MLM.spec import parse_stack_spec
from MLM.strain import fit_strain, scan_tolerance, strain_magnitude, strain_tensor
from MLM.structure import MoireStructure

PTO = "moire_structures/PTO_STO/pto_tetra_monolayer.vasp"
STO = "moire_structures/PTO_STO/STO_centered_unit_cell_cubic_monolayer.vasp"
A_PTO, A_STO = 3.8795519, 3.91270131
MISMATCH = A_PTO / A_STO - 1.0  # strain that puts STO onto the PTO lattice (≈ −0.847 %)


@pytest.fixture(scope="module")
def pto():
    return MoireStructure.from_poscar(PTO, "pto")


@pytest.fixture(scope="module")
def sto():
    return MoireStructure.from_poscar(STO, "sto")


# --- strain.py ----------------------------------------------------------------------
def test_strain_tensor_forms():
    assert strain_tensor("isotropic", np.array([0.01])).tolist() == [[0.01, 0.0], [0.0, 0.01]]
    assert strain_tensor("axial", np.array([0.01, -0.02])).tolist() == [[0.01, 0.0], [0.0, -0.02]]
    assert strain_tensor("full", np.array([0.01, -0.02, 0.003])).tolist() == [[0.01, 0.003], [0.003, -0.02]]
    with pytest.raises(ValueError):
        strain_tensor("shear", np.array([0.0]))
    assert strain_magnitude(np.array([[0.01, 0.0], [0.0, -0.02]])) == pytest.approx(0.02)
    assert strain_magnitude(None) == 0.0


SQUARE_PTO = np.eye(2) * A_PTO
SQUARE_STO = np.eye(2) * A_STO
CELL = np.array([[3, 2], [-2, 3]])


@pytest.mark.parametrize("mode", ["isotropic", "axial", "full"])
@pytest.mark.parametrize("limits, lower_share", [((0.0, 0.02), 0.0), ((0.02, 0.0), 1.0),
                                                  ((0.01, 0.01), 0.5), ((0.005, 0.015), 0.25)])
def test_fit_is_exact_and_shares_in_proportion_to_limits(mode, limits, lower_share):
    fit = fit_strain(CELL @ SQUARE_PTO, SQUARE_STO, CELL, 0.3, mode, *limits)
    assert fit.residual < 1e-9  # exact
    assert fit.twist_correction == pytest.approx(-0.3, abs=1e-9)  # the true twist is 0
    lower = 0.0 if fit.lower_strain is None else fit.lower_strain[0, 0]
    upper = 0.0 if fit.upper_strain is None else fit.upper_strain[0, 0]
    # (1 + e_lower) a_PTO = (1 + e_upper) a_STO, and the lower layer takes `lower_share`
    assert (1 + lower) * A_PTO == pytest.approx((1 + upper) * A_STO, rel=1e-10)
    total = abs(lower) + abs(upper)
    assert abs(lower) / total == pytest.approx(lower_share, abs=0.01)
    assert (fit.lower_strain is None) == (limits[0] == 0)


def test_isotropic_cannot_fix_shape_but_axial_and_full_can():
    rectangular = np.diag([A_STO, A_STO * 1.003])  # upper cell with a different aspect ratio
    iso = fit_strain(CELL @ SQUARE_PTO, rectangular, CELL, 0.0, "isotropic", 0.0, 0.02)
    axial = fit_strain(CELL @ SQUARE_PTO, rectangular, CELL, 0.0, "axial", 0.0, 0.02)
    full = fit_strain(CELL @ SQUARE_PTO, rectangular, CELL, 0.0, "full", 0.0, 0.02)
    assert iso.residual > 1e-3
    assert axial.residual < 1e-9 and full.residual < 1e-9
    assert axial.upper_strain[0, 0] != pytest.approx(axial.upper_strain[1, 1])  # anisotropic


def test_full_mode_handles_shear():
    sheared = np.array([[A_STO, 0.0], [0.004, A_STO]])  # a2 tilted a little: needs shear
    full = fit_strain(CELL @ SQUARE_PTO, sheared, CELL, 0.0, "full", 0.0, 0.02)
    axial = fit_strain(CELL @ SQUARE_PTO, sheared, CELL, 0.0, "axial", 0.0, 0.02)
    assert full.residual < 1e-9 and axial.residual > full.residual
    assert full.upper_strain[0, 1] != 0.0


def test_scan_tolerance_grows_with_strain_and_is_capped():
    base = scan_tolerance(5e-4, 0.0, 10, SQUARE_PTO, SQUARE_STO, 0.0)
    assert base == 5e-4
    assert scan_tolerance(5e-4, 0.01, 10, SQUARE_PTO, SQUARE_STO, 0.1) > 0.05
    assert scan_tolerance(5e-4, 0.5, 10, SQUARE_PTO, SQUARE_STO, 0.1) == 0.45


# --- search -------------------------------------------------------------------------------
NEAR_ZERO = SearchParams(degmin=-1.0, degmax=1.0, degstep=0.1, nmin=-10, nmax=10)


def test_aligned_cell_needs_exactly_the_lattice_mismatch(pto, sto):
    (cell,) = find_candidates(pto, sto, NEAR_ZERO, strain_limits=(0.0, 1.0))
    assert cell.twist == pytest.approx(0.0, abs=1e-9)
    assert cell.residual < 1e-9 and cell.worst_layer_residual < 1e-9
    assert cell.layer_strains[0] == [[0.0, 0.0], [0.0, 0.0]]  # PTO rigid
    assert cell.layer_strains[1][0][0] == pytest.approx(100 * MISMATCH, rel=1e-9)
    assert cell.max_strain == pytest.approx(100 * abs(MISMATCH), rel=1e-9)


def test_strain_limit_excludes_cells(pto, sto):
    assert find_candidates(pto, sto, NEAR_ZERO, strain_limits=(0.0, 0.5)) == []
    (shared,) = find_candidates(pto, sto, NEAR_ZERO, strain_limits=(0.5, 0.5))  # 0.42 % each
    assert shared.layer_strains[0][0][0] == pytest.approx(-shared.layer_strains[1][0][0], rel=0.01)
    assert shared.max_strain < 0.5


def test_twists_stay_inside_the_window(pto, sto):
    params = SearchParams(degmin=5.0, degmax=10.0, degstep=0.1, nmin=-10, nmax=10)
    cells = find_candidates(pto, sto, params, strain_limits=(0.0, 1.0))
    assert cells and all(5.0 <= c.twist < 10.0 for c in cells)
    twists = [round(c.twist, 2) for c in cells]
    assert 7.13 in twists  # the unstrained 7.13° cell (0.024 Å off) becomes exact
    exact = [c for c in cells if round(c.twist, 2) == 7.13][0]
    assert exact.residual < 1e-9 and exact.max_strain < 0.1


def test_blocks_cannot_be_strained(pto, sto):
    (bilayer,) = find_candidates(pto, sto, NEAR_ZERO, strain_limits=(0.0, 1.0))
    with pytest.raises(ValueError):
        find_candidates(bilayer, pto, NEAR_ZERO, strain_limits=(1.0, 0.0))
    with pytest.raises(ValueError):
        find_candidates(pto, sto, NEAR_ZERO, strain_limits=(-1.0, 0.0))


def test_unstrained_ids_are_unchanged(pto):
    """Recipes without strain keep the ids they had before strain support (hash rule)."""
    bto = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    assert bto.id == "bto_1_406932"
    (bilayer,) = find_candidates(bto, bto, SearchParams(degmin=22.0, degmax=23.0))
    assert bilayer.id == "bto_2_22.62_1fdb6f" and not bilayer.is_strained


def test_yaml_round_trip_keeps_strain(pto, sto, tmp_path):
    (cell,) = find_candidates(pto, sto, NEAR_ZERO, strain_limits=(0.0, 1.0))
    again = MoireStructure.from_yaml(cell.to_yaml(tmp_path))
    assert again == cell and again.is_strained
    data = yaml.safe_load((tmp_path / f"{cell.id}.yaml").read_text())
    assert data["layer_strains"][1][0][0] == pytest.approx(100 * MISMATCH)


def test_old_recipes_without_layer_strains_load(pto):
    data = pto.to_dict()
    del data["layer_strains"]
    assert MoireStructure.from_dict(data).layer_strains == [[[0.0, 0.0], [0.0, 0.0]]]


# --- driver, spec, build ---------------------------------------------------------------
@pytest.fixture(scope="module")
def strained_trilayer(tmp_path_factory, pto, sto):
    run_dir = tmp_path_factory.mktemp("strained")
    schedule = SearchSchedule(SearchParams(degmin=9.0, degmax=10.0, degstep=0.1, nmin=-10, nmax=10),
                              {2: {"nmin": -40, "nmax": 40}})
    result = build_stack([pto, sto, pto], schedule, Filters(max_residual=1e-6), run_dir=run_dir,
                         log=None, strain_limits=[0.0, 1.0, 1.0])
    return run_dir, result


def test_strained_trilayer_through_the_tree(strained_trilayer):
    run_dir, result = strained_trilayer
    assert result.final
    for trilayer in result.final:
        assert trilayer.layer_strains[0] == [[0.0, 0.0], [0.0, 0.0]]  # layer 1 limit 0
        assert trilayer.worst_layer_residual < 1e-9
        bilayer = get(run_dir, trilayer.parents[0])
        assert trilayer.layer_strains[:2] == bilayer.layer_strains  # the block stayed rigid
    table = load_index(run_dir)
    assert (table["max_strain"] <= 1.0 + 1e-9).all()
    assert not find(run_dir, n_layers=3, max_strain=1.0).empty
    spec = yaml.safe_load((run_dir / "stack_spec.yaml").read_text())
    assert [layer.get("strain", 0.0) for layer in spec["layers"]] == [0.0, 1.0, 1.0]


def test_strained_structures_build_exactly(strained_trilayer, tmp_path):
    run_dir, result = strained_trilayer
    from conftest import REPO_ROOT

    record = sorted(result.final, key=lambda r: r.cell_area)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a count mismatch would warn
        built = build_atoms(record, search_dirs=[REPO_ROOT])
    assert built.counts_ok
    # Strained STO layer: its in-plane cell area is scaled by det(I + ε).
    sto_strain = record.strain_tensors()[1]
    expected_sto = round(record.cell_area / (A_STO ** 2 * np.linalg.det(np.eye(2) + sto_strain))) * 7
    assert built.layer_counts[1] == expected_sto
    out = write_structure(run_dir, record.id, out=tmp_path / "s.vasp", search_dirs=[REPO_ROOT])
    assert out.exists()


def test_spec_layer_strain_and_mode():
    spec = parse_stack_spec({
        "name": "s",
        "layers": [{"poscar": PTO, "material": "pto"}, {"poscar": STO, "material": "sto", "strain": 1.5}],
        "search": {"default": {"strain_mode": "full"}},
    })
    assert [layer.strain for layer in spec.layers] == [0.0, 1.5]
    assert spec.schedule.params_for_round(1).strain_mode == "full"
    with pytest.raises(ValueError):
        parse_stack_spec({"name": "s", "layers": [{"poscar": PTO, "material": "pto", "strain": -1}]})
    with pytest.raises(ValueError):
        SearchParams(strain_mode="shear")


def test_cells_that_strain_cannot_close_are_dropped(pto, sto):
    """At 45° a 3:2 PTO rectangle meets a 4:3 STO rectangle: isotropic strain cannot fix
    the aspect ratio (best fit leaves 0.88 Å), so it is not a strained match."""
    params = SearchParams(degmin=44.5, degmax=45.0, degstep=0.1, nmin=-10, nmax=10)
    assert find_candidates(pto, sto, params, strain_limits=(0.0, 1.0)) == []
    loose = SearchParams(degmin=44.5, degmax=45.0, degstep=0.1, nmin=-10, nmax=10, strain_residual_tol=1.0)
    kept = find_candidates(pto, sto, loose, strain_limits=(0.0, 1.0))
    assert kept and min(c.residual for c in kept) > 0.5
