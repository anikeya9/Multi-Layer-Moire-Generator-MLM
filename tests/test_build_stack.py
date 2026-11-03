"""Tests for the pairing planner, per-round parameters and the tree driver (M3)."""

import pytest
import yaml

from MLM.index import find, lineage, load_index
from MLM.search import (
    Filters,
    SearchParams,
    SearchSchedule,
    Step,
    build_stack,
    plan_rounds,
)
from MLM.structure import MoireStructure

BTO = "moire_structures/bto/bto_1-layer.vasp"

# Narrow windows around known commensurate angles keep these tests fast.
SCHEDULE_3 = SearchSchedule(
    SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,)),
    {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -40, "nmax": 40}},
)


# --- planner -----------------------------------------------------------------
EXPECTED_PLANS = {
    1: [],
    2: [[Step((0, 1), (1, 2))]],
    3: [[Step((0, 1), (1, 2)), Step((2, 3))], [Step((0, 2), (2, 3))]],
    4: [[Step((0, 1), (1, 2)), Step((2, 3), (3, 4))], [Step((0, 2), (2, 4))]],
    5: [
        [Step((0, 1), (1, 2)), Step((2, 3), (3, 4)), Step((4, 5))],
        [Step((0, 2), (2, 4)), Step((4, 5))],
        [Step((0, 4), (4, 5))],
    ],
    6: [
        [Step((0, 1), (1, 2)), Step((2, 3), (3, 4)), Step((4, 5), (5, 6))],
        [Step((0, 2), (2, 4)), Step((4, 6))],
        [Step((0, 4), (4, 6))],
    ],
    7: [
        [Step((0, 1), (1, 2)), Step((2, 3), (3, 4)), Step((4, 5), (5, 6)), Step((6, 7))],
        [Step((0, 2), (2, 4)), Step((4, 6), (6, 7))],
        [Step((0, 4), (4, 7))],
    ],
}


@pytest.mark.parametrize("n_layers", sorted(EXPECTED_PLANS))
def test_plan_rounds(n_layers):
    plan = plan_rounds(n_layers)
    assert plan == EXPECTED_PLANS[n_layers]
    # Invariants: N-1 merges in total, and the last round produces the full span.
    assert sum(not step.is_carry for steps in plan for step in steps) == n_layers - 1
    if plan:
        assert [step.result for step in plan[-1]] == [(0, n_layers)]


def test_plan_rejects_zero_layers():
    with pytest.raises(ValueError):
        plan_rounds(0)


# --- per-round parameters ------------------------------------------------------
def test_schedule_merges_overrides():
    schedule = SearchSchedule(SearchParams(degmin=1, degmax=45, nmin=-10, nmax=10), {2: {"nmin": -40, "nmax": 40}})
    assert schedule.params_for_round(1).nmax == 10
    round2 = schedule.params_for_round(2)
    assert (round2.nmin, round2.nmax, round2.degmax) == (-40, 40, 45)
    assert schedule.params_for_round(3).nmax == 10  # not overridden -> default


def test_schedule_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SearchSchedule(SearchParams(), {2: {"nmaxx": 40}})


def test_notebook_defaults():
    schedule = SearchSchedule.notebook_defaults()
    assert schedule.params_for_round(1).nmax == 10
    assert schedule.params_for_round(2).nmax == 40
    assert schedule.params_for_round(1).dtol == 5e-4 and schedule.params_for_round(1).degstep == 0.01


# --- driver ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def trilayer_run(tmp_path_factory):
    root = MoireStructure.from_poscar(BTO, "bto")
    run_dir = tmp_path_factory.mktemp("bto3")
    result = build_stack([root] * 3, SCHEDULE_3, Filters(max_residual=1e-3), run_dir=run_dir, log=None)
    return root, result


def test_dry_run_counts_without_searching(tmp_path):
    root = MoireStructure.from_poscar(BTO, "bto")
    result = build_stack([root] * 5, SCHEDULE_3, run_dir=tmp_path / "dry", dry_run=True, log=None)
    assert result.dry_run and result.final == []
    assert not (tmp_path / "dry").exists()  # nothing written
    searches = [report.n_searches for report in result.reports]
    # round 1: two 1x1 searches + a carry; round 2: unknown + carry; round 3: unknown
    assert searches == [1, 1, 0, None, 0, None]
    assert result.reports[0].cost_per_search == SCHEDULE_3.params_for_round(1).estimated_cost()
    assert "DRY RUN" in result.summary()


def test_max_candidates_is_an_unimplemented_hook():
    root = MoireStructure.from_poscar(BTO, "bto")
    with pytest.raises(NotImplementedError):
        build_stack([root] * 2, SCHEDULE_3, max_candidates=5, log=None)


def test_trilayer_tree(trilayer_run):
    root, result = trilayer_run
    (trilayer,) = result.final
    assert trilayer.n_layers == 3 and trilayer.level == 2
    assert trilayer.twist == pytest.approx(36.87, abs=1e-6)
    bilayer_id, upper_id = trilayer.parents
    assert upper_id == root.id  # layer 3 was carried forward unchanged
    assert trilayer.layer_rotations == pytest.approx([0.0, 22.62, 36.87])
    assert trilayer.twist_sequence == pytest.approx([22.62, 36.87])
    assert trilayer.id.startswith("bto_3_22.62_36.87_")
    tree = lineage(result.run_dir, trilayer.id)
    assert tree.parents[0].record.id == bilayer_id
    assert [r.id for r in tree.roots()] == [root.id] * 3


def test_run_directory_layout(trilayer_run):
    root, result = trilayer_run
    run_dir = result.run_dir
    spec = yaml.safe_load((run_dir / "stack_spec.yaml").read_text())
    assert [layer["poscar"] for layer in spec["layers"]] == [BTO] * 3
    assert spec["search"]["rounds"][2]["nmax"] == 40
    table = load_index(run_dir)
    assert sorted(table["round"]) == [0, 1, 2]
    assert sorted(table["n_layers"]) == [1, 2, 3]
    assert len(list((run_dir / "yamls").glob("*.yaml"))) == 3
    assert list(find(run_dir, n_layers=3)["id"]) == [result.final[0].id]


def test_rejected_candidates_are_saved_but_not_carried(tmp_path):
    root = MoireStructure.from_poscar(BTO, "bto")
    result = build_stack([root] * 3, SCHEDULE_3, Filters(max_residual=1e-5), run_dir=tmp_path, log=None)
    table = load_index(tmp_path)
    bilayer_rows = table[table["n_layers"] == 2]
    assert len(bilayer_rows) == 1 and not bilayer_rows["passed_filter"].iloc[0]
    assert result.final == []  # nothing survived round 1, so round 2 had 0 x 1 searches
    assert result.reports[-1].n_searches == 0


def test_four_layers_combine_two_bilayers(tmp_path):
    root = MoireStructure.from_poscar(BTO, "bto")
    schedule = SearchSchedule(
        SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10),
        {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -10, "nmax": 10}},
    )
    result = build_stack([root] * 4, schedule, Filters(max_residual=1e-3), run_dir=tmp_path, log=None)
    assert result.final
    four = result.final[0]
    # Both round-1 pairs give the identical bilayer recipe -> parents are the same id.
    assert four.parents[0] == four.parents[1]
    assert four.layer_rotations == pytest.approx([0.0, 22.62, four.twist, four.twist + 22.62])
    assert four.twist_sequence == pytest.approx([22.62, 22.62, four.twist])
