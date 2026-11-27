"""Parallel searches, resume, stop, progress callbacks and run_info.yaml."""

import pytest
import yaml

from MLM.index import load_index
from MLM.runinfo import load_run_info
from MLM.search import Filters, SearchParams, SearchSchedule, build_stack
from MLM.structure import MoireStructure

BTO = "moire_structures/bto/bto_1-layer.vasp"
# Round 1 gives 2 bilayers (22.62°, 36.87°); round 2 then runs 2 searches.
SCHEDULE = SearchSchedule(
    SearchParams(degmin=22.0, degmax=37.5, degstep=0.01, nmin=-10, nmax=10),
    interfaces={1: {"degmin": 22.0, "degmax": 23.0}, 2: {"degmin": 36.5, "degmax": 37.5}},
)
# make round 1 find two cells: a window with two commensurate twists
SCHEDULE_TWO = SearchSchedule(
    SearchParams(nmin=-10, nmax=10),
    {2: {"nmin": -20, "nmax": 20}},
    interfaces={1: {"degmin": 22.0, "degmax": 37.0}, 2: {"degmin": 36.5, "degmax": 37.5}},
)


@pytest.fixture(scope="module")
def root():
    return MoireStructure.from_poscar(BTO, "bto")


def ids(run_dir):
    return sorted(load_index(run_dir)["id"])


def test_parallel_gives_identical_results(root, tmp_path):
    serial = build_stack([root] * 3, SCHEDULE_TWO, Filters(max_residual=1e-3), run_dir=tmp_path / "s", log=None)
    parallel = build_stack([root] * 3, SCHEDULE_TWO, Filters(max_residual=1e-3), run_dir=tmp_path / "p",
                           log=None, workers=2)
    assert serial.reports[-1].n_searches >= 2  # so the pool really ran several searches
    assert [r.id for r in serial.final] == [r.id for r in parallel.final]
    assert ids(tmp_path / "s") == ids(tmp_path / "p")
    info = load_run_info(tmp_path / "p")
    assert info["status"] == "finished" and info["workers"] == 2
    assert info["mlm_version"] and len(info["steps"]) == 2 and info["final_candidates"] == len(parallel.final)


def test_stop_and_resume(root, tmp_path):
    calls = {"n": 0}

    def stop_after_round_one():
        calls["n"] += 1
        return calls["n"] > 1  # allow the first search (round 1), stop before round 2

    stopped = build_stack([root] * 3, SCHEDULE_TWO, Filters(max_residual=1e-3), run_dir=tmp_path,
                          log=None, should_stop=stop_after_round_one)
    assert stopped.status == "stopped" and stopped.final == []
    assert load_run_info(tmp_path)["status"] == "stopped"
    assert sorted(load_index(tmp_path)["n_layers"].unique()) == [1, 2]

    events = []
    resumed = build_stack([root] * 3, SCHEDULE_TWO, Filters(max_residual=1e-3), run_dir=tmp_path,
                          log=None, resume=True, progress_callback=events.append)
    assert resumed.status == "finished" and resumed.final
    assert events and all(e["round"] == 2 for e in events)  # round 1 was not searched again
    assert events[-1]["done"] == events[-1]["total"]
    fresh = build_stack([root] * 3, SCHEDULE_TWO, Filters(max_residual=1e-3), run_dir=tmp_path / "fresh", log=None)
    assert [r.id for r in resumed.final] == [r.id for r in fresh.final]
    assert ids(tmp_path) == ids(tmp_path / "fresh")


def test_resume_refuses_a_different_job(root, tmp_path):
    build_stack([root] * 2, SCHEDULE, Filters(max_residual=1e-3), run_dir=tmp_path, log=None)
    with pytest.raises(ValueError):
        build_stack([root] * 2, SCHEDULE, Filters(max_residual=1e-4), run_dir=tmp_path, log=None, resume=True)
    with pytest.raises(FileNotFoundError):
        build_stack([root] * 2, SCHEDULE, run_dir=tmp_path / "empty", log=None, resume=True)


def test_progress_file_lists_finished_steps(root, tmp_path):
    build_stack([root] * 2, SCHEDULE, Filters(max_residual=1e-3), run_dir=tmp_path, log=None)
    progress = yaml.safe_load((tmp_path / "progress.yaml").read_text())
    assert [entry["step"] for entry in progress["completed"]] == ["L1 + L2 -> L1-2"]
    assert progress["completed"][0]["passed"]
