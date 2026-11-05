"""Tests for the run store, index table and query helpers."""

import numpy as np
import pytest

from MLM.index import INDEX_COLUMNS, RunStore, find, get, lineage, load_index, rebuild_index
from MLM.search import SearchParams, find_candidates
from MLM.structure import MoireStructure


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """Hand-assembled run: root, two bilayers (22.62°, 36.87°), one trilayer."""
    run_dir = tmp_path_factory.mktemp("run")
    root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    bilayers = []
    for low, high in [(22.0, 23.0), (36.5, 37.5)]:
        bilayers += find_candidates(root, root, SearchParams(degmin=low, degmax=high, nmin=-10, nmax=10))
    trilayers = find_candidates(bilayers[0], root, SearchParams(degmin=36.5, degmax=37.5, nmin=-40, nmax=40))
    store = RunStore(run_dir)
    store.add(root, 0, True)
    store.add(root, 0, True)  # adding twice is harmless
    store.add(bilayers[0], 1, True)
    store.add(bilayers[1], 1, False)
    for trilayer in trilayers:
        store.add(trilayer, 2, True)
    store.write_index()
    return run_dir, root, bilayers, trilayers


def test_index_has_one_row_per_record(small_run):
    run_dir, root, bilayers, trilayers = small_run
    table = load_index(run_dir)
    assert list(table.columns) == list(INDEX_COLUMNS)
    assert len(table) == 1 + len(bilayers) + len(trilayers)
    assert len(list((run_dir / "yamls").glob("*.yaml"))) == len(table)
    root_row = table[table["id"] == root.id].iloc[0]
    assert np.isnan(root_row["twist"]) and root_row["round"] == 0


def test_find_filters(small_run):
    run_dir, root, bilayers, trilayers = small_run
    assert set(find(run_dir, n_layers=2)["id"]) == {b.id for b in bilayers}
    assert list(find(run_dir, n_layers=2, passed_filter=True)["id"]) == [bilayers[0].id]
    assert set(find(run_dir, twist_range=(36.0, 37.0), n_layers=2)["id"]) == {bilayers[1].id}
    assert len(find(run_dir, material="bto")) == len(load_index(run_dir))
    assert find(run_dir, material="sto").empty
    assert len(find(run_dir, material=["bto", "bto", "bto"])) == len(trilayers)
    assert len(find(run_dir, angle_between_vectors=90.0)) == len(load_index(run_dir))
    tight = find(run_dir, max_cumulative_residual=1e-4)
    assert (tight["cumulative_residual"] <= 1e-4).all()


def test_get_and_lineage(small_run):
    run_dir, root, bilayers, trilayers = small_run
    trilayer = trilayers[0]
    assert get(run_dir, trilayer.id) == trilayer
    with pytest.raises(KeyError):
        get(run_dir, "does_not_exist")
    tree = lineage(run_dir, trilayer.id)
    assert tree.record == trilayer
    assert [p.record.id for p in tree.parents] == [bilayers[0].id, root.id]
    assert [r.id for r in tree.roots()] == [root.id] * 3
    assert len(tree.walk()) == 5  # trilayer, bilayer, root, root, root
    text = str(tree)
    assert trilayer.id in text and root.id in text


def test_rebuild_index_from_yamls(small_run, tmp_path):
    run_dir, *_ = small_run
    original = load_index(run_dir)
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(run_dir / "yamls", copy / "yamls")
    rebuild_index(copy)
    rebuilt = load_index(copy)
    assert set(rebuilt["id"]) == set(original["id"])
    assert dict(zip(rebuilt["id"], rebuilt["round"])) == dict(zip(original["id"], original["round"]))


def test_sqlite_index_gives_the_same_answers(small_run, tmp_path):
    """index_format='sqlite': same rows, same find() results (filtered in SQL), same get()."""
    import shutil

    from MLM.index import SQLITE_FILENAME, write_index

    run_dir, root, bilayers, trilayers = small_run
    copy = tmp_path / "sqlite_run"
    shutil.copytree(run_dir, copy)
    csv_rows = load_index(copy).to_dict(orient="records")
    (copy / "index.csv").unlink()
    write_index(copy, csv_rows, index_format="sqlite")
    assert (copy / SQLITE_FILENAME).exists()
    a, b = load_index(run_dir), load_index(copy)
    assert list(a.columns) == list(b.columns) and sorted(a["id"]) == sorted(b["id"])
    assert b["passed_filter"].dtype == bool
    queries = [dict(n_layers=2), dict(n_layers=2, passed_filter=True), dict(twist_range=(36.0, 37.0)),
               dict(material=["bto", "bto", "bto"]), dict(angle_between_vectors=90.0),
               dict(max_cumulative_residual=1e-4), dict(max_strain=0.0)]
    for query in queries:
        assert list(find(run_dir, **query)["id"]) == list(find(copy, **query)["id"]), query
    assert get(copy, trilayers[0].id) == trilayers[0]
    with pytest.raises(KeyError):
        get(copy, "nope")


def test_build_stack_with_sqlite_index(tmp_path):
    from MLM.search import Filters, SearchParams, SearchSchedule, build_stack

    root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    schedule = SearchSchedule(SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10))
    build_stack([root] * 2, schedule, Filters(max_residual=1e-3), run_dir=tmp_path, log=None, index_format="sqlite")
    assert (tmp_path / "index.sqlite").exists() and not (tmp_path / "index.csv").exists()
    assert len(find(tmp_path, n_layers=2)) == 1
    with pytest.raises(ValueError):
        build_stack([root] * 2, schedule, run_dir=tmp_path / "x", log=None, index_format="parquet")
