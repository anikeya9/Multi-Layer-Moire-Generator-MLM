"""The GUI: helpers and a full click-through (plan, run, results, viewer, export) without a browser."""

import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("py3Dmol")

from conftest import REPO_ROOT  # noqa: E402

from MLM.gui import (  # noqa: E402
    MLMApp,
    build_job,
    download_link,
    list_layer_files,
    parse_shift,
    plan_table,
    results_table,
)


def test_helpers():
    assert parse_shift("1/3, 2/3") == pytest.approx((1 / 3, 2 / 3))
    assert parse_shift("  ") is None
    with pytest.raises(ValueError):
        parse_shift("1 2 3")
    files = list_layer_files(REPO_ROOT)
    assert "moire_structures/bto/bto_1-layer.vasp" in files
    assert not any("moire_strs_vasp" in f for f in files)  # finished moiré cells are not layers


def test_build_job_and_plan():
    layers = [{"poscar": "moire_structures/bto/bto_1-layer.vasp", "label": "bto"}] * 4
    job = build_job(layers, [(22, 23), (30, 31), (36, 38)], [90.0], 1e-3, None, root=REPO_ROOT)
    table = plan_table(job)
    assert list(table["interface"][table["interface"] != "—"]) == ["layer 1 → 2", "layer 3 → 4", "layer 2 → 3"]
    assert list(table["twist range (deg)"][table["interface"] == "layer 3 → 4"]) == ["36 – 38"]
    with pytest.raises(ValueError):
        build_job(layers, [(22, 23)], [90.0], 1e-3, None, root=REPO_ROOT)  # wrong number of ranges
    with pytest.raises(ValueError):
        build_job(layers[:2], [(23, 22)], [90.0], 1e-3, None, root=REPO_ROOT)  # min > max
    strained = build_job([{"poscar": "moire_structures/PTO_STO/pto_tetra_monolayer.vasp", "label": "pto"},
                          {"poscar": "moire_structures/PTO_STO/STO_centered_unit_cell_cubic_monolayer.vasp",
                           "label": "sto", "strain": 1.0, "young": 150.0, "poisson": 0.3}],
                         [(5, 10)], [90.0], 1e-6, None, root=REPO_ROOT)
    assert strained["strain_limits"] == [0.0, 1.0] and strained["elastic"][0] is None
    assert plan_table(strained)["strain limits (%)"].iloc[0] == "0 / 1"


def test_download_link(tmp_path):
    path = tmp_path / "x.vasp"
    path.write_text("hello")
    assert 'download="x.vasp"' in download_link(path) and "base64," in download_link(path)


def test_click_through(tmp_path):
    app = MLMApp(root=REPO_ROOT, runs_dir=tmp_path)
    # Stack: the default example (BTO trilayer); edit the layer list with the buttons
    assert len(app.layer_rows) == 3 and len(app.twist_sliders) == 2
    app.layers_box.children[-1].children[-1].click()  # trash button of layer 3
    assert len(app.layer_rows) == 2 and len(app.twist_sliders) == 1
    app.set_layers([{"poscar": "moire_structures/bto/bto_1-layer.vasp", "label": "bto"}] * 3,
                   [(22.0, 23.0), (36.0, 38.0)])
    app.run_name.value = "click"
    # Plan
    table = app.plan()
    assert table is not None and "<table" in app.plan_out.value
    # Run (synchronously here; the button runs it in a thread)
    app.run(background=False)
    assert app.run_select.value == "click"
    assert app.progress.value == app.progress.max  # progress callback reached the end
    # Results
    results = app.show_results()
    assert len(results) >= 1 and len(app.candidates.options) == len(results)
    assert results_table(tmp_path / "click").equals(results)
    # Viewer
    record = app.select(app.candidates.options[0][1])
    assert record.n_layers == 3 and app.tabs.selected_index == 4
    markup = app.show_structure()
    assert "3Dmol" in markup and "Checks" in app.viewer_info.value and "passed" in app.viewer_info.value
    app.repeat.value, app.colour_by.value, app.view_direction.value = 2, "element", "side"
    app.visible_layers.value = (1, 3)
    assert "3Dmol" in app.show_structure()
    # Export with a lateral shift of layer 2
    app.formats.value = ("vasp", "lammps")
    app.shifts_text.value = "; 0.5 0.5 ;"
    paths = app.export()
    assert len(paths) == 2 and all(path.exists() for path in paths)
    assert "download=" in app.export_out.value and "stack_spec.yaml" in app.export_out.value


def test_bad_input_is_reported_not_raised(tmp_path):
    app = MLMApp(root=REPO_ROOT, runs_dir=tmp_path)
    app.twist_sliders[0].value = (22.0, 22.0)  # min == max
    assert app.plan() is None and "⚠" in app.plan_out.value
    app.run(background=False)  # logs the problem instead of raising
    assert app.export() == [] and "select a structure" in app.export_out.value


def test_stop_button_stops_a_background_run(tmp_path):
    """The Stop button sets an event that build_stack polls before every search."""
    app = MLMApp(root=REPO_ROOT, runs_dir=tmp_path)
    app.run_name.value = "stopper"
    calls = {"n": 0}

    def stop_after_first():  # stands in for a click on Stop after the first search
        calls["n"] += 1
        return calls["n"] > 1

    app._stop.is_set = stop_after_first
    app.run(background=False)
    from MLM.runinfo import load_run_info

    assert load_run_info(tmp_path / "stopper")["status"] == "stopped"
