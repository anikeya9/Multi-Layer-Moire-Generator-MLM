"""Tests for the stack-spec loader (milestone M4)."""

import pytest
import yaml

from MLM.index import load_index
from MLM.spec import load_stack_spec, parse_stack_spec, run_stack_spec

SPEC = {
    "name": "t",
    "output_dir": "runs/t",
    "layers": [{"poscar": "moire_structures/bto/bto_1-layer.vasp", "material": "bto"}] * 3,
    "search": {
        "default": {"degmin": 22.0, "degmax": 23.0, "degstep": 0.01, "dtol": 5e-4, "nmin": -10, "nmax": 10,
                    "goodangles": [90.0]},
        "rounds": {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -40, "nmax": 40}},
    },
    "filters": {"max_residual": 1e-3, "max_cumulative_residual": None},
}


def test_parse_and_merge_rounds():
    spec = parse_stack_spec(SPEC)
    assert spec.name == "t" and len(spec.layers) == 3
    round1, round2 = spec.schedule.params_for_round(1), spec.schedule.params_for_round(2)
    assert (round1.nmin, round1.nmax, round1.degmin) == (-10, 10, 22.0)
    assert (round2.nmin, round2.nmax, round2.degmin, round2.dtol) == (-40, 40, 36.5, 5e-4)
    assert round2.goodangles == (90.0,)
    assert spec.filters.max_residual == 1e-3 and spec.filters.max_cumulative_residual is None


def test_defaults_when_sections_are_missing():
    spec = parse_stack_spec({"name": "x", "layers": SPEC["layers"][:1]})
    assert spec.output_dir == "runs/x"
    assert spec.filters.max_residual is None
    assert spec.schedule.params_for_round(1).degstep == 0.01


def test_round_keys_may_be_strings():
    data = dict(SPEC, search={"default": {}, "rounds": {"2": {"nmax": 40}}})
    assert parse_stack_spec(data).schedule.params_for_round(2).nmax == 40


@pytest.mark.parametrize(
    "broken",
    [
        {"layers": SPEC["layers"]},  # no name
        dict(SPEC, colour="red"),  # unknown top-level key
        dict(SPEC, layers=[{"poscar": "x.vasp"}]),  # no material
        dict(SPEC, search={"default": {"nmaxx": 3}}),  # typo in a parameter
        dict(SPEC, search={"rounds": {2: {"degstepp": 1}}}),
        dict(SPEC, filters={"max_resid": 1}),
        dict(SPEC, layers=[]),
    ],
)
def test_invalid_specs_raise(broken):
    with pytest.raises(ValueError):
        parse_stack_spec(broken)


def test_example_spec_parses():
    spec = load_stack_spec("examples/bto_trilayer_demo.yaml")
    assert spec.schedule.params_for_round(2).nmax == 40


def test_run_stack_spec_one_call(tmp_path):
    spec_path = tmp_path / "job.yaml"
    data = dict(SPEC, output_dir="out")
    spec_path.write_text("# my comment\n" + yaml.safe_dump(data, sort_keys=False))
    # Relative paths are resolved against base_dir: POSCARs from the repo root, output too.
    result = run_stack_spec(spec_path, base_dir=".", log=None)
    run_dir = result.run_dir
    try:
        assert (run_dir / "stack_spec.yaml").read_text().startswith("# my comment")  # copied verbatim
        table = load_index(run_dir)
        assert sorted(table["n_layers"]) == [1, 2, 3]
        assert result.final[0].source_poscar == ["moire_structures/bto/bto_1-layer.vasp"] * 3
    finally:
        import shutil

        shutil.rmtree(run_dir)


def test_run_stack_spec_dry_run(tmp_path):
    spec_path = tmp_path / "job.yaml"
    spec_path.write_text(yaml.safe_dump(dict(SPEC, output_dir=str(tmp_path / "out"))))
    result = run_stack_spec(spec_path, dry_run=True, log=None)
    assert result.dry_run and not (tmp_path / "out").exists()


def test_interfaces_section():
    data = dict(SPEC, search={"default": {"nmin": -10, "nmax": 10},
                              "interfaces": {1: {"degmin": 22.0, "degmax": 23.0}, "2": {"degmin": 36.0}}})
    schedule = parse_stack_spec(data).schedule
    assert schedule.params_for_round(1, interface=1).degmax == 23.0
    assert schedule.params_for_round(2, interface=2).degmin == 36.0
    with pytest.raises(ValueError):
        parse_stack_spec(dict(SPEC, search={"interfaces": {1: {"degmn": 1}}}))
