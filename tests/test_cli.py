"""Smoke tests for ``python -m MLM`` (stretch goal S3)."""

import shutil

import yaml

from MLM.cli import main
from MLM.index import find

SPEC = {
    "name": "cli",
    "output_dir": None,  # filled in per test
    "layers": [{"poscar": "moire_structures/bto/bto_1-layer.vasp", "material": "bto"}] * 2,
    "search": {"default": {"degmin": 22.0, "degmax": 23.0, "nmin": -10, "nmax": 10, "goodangles": [90.0]}},
    "filters": {"max_residual": 1e-3},
}


def test_cli_run_find_show_write(tmp_path, capsys):
    run_dir = tmp_path / "run"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(dict(SPEC, output_dir=str(run_dir))))
    assert main(["run", str(spec_path), "--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert main(["run", str(spec_path), "--quiet"]) == 0
    assert main(["find", str(run_dir), "--n-layers", "2", "--passed", "yes"]) == 0
    assert "1 matching structure" in capsys.readouterr().out
    bilayer_id = find(run_dir, n_layers=2)["id"].iloc[0]
    assert main(["show", str(run_dir), bilayer_id]) == 0
    out = capsys.readouterr().out
    assert f"id: {bilayer_id}" in out and "lineage:" in out
    # write needs the POSCAR paths: run_dir is in tmp, so copy it next to the repo's files.
    from conftest import REPO_ROOT

    local = REPO_ROOT / "runs" / "_cli_test"
    shutil.rmtree(local, ignore_errors=True)
    shutil.copytree(run_dir, local)
    try:
        assert main(["write", str(local), bilayer_id, "--fmt", "lammps"]) == 0
        assert (local / "structures" / f"{bilayer_id}.data").exists()
    finally:
        shutil.rmtree(local)
