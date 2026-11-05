"""Tests for the MoireStructure record and its YAML round trip (milestone M1)."""

import numpy as np
import pytest
import yaml

from MLM.structure import SCHEMA_KEYS, MoireStructure, _to_builtin, material_label
from conftest import BTO_POSCAR, PTO_POSCAR


def make_bilayer(root: MoireStructure, twist: float = 22.62, residual: float = 3.4e-5) -> MoireStructure:
    """A hand-made level-1 record (no search needed)."""
    return MoireStructure(
        level=1,
        n_layers=2,
        material=root.material * 2,
        parents=[root.id, root.id],
        twist=np.float64(twist),  # deliberately a NumPy scalar
        lattice_vectors=np.array([[11.97, -7.98], [7.98, 11.97]]),
        residual=np.float64(residual),
        cumulative_residual=residual,
        source_poscar=root.source_poscar * 2,
        twist_sequence=[twist],
        layer_rotations=[0.0, twist],
        layer_residuals=[0.0, residual],
        coefficients={"lower": [[3, -2], [2, 3]], "upper": [[2, -3], [3, 2]]},
    )


def test_from_poscar_reads_in_plane_cell(bto_root):
    assert bto_root.level == 0 and bto_root.n_layers == 1
    assert bto_root.material == ["bto"]
    assert bto_root.source_poscar == [BTO_POSCAR]
    np.testing.assert_allclose(bto_root.lattice_vectors, [[3.9903790951, 0.0], [0.0, 3.9903790951]])
    assert bto_root.angle_between_vectors == pytest.approx(90.0)


def test_level0_nulls_are_null_not_zero(bto_root, tmp_path):
    data = yaml.safe_load(bto_root.to_yaml(tmp_path).read_text())
    assert data["parents"] == []
    for key in ("twist", "residual", "cumulative_residual", "layer_residuals", "coefficients"):
        assert data[key] is None, key


def test_yaml_round_trip_level0(bto_root, tmp_path):
    path = bto_root.to_yaml(tmp_path)
    assert path.name == f"{bto_root.id}.yaml"
    again = MoireStructure.from_yaml(path)
    assert again == bto_root
    assert again.id == bto_root.id


def test_yaml_round_trip_with_numpy_values(bto_root, tmp_path):
    bilayer = make_bilayer(bto_root)
    path = bilayer.to_yaml(tmp_path)  # would raise if NumPy types leaked into safe_dump
    again = MoireStructure.from_yaml(path)
    assert again == bilayer
    assert isinstance(again.twist, float) and not isinstance(again.twist, np.floating)


def test_schema_is_uniform_across_levels(bto_root, tmp_path):
    root_keys = list(yaml.safe_load(bto_root.to_yaml(tmp_path).read_text()))
    bilayer_keys = list(yaml.safe_load(make_bilayer(bto_root).to_yaml(tmp_path).read_text()))
    assert root_keys == bilayer_keys == list(SCHEMA_KEYS)


def test_id_format_and_uniqueness(bto_root):
    assert bto_root.id.startswith("bto_1_")
    a = make_bilayer(bto_root, twist=22.620000)
    b = make_bilayer(bto_root, twist=22.621)  # same rounded angle, different content
    assert a.id.startswith("bto_2_22.62_") and b.id.startswith("bto_2_22.62_")
    assert a.id != b.id
    # Same content -> same id (deterministic).
    assert make_bilayer(bto_root).id == make_bilayer(bto_root).id


def test_heterostructure_label():
    assert material_label(["pto", "sto"]) == "pto_sto"
    assert material_label(["bto", "bto"]) == "bto"
    root = MoireStructure.from_poscar(PTO_POSCAR, "pto")
    assert root.id.startswith("pto_1_")


def test_stored_id_is_authoritative(bto_root, tmp_path):
    """The id inside the file wins, even if it no longer matches the content hash."""
    data = bto_root.to_dict()
    data["id"] = "renamed_by_hand"
    assert MoireStructure.from_dict(data).id == "renamed_by_hand"


def test_validation_rejects_inconsistent_records(bto_root):
    data = bto_root.to_dict()
    data["level"] = 1
    with pytest.raises(ValueError):
        MoireStructure.from_dict(data)
    data = bto_root.to_dict()
    data["twist"] = 0.0  # level 0 must have twist = null
    with pytest.raises(ValueError):
        MoireStructure.from_dict(data)
    data = bto_root.to_dict()
    data["unexpected"] = 1
    with pytest.raises(ValueError):
        MoireStructure.from_dict(data)


def test_to_builtin_converts_nested_numpy():
    out = _to_builtin({"x": [np.int64(1), np.array([1.5, 2.5])], "y": np.bool_(True)})
    assert out == {"x": [1, [1.5, 2.5]], "y": True}
    assert type(out["x"][0]) is int


def test_schema_version_written_and_old_files_migrate(bto_root, tmp_path):
    from MLM.structure import SCHEMA_VERSION, migrate_record

    data = yaml.safe_load(bto_root.to_yaml(tmp_path).read_text())
    assert data["schema_version"] == SCHEMA_VERSION == 3
    old = {key: value for key, value in data.items() if key not in ("schema_version", "layer_strains", "strain_energy")}
    migrated = migrate_record(old)
    assert migrated["schema_version"] == 3 and migrated["layer_strains"] == [[[0.0, 0.0], [0.0, 0.0]]]
    assert migrated["strain_energy"] is None
    version_2 = dict(data, schema_version=2)
    del version_2["strain_energy"]
    assert migrate_record(version_2)["strain_energy"] is None
    assert MoireStructure.from_dict(old) == bto_root  # a version-1 file loads unchanged
    assert MoireStructure.from_dict(old).id == bto_root.id
    with pytest.raises(ValueError):
        migrate_record(dict(data, schema_version=SCHEMA_VERSION + 1))
