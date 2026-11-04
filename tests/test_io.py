"""Tests for MLM.io: one ASE-based reader for every POSCAR variant and CIF."""

import numpy as np
import pytest
from ase.io import write

from MLM.io import read_structure, read_vasp
from MLM.structure import MoireStructure, read_primitive

BTO = "moire_structures/bto/bto_1-layer.vasp"


def legacy_read_vasp(file_path):
    """The original np.loadtxt reader (Cartesian, scale factor 1 only)."""
    lattice = {k: np.loadtxt(file_path, skiprows=s, max_rows=1) for k, s in zip("abc", (2, 3, 4))}
    names = np.loadtxt(file_path, skiprows=5, max_rows=1, dtype="str")
    counts = np.loadtxt(file_path, skiprows=6, max_rows=1, dtype="int")
    return lattice, np.repeat(names, counts), np.loadtxt(file_path, skiprows=8)


@pytest.mark.parametrize("path", [BTO, "moire_structures/graphene/graphene_monolayer.vasp",
                                  "moire_structures/bto/moire_strs_vasp/bto_22.62.vasp"])
def test_same_result_as_the_legacy_reader(path):
    old, new = legacy_read_vasp(path), read_vasp(path)
    for key in "abc":
        np.testing.assert_allclose(new[0][key], old[0][key])
    assert list(new[1]) == list(old[1])
    np.testing.assert_allclose(new[2], np.atleast_2d(old[2])[:, :3])


def test_legacy_names_still_work():
    import MLM.moire_lattice_vector_finder as mlf
    import MLM.structure_writer as sw
    from MLM.tool import angle_in_degrees, load_layer_basis

    assert mlf.read_vasp is read_vasp and sw.read_vasp is read_vasp
    np.testing.assert_allclose(load_layer_basis(BTO), np.eye(2) * 3.9903790951)
    assert angle_in_degrees([1, 0], [0, 2]) == pytest.approx(90.0)


def _variants(tmp_path):
    atoms = read_structure(BTO)
    direct = tmp_path / "direct.vasp"
    write(direct, atoms, format="vasp", direct=True)
    selective = tmp_path / "POSCAR_selective"
    from ase.constraints import FixAtoms

    fixed = atoms.copy()
    fixed.set_constraint(FixAtoms(indices=[0, 1]))
    write(selective, fixed, format="vasp")
    scaled = tmp_path / "scaled.vasp"
    lines = open(BTO).read().splitlines()
    lines[1] = "2.0"  # scale factor 2: halve the lattice and positions so the result is unchanged
    for row in (2, 3, 4):
        lines[row] = " ".join(str(float(x) / 2) for x in lines[row].split())
    for row in range(8, len(lines)):
        lines[row] = " ".join(str(float(x) / 2) for x in lines[row].split())
    scaled.write_text("\n".join(lines) + "\n")
    cif = tmp_path / "bto.cif"
    write(cif, atoms, format="cif")
    return {"direct": direct, "selective dynamics": selective, "scale factor": scaled, "cif": cif}


def test_variants_read_the_same_layer(tmp_path):
    reference = read_primitive(BTO)
    for name, path in _variants(tmp_path).items():
        primitive = read_primitive(path)
        np.testing.assert_allclose(primitive["in_plane"], reference["in_plane"], atol=1e-6, err_msg=name)
        assert list(primitive["atom_types"]) == list(reference["atom_types"]), name
        np.testing.assert_allclose(primitive["coords"], reference["coords"], atol=1e-6, err_msg=name)
    # And a root built from a CIF has the same lattice as the POSCAR root.
    root = MoireStructure.from_poscar(_variants(tmp_path)["cif"], "bto")
    np.testing.assert_allclose(root.lattice_vectors, MoireStructure.from_poscar(BTO, "bto").lattice_vectors, atol=1e-6)


def test_out_of_plane_cell_is_rejected(tmp_path):
    atoms = read_structure(BTO)
    atoms.cell[2] = [1.0, 0.0, 8.0]  # c not along z
    path = tmp_path / "tilted.vasp"
    write(path, atoms, format="vasp")
    with pytest.raises(ValueError):
        read_primitive(path)
