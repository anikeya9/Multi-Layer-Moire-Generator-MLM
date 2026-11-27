"""Reading structure files (one implementation for the whole package).

MLM used to have three copies of a hand-written ``read_vasp`` based on
``np.loadtxt``. They ignored the POSCAR scale factor and failed on Direct
coordinates and on Selective Dynamics. All readers now go through ASE, which
handles every POSCAR variant and also CIF files.

- :func:`read_structure` → an ``ase.Atoms`` object (POSCAR/CONTCAR/``*.vasp``, CIF, …)
- :func:`read_vasp` → the legacy tuple ``(lattice dict a/b/c, atom types, Cartesian coords)``
  used by the original notebooks; same values as before for the files in the repository.

This module deliberately does not import numba, so reading a file is fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from ase import Atoms
from ase.io import read

PathLike = Union[str, Path]

#: File-name suffixes read as VASP POSCAR files.
VASP_SUFFIXES = {".vasp", ".poscar", ".contcar"}


def guess_format(path: PathLike) -> str | None:
    """ASE format name for ``path``, or ``None`` to let ASE guess.

    >>> guess_format("moire_structures/bto/bto_1-layer.vasp"), guess_format("POSCAR"), guess_format("x.cif")
    ('vasp', 'vasp', 'cif')
    """
    path = Path(path)
    name = path.name.upper()
    if path.suffix.lower() in VASP_SUFFIXES or name.startswith(("POSCAR", "CONTCAR")):
        return "vasp"
    if path.suffix.lower() == ".cif":
        return "cif"
    return None


def read_structure(path: PathLike) -> Atoms:
    """Read one structure with ASE (POSCAR in any variant, CIF, or any ASE format).

    >>> atoms = read_structure("moire_structures/bto/bto_1-layer.vasp")
    >>> atoms.get_chemical_formula(), len(atoms)
    ('Ba2O4Ti', 7)
    """
    atoms = read(str(path), format=guess_format(path))
    if isinstance(atoms, list):  # multi-frame files: take the first frame
        atoms = atoms[0]
    return atoms


def read_vasp(file_path: PathLike) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Legacy interface: ``(lattice_vectors, atom_type_arr, coords)``.

    - ``lattice_vectors``: dict with keys ``'a'``, ``'b'``, ``'c'`` (3-vectors, Å; the
      scale factor is applied),
    - ``atom_type_arr``: one chemical symbol per atom, in file order,
    - ``coords``: Cartesian positions (N x 3, Å), also for files written in Direct coordinates.

    >>> lattice, types, coords = read_vasp("moire_structures/bto/bto_1-layer.vasp")
    >>> lattice["a"].tolist(), types[:3].tolist(), coords.shape
    ([3.9903790951, 0.0, 0.0], ['Ba', 'Ba', 'Ti'], (7, 3))
    """
    atoms = read_structure(file_path)
    cell = np.array(atoms.cell, dtype=float)
    lattice_vectors = {"a": cell[0], "b": cell[1], "c": cell[2]}
    return lattice_vectors, np.array(atoms.get_chemical_symbols()), np.array(atoms.positions, dtype=float)
