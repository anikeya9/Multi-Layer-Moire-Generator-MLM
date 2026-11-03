"""Rebuild atoms from a structure recipe and export them (VASP / LAMMPS).

A recipe holds no atoms. To build them we

1. walk the lineage back to the roots to get each layer's primitive POSCAR and its
   absolute rotation in the structure's frame,
2. replicate each primitive layer over a patch large enough to cover the moiré cell
   (``structure_writer.replicate_atoms``),
3. stack the layers in z and rotate each one about the z axis,
4. fold everything into one moiré cell and drop periodic images
   (``structure_writer.select_atoms_fractional``),
5. write the result with ASE (``structure_writer.write_vasp_ase`` /
   ``write_lammps_ase``).

Steps 2–5 follow ``notebooks/trilayer_structure_creator.ipynb``; the stacking rule
(next layer starts ``interlayer_gap`` Å above the highest atom of the previous
layer) and the 50 Å vacuum are the notebook's.

Example
-------
>>> write_structure("runs/bto_trilayer_demo", "bto_3_22.62_36.87_c26a56", fmt="vasp")   # doctest: +SKIP
PosixPath('runs/bto_trilayer_demo/structures/bto_3_22.62_36.87_c26a56.vasp')
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import MLM.structure_writer as sw
from MLM.index import get, lineage
from MLM.structure import MoireStructure, PathLike, read_primitive

#: File extension written for each supported format.
FORMATS = {"vasp": ".vasp", "lammps": ".data"}


@dataclass
class BuiltStructure:
    """Atoms of one moiré cell.

    ``atoms`` has columns ``x, y, z, type, layer`` (``layer`` = 0-based index in
    stacking order); ``cell`` is the 3x3 cell (rows A1, A2, A3).
    ``expected_counts`` is the number of atoms each layer must contribute
    (cell area / primitive area × atoms per primitive cell).
    """

    record: MoireStructure
    atoms: pd.DataFrame
    cell: np.ndarray
    expected_counts: list[int]

    @property
    def layer_counts(self) -> list[int]:
        """Atoms actually kept per layer."""
        counts = self.atoms["layer"].value_counts()
        return [int(counts.get(layer, 0)) for layer in range(self.record.n_layers)]

    @property
    def composition(self) -> dict[str, int]:
        """Number of atoms per element, e.g. ``{'Ba': 390, 'O': 780, 'Ti': 195}``."""
        return {str(k): int(v) for k, v in self.atoms["type"].value_counts().sort_index().items()}

    @property
    def counts_ok(self) -> bool:
        """True if every layer contributed exactly the expected number of atoms."""
        return self.layer_counts == self.expected_counts


# ---------------------------------------------------------------------------
# Lineage → per-layer rotations
# ---------------------------------------------------------------------------
def rotations_from_lineage(run_dir: PathLike, structure_id: str) -> list[float]:
    """Absolute rotation (deg) of every layer, derived by walking the lineage.

    Block-rigid rule: a root has ``[0]``; a child of (lower, upper) at ``twist`` has
    ``rotations(lower) + [r + twist for r in rotations(upper)]``.

    >>> rotations_from_lineage("runs/bto_trilayer_demo", trilayer_id)   # doctest: +SKIP
    [0.0, 22.62, 36.87]
    """
    def rotations(node) -> list[float]:
        if not node.parents:
            return [0.0]
        lower, upper = node.parents
        return rotations(lower) + [angle + node.record.twist for angle in rotations(upper)]

    return rotations(lineage(run_dir, structure_id))


# ---------------------------------------------------------------------------
# Building atoms
# ---------------------------------------------------------------------------
def replicas_needed(primitive: np.ndarray, radius: float) -> int:
    """How many primitive cells (each way) cover a disc of ``radius`` around the origin.

    ``replicate_atoms(..., na, nb)`` covers cells ``-na … na-1``; the patch contains a
    disc of radius ``na * h`` where ``h`` is the smallest height of the primitive cell.
    One extra cell absorbs the basis atoms' offset inside the cell. (The notebooks
    used ``8 * (|A1| + |A2|) / |a|``, which is many times larger than needed;
    folding gives the same atoms either way.)
    """
    area = abs(np.linalg.det(primitive))
    height = area / max(np.linalg.norm(primitive[0]), np.linalg.norm(primitive[1]))
    return int(math.ceil(radius / height)) + 1


def _rotate_xy(frame: pd.DataFrame, angle_deg: float) -> pd.DataFrame:
    """Rotate atom positions anticlockwise about the z axis (z unchanged)."""
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    rotated = frame.copy()
    rotated["x"] = cos * x - sin * y
    rotated["y"] = sin * x + cos * y
    return rotated


def _replicated_layer(poscar: str, radius: float, search_dirs: Sequence[PathLike]) -> tuple[pd.DataFrame, int]:
    """Atoms of one primitive layer replicated over a disc of ``radius``; also atoms/cell."""
    primitive = read_primitive(poscar, search_dirs)
    cell, types, coords = primitive["cell"], primitive["atom_types"], primitive["coords"]
    n = replicas_needed(primitive["in_plane"], radius)
    _, _, positions, replicated_types = sw.replicate_atoms(
        a=cell[0], b=cell[1], c=cell[2],
        atom_data=coords, atom_type_arr=types, natoms=len(coords),
        na=n, nb=n, nc=1,
    )
    frame = pd.DataFrame(positions, columns=["x", "y", "z"])
    frame["type"] = replicated_types
    return frame, len(coords)


def build_atoms(
    record: MoireStructure,
    rotations: Optional[Sequence[float]] = None,
    interlayer_gap: float = 3.0,
    vacuum: float = 50.0,
    fold_tolerance: Optional[float] = None,
    search_dirs: Sequence[PathLike] = (),
) -> BuiltStructure:
    """Atoms of ``record``'s moiré cell.

    - ``rotations``: absolute rotation of each layer; defaults to the stored
      ``record.layer_rotations`` (``write_structure`` passes the lineage-derived ones).
    - ``interlayer_gap``: Å between the highest atom of a layer and the lowest
      *possible* position of the next (the notebook rule: ``z += gap + max(z_below)``).
    - ``vacuum``: c = (top − bottom) + ``vacuum`` (notebook: 50 Å).
    - ``fold_tolerance``: ``dtol`` for ``select_atoms_fractional``; defaults to the
      cell's worst per-layer residual (the notebooks passed the step's ``delvec``).

    >>> built = build_atoms(get("runs/demo", some_id))   # doctest: +SKIP
    >>> built.counts_ok, built.composition                # doctest: +SKIP
    (True, {'Ba': 390, 'O': 780, 'Ti': 195})
    """
    rotations = list(record.layer_rotations if rotations is None else rotations)
    a1, a2 = record.lattice_vectors
    radius = float(np.linalg.norm(a1) + np.linalg.norm(a2))  # farthest cell corner from the origin

    layers = []
    expected_counts = []
    top_of_previous: Optional[float] = None
    for position, (poscar, angle) in enumerate(zip(record.source_poscar, rotations)):
        frame, atoms_per_cell = _replicated_layer(poscar, radius, search_dirs)
        if top_of_previous is not None:
            frame["z"] = frame["z"] + interlayer_gap + top_of_previous
        top_of_previous = float(frame["z"].max())
        frame = _rotate_xy(frame, angle)
        frame["layer"] = position
        layers.append(frame)
        primitive_area = abs(np.linalg.det(read_primitive(poscar, search_dirs)["in_plane"]))
        expected_counts.append(int(round(record.cell_area / primitive_area)) * atoms_per_cell)

    stacked = pd.concat(layers, ignore_index=True)
    if fold_tolerance is None:
        fold_tolerance = record.worst_layer_residual or 1e-6
    selected = sw.select_atoms_fractional(stacked, a1, a2, dtol=fold_tolerance)

    height = float(stacked["z"].max() - layers[0]["z"].min()) + vacuum
    cell = np.array([[a1[0], a1[1], 0.0], [a2[0], a2[1], 0.0], [0.0, 0.0, height]])
    built = BuiltStructure(record, selected, cell, expected_counts)
    if not built.counts_ok:
        warnings.warn(
            f"{record.id}: atoms per layer {built.layer_counts} differ from the expected "
            f"{expected_counts}; the fold tolerance ({fold_tolerance:.2e} Å) may be off.",
            RuntimeWarning,
            stacklevel=2,
        )
    return built


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------
def default_search_dirs(run_dir: PathLike) -> list[Path]:
    """Where relative POSCAR paths are looked for: the run directory and its ancestors.

    Runs normally live in ``<repo>/runs/<name>``, and recipes store paths relative to
    the repository root, so walking up from the run directory finds them no matter
    which directory the caller is in.
    """
    run_dir = Path(run_dir).resolve()
    return [run_dir, *run_dir.parents]


def write_structure(
    run_dir: PathLike,
    structure_id: str,
    fmt: str = "vasp",
    out: Optional[PathLike] = None,
    interlayer_gap: float = 3.0,
    vacuum: float = 50.0,
    search_dirs: Optional[Sequence[PathLike]] = None,
) -> Path:
    """Rebuild the atoms of ``structure_id`` from its recipe and write a file.

    ``fmt`` is ``"vasp"`` (POSCAR, Cartesian, via ASE) or ``"lammps"`` (LAMMPS data
    file, ``atom_style atomic``). ``out`` defaults to
    ``<run_dir>/structures/<id>.vasp`` (or ``.data``).

    >>> write_structure("runs/bto_trilayer_demo", trilayer_id, fmt="lammps")   # doctest: +SKIP
    PosixPath('runs/bto_trilayer_demo/structures/bto_3_22.62_36.87_c26a56.data')
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {sorted(FORMATS)}, got {fmt!r}")
    record = get(run_dir, structure_id)
    rotations = rotations_from_lineage(run_dir, structure_id)
    if not np.allclose(rotations, record.layer_rotations, atol=1e-9):
        raise ValueError(
            f"{structure_id}: stored layer_rotations {record.layer_rotations} disagree with the "
            f"lineage ({rotations}); the run directory is inconsistent."
        )
    if search_dirs is None:
        search_dirs = default_search_dirs(run_dir)
    built = build_atoms(record, rotations, interlayer_gap, vacuum, search_dirs=search_dirs)

    out = Path(out) if out is not None else Path(run_dir) / "structures" / f"{structure_id}{FORMATS[fmt]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    a1, a2, a3 = built.cell
    atoms = built.atoms[["type", "x", "y", "z"]]
    if fmt == "vasp":
        sw.write_vasp_ase(str(out), a1, a2, a3, atoms)
    else:
        sw.write_lammps_ase(str(out), a1, a2, a3, atoms)
    return out
