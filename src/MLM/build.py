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
>>> write_structure("runs/bto_trilayer_demo", "bto_3_22.62_36.87_015688", fmt="vasp")   # doctest: +SKIP
PosixPath('runs/bto_trilayer_demo/structures/bto_3_22.62_36.87_015688.vasp')
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

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

    Block-rigid rule, with the twist measured from the layer directly below: a root
    has ``[0]``; a child of (lower, upper) at ``twist`` has
    ``rotations(lower) + [rotations(lower)[-1] + twist + r for r in rotations(upper)]``.

    >>> rotations_from_lineage("runs/bto_trilayer_demo", trilayer_id)   # doctest: +SKIP
    [0.0, 22.62, 59.49]
    """
    def rotations(node) -> list[float]:
        if not node.parents:
            return [0.0]
        lower, upper = node.parents
        lower_rotations = rotations(lower)
        upper_base = lower_rotations[-1] + node.record.twist  # new angle of the upper block's bottom layer
        return lower_rotations + [upper_base + angle for angle in rotations(upper)]

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


def _strain_xy(frame: pd.DataFrame, strain: np.ndarray) -> pd.DataFrame:
    """Apply the in-plane strain ``(I + ε)`` to x, y (fractional coordinates kept, z unchanged)."""
    if not np.any(strain):
        return frame
    deformation = np.eye(2) + strain
    xy = frame[["x", "y"]].to_numpy() @ deformation.T
    strained = frame.copy()
    strained["x"], strained["y"] = xy[:, 0], xy[:, 1]
    return strained


def _replicated_layer(
    poscar: str, radius: float, search_dirs: Sequence[PathLike], strain: Optional[np.ndarray] = None
) -> tuple[pd.DataFrame, int]:
    """Atoms of one (optionally strained) primitive layer over a disc of ``radius``; also atoms/cell.

    ``strain`` is the layer's 2x2 strain tensor (fractions) in its own frame. It is
    applied to the replicated positions, which is the same as straining the primitive
    cell and keeping every atom's fractional coordinates; z is not changed.
    """
    strain = np.zeros((2, 2)) if strain is None else np.asarray(strain)
    primitive = read_primitive(poscar, search_dirs)
    cell, types, coords = primitive["cell"], primitive["atom_types"], primitive["coords"]
    strained_in_plane = primitive["in_plane"] @ (np.eye(2) + strain).T
    n = replicas_needed(strained_in_plane, radius)
    _, _, positions, replicated_types = sw.replicate_atoms(
        a=cell[0], b=cell[1], c=cell[2],
        atom_data=coords, atom_type_arr=types, natoms=len(coords),
        na=n, nb=n, nc=1,
    )
    frame = pd.DataFrame(positions, columns=["x", "y", "z"])
    frame["type"] = replicated_types
    return _strain_xy(frame, strain), len(coords)


def build_atoms(
    record: MoireStructure,
    rotations: Optional[Sequence[float]] = None,
    interlayer_gap: float = 3.0,
    vacuum: float = 50.0,
    fold_tolerance: Optional[float] = None,
    search_dirs: Sequence[PathLike] = (),
    shifts: Optional[Sequence[Optional[Sequence[float]]]] = None,
    poisson_ratios: Optional[Sequence[Optional[float]]] = None,
) -> BuiltStructure:
    """Atoms of ``record``'s moiré cell.

    - ``rotations``: absolute rotation of each layer; defaults to the stored
      ``record.layer_rotations`` (``write_structure`` passes the lineage-derived ones).
    - ``interlayer_gap``: Å between the highest atom of a layer and the lowest
      *possible* position of the next (the notebook rule: ``z += gap + max(z_below)``).
    - ``vacuum``: c = (top − bottom) + ``vacuum`` (notebook: 50 Å).
    - ``fold_tolerance``: ``dtol`` for ``select_atoms_fractional``; defaults to the
      cell's worst per-layer residual (the notebooks passed the step's ``delvec``).

    - ``shifts``: per layer, an in-plane translation ``(s1, s2)`` in fractional
      coordinates of that layer's own (strained) primitive cell, applied before the
      rotation (``None`` or ``(0, 0)`` = no shift). Stacking registry, e.g. AB stacking of
      hexagonal layers = ``(1/3, 2/3)``. A translation does not change commensurability.
    - ``poisson_ratios``: per layer, ν for the thickness response to in-plane strain:
      ε_zz = −ν/(1−ν)·(ε_xx + ε_yy) (thin isotropic film under biaxial in-plane stress),
      applied about the middle of the layer. ``None`` = thickness unchanged.

    Each layer is strained by its ``record.layer_strains`` tensor (in its own frame)
    before it is rotated, so strained layers fit the cell exactly.

    >>> built = build_atoms(get("runs/demo", some_id))   # doctest: +SKIP
    >>> built.counts_ok, built.composition                # doctest: +SKIP
    (True, {'Ba': 390, 'O': 780, 'Ti': 195})
    """
    rotations = list(record.layer_rotations if rotations is None else rotations)
    shifts = _per_layer(shifts, record.n_layers, "shifts")
    poisson_ratios = _per_layer(poisson_ratios, record.n_layers, "poisson_ratios")
    a1, a2 = record.lattice_vectors
    strains = record.strain_tensors()
    displacements = [_shift_vector(poscar, shift, strain, search_dirs)
                     for poscar, shift, strain in zip(record.source_poscar, shifts, strains)]
    # farthest cell corner from the origin, plus the largest shift
    radius = float(np.linalg.norm(a1) + np.linalg.norm(a2) + max(np.linalg.norm(d) for d in displacements))

    layers = []
    expected_counts = []
    top_of_previous: Optional[float] = None
    for position, (poscar, angle) in enumerate(zip(record.source_poscar, rotations)):
        frame, atoms_per_cell = _replicated_layer(poscar, radius, search_dirs, strains[position])
        frame = _poisson_z(frame, strains[position], poisson_ratios[position])
        frame["x"] = frame["x"] + displacements[position][0]
        frame["y"] = frame["y"] + displacements[position][1]
        if top_of_previous is not None:
            frame["z"] = frame["z"] + interlayer_gap + top_of_previous
        top_of_previous = float(frame["z"].max())
        frame = _rotate_xy(frame, angle)
        frame["layer"] = position
        layers.append(frame)
        primitive_area = abs(np.linalg.det(read_primitive(poscar, search_dirs)["in_plane"]))
        primitive_area *= abs(np.linalg.det(np.eye(2) + strains[position]))  # strained cell area
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


def _per_layer(values: Optional[Sequence[Any]], n_layers: int, name: str) -> list[Any]:
    """One entry per layer (``None`` → all ``None``); raise if the length is wrong."""
    if values is None:
        return [None] * n_layers
    values = list(values)
    if len(values) != n_layers:
        raise ValueError(f"{name} needs one entry per layer ({n_layers}), got {len(values)}")
    return values


def _shift_vector(poscar: str, shift: Optional[Sequence[float]], strain: np.ndarray,
                  search_dirs: Sequence[PathLike]) -> np.ndarray:
    """Cartesian (own-frame) displacement for a fractional shift of the strained primitive cell."""
    if shift is None:
        return np.zeros(2)
    s1, s2 = (float(value) for value in shift)
    in_plane = read_primitive(poscar, search_dirs)["in_plane"] @ (np.eye(2) + strain).T  # rows: strained a, b
    return s1 * in_plane[0] + s2 * in_plane[1]


def _poisson_z(frame: pd.DataFrame, strain: np.ndarray, nu: Optional[float]) -> pd.DataFrame:
    """Scale the layer thickness by (1 + ε_zz), ε_zz = −ν/(1−ν)·tr(ε), about the layer's middle."""
    if nu is None or not np.any(strain):
        return frame
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio must lie in (-1, 0.5), got {nu}")
    eps_zz = -nu / (1.0 - nu) * float(strain[0, 0] + strain[1, 1])
    middle = 0.5 * (frame["z"].min() + frame["z"].max())
    scaled = frame.copy()
    scaled["z"] = middle + (frame["z"] - middle) * (1.0 + eps_zz)
    return scaled


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
    shifts: Optional[Sequence[Optional[Sequence[float]]]] = None,
    poisson_ratios: Optional[Sequence[Optional[float]]] = None,
    check: bool = True,
) -> Path:
    """Rebuild the atoms of ``structure_id`` from its recipe and write a file.

    ``fmt`` is ``"vasp"`` (POSCAR, Cartesian, via ASE) or ``"lammps"`` (LAMMPS data
    file, ``atom_style atomic``). ``out`` defaults to
    ``<run_dir>/structures/<id>.vasp`` (or ``.data``). ``shifts`` and
    ``poisson_ratios`` as in :func:`build_atoms`. With ``check=True`` (default) the
    atoms are checked first (:func:`MLM.checks.check_built`) and a ``RuntimeWarning``
    describes any failed check; :func:`check_structure` returns the full report.

    >>> write_structure("runs/bto_trilayer_demo", trilayer_id, fmt="lammps")   # doctest: +SKIP
    PosixPath('runs/bto_trilayer_demo/structures/bto_3_22.62_36.87_015688.data')
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {sorted(FORMATS)}, got {fmt!r}")
    if search_dirs is None:
        search_dirs = default_search_dirs(run_dir)
    built = _build_from_run(run_dir, structure_id, interlayer_gap, vacuum, search_dirs, shifts, poisson_ratios)
    if check:
        from MLM.checks import check_built

        report = check_built(built, search_dirs=search_dirs, symmetry=False)
        if not report.ok:
            warnings.warn(f"{structure_id}: {report}", RuntimeWarning, stacklevel=2)

    out = Path(out) if out is not None else Path(run_dir) / "structures" / f"{structure_id}{FORMATS[fmt]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    a1, a2, a3 = built.cell
    atoms = built.atoms[["type", "x", "y", "z"]]
    if fmt == "vasp":
        sw.write_vasp_ase(str(out), a1, a2, a3, atoms)
    else:
        sw.write_lammps_ase(str(out), a1, a2, a3, atoms)
    return out


def _build_from_run(run_dir, structure_id, interlayer_gap, vacuum, search_dirs, shifts, poisson_ratios):
    """Load a recipe, check its rotations against the lineage and build its atoms."""
    record = get(run_dir, structure_id)
    rotations = rotations_from_lineage(run_dir, structure_id)
    if not np.allclose(rotations, record.layer_rotations, atol=1e-9):
        raise ValueError(
            f"{structure_id}: stored layer_rotations {record.layer_rotations} disagree with the "
            f"lineage ({rotations}); the run directory is inconsistent."
        )
    return build_atoms(record, rotations, interlayer_gap, vacuum, search_dirs=search_dirs,
                       shifts=shifts, poisson_ratios=poisson_ratios)


def check_structure(
    run_dir: PathLike,
    structure_id: str,
    interlayer_gap: float = 3.0,
    vacuum: float = 50.0,
    search_dirs: Optional[Sequence[PathLike]] = None,
    shifts: Optional[Sequence[Optional[Sequence[float]]]] = None,
    poisson_ratios: Optional[Sequence[Optional[float]]] = None,
):
    """Build a structure exactly as :func:`write_structure` would and return its
    :class:`MLM.checks.CheckReport` (including the space group, if spglib is installed).

    >>> print(check_structure("runs/bto_trilayer_demo", trilayer_id))   # doctest: +SKIP
    """
    from MLM.checks import check_built

    if search_dirs is None:
        search_dirs = default_search_dirs(run_dir)
    built = _build_from_run(run_dir, structure_id, interlayer_gap, vacuum, search_dirs, shifts, poisson_ratios)
    return check_built(built, search_dirs=search_dirs)

