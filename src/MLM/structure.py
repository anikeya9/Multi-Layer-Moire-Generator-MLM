"""The structure *recipe*: one record per (multi)layer moiré candidate.

A :class:`MoireStructure` holds **metadata only** — lattice vectors, twist angles,
residuals, the ids of its parents and the paths to the primitive POSCAR files.
It never holds atom coordinates. Atoms are rebuilt on demand (see ``MLM.build``)
by following the ``parents`` links back to the level-0 roots.

Every record, whatever its level, is written to YAML with the *same* keys
(``null`` where a field does not apply). See ``docs/DESIGN.md`` §5 for the schema.

Frame convention
----------------
Every record lives in its own 2-D Cartesian frame: the frame of its **bottom
layer**, in which that layer is unrotated. ``lattice_vectors`` and
``layer_rotations`` are expressed in that frame. When two blocks are combined
(``MLM.search``), the upper block is rotated rigidly by ``twist`` (anticlockwise,
degrees) relative to the lower block's frame, and the new record inherits the
lower block's frame.

Example
-------
>>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
>>> root.level, root.n_layers, root.twist
(0, 1, None)
>>> path = root.to_yaml("runs/demo/yamls")          # doctest: +SKIP
>>> MoireStructure.from_yaml(path) == root           # doctest: +SKIP
True
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import yaml

from MLM.angle_between import angle_between
from MLM.moire_lattice_vector_finder import read_vasp

PathLike = Union[str, Path]

#: Order of the keys in every YAML recipe. The first twelve are the agreed schema
#: (DESIGN.md §5); the last four were added because the rebuild / index need them
#: (DESIGN.md §5.1). Keep this tuple and ``to_dict`` in sync.
SCHEMA_KEYS: tuple[str, ...] = (
    "id",
    "level",
    "n_layers",
    "material",
    "parents",
    "twist",
    "lattice_vectors",
    "angle_between_vectors",
    "residual",
    "cumulative_residual",
    "source_poscar",
    "units",
    # --- added fields ---
    "twist_sequence",
    "layer_rotations",
    "layer_residuals",
    "coefficients",
)

#: Number of hex characters of the sha1 hash used in the id.
HASH_LENGTH = 6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays into plain Python types.

    ``yaml.safe_dump`` only understands ``dict``/``list``/``str``/``int``/
    ``float``/``bool``/``None``; it refuses ``np.float64`` and ``np.ndarray``.

    >>> _to_builtin({"a": np.float64(1.5), "b": np.array([1, 2])})
    {'a': 1.5, 'b': [1, 2]}
    """
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def material_label(materials: Sequence[str]) -> str:
    """Material part of an id: a homostructure collapses to one label.

    >>> material_label(["bto", "bto", "bto"])
    'bto'
    >>> material_label(["pto", "sto"])
    'pto_sto'
    """
    if len(set(materials)) == 1:
        return materials[0]
    return "_".join(materials)


def resolve_path(path: PathLike, search_dirs: Sequence[PathLike] = ()) -> Path:
    """Find a (possibly relative) file path.

    ``source_poscar`` entries are stored exactly as the user wrote them, usually
    relative to the repository root. This function tries, in order: the path
    itself (absolute, or relative to the current working directory), then the
    path relative to each directory in ``search_dirs``.

    >>> resolve_path("moire_structures/bto/bto_1-layer.vasp", [".."])  # doctest: +SKIP
    PosixPath('../moire_structures/bto/bto_1-layer.vasp')
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for directory in search_dirs:
        joined = Path(directory) / candidate
        if joined.exists():
            return joined
    tried = [str(candidate)] + [str(Path(d) / candidate) for d in search_dirs]
    raise FileNotFoundError(f"Could not find {path!s}; tried: {tried}")


@lru_cache(maxsize=None)
def _read_primitive_cached(resolved_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a POSCAR once and cache it (search and build read the same files often)."""
    lattice, atom_types, coords = read_vasp(resolved_path)
    cell = np.array([lattice["a"], lattice["b"], lattice["c"]], dtype=float)
    _check_in_plane(cell, resolved_path)
    coords = np.atleast_2d(coords).astype(float)
    in_plane = cell[:2, :2].copy()
    return in_plane, cell, np.asarray(atom_types), coords


def _check_in_plane(cell: np.ndarray, path: str) -> None:
    """MLM assumes a, b lie in the xy plane and c points along z."""
    if not np.allclose(cell[:2, 2], 0.0) or not np.allclose(cell[2, :2], 0.0):
        raise ValueError(
            f"{path}: lattice vectors a, b must lie in the xy plane and c along z "
            f"(got\n{cell})."
        )


def read_primitive(poscar: PathLike, search_dirs: Sequence[PathLike] = ()) -> dict[str, np.ndarray]:
    """Read a primitive-layer POSCAR (cached).

    Returns a dict with ``in_plane`` (2x2, rows a1, a2), ``cell`` (3x3),
    ``atom_types`` (one symbol per atom) and ``coords`` (N x 3, Cartesian).

    >>> prim = read_primitive("moire_structures/bto/bto_1-layer.vasp")  # doctest: +SKIP
    >>> prim["in_plane"]                                                 # doctest: +SKIP
    array([[3.9903790951, 0.], [0., 3.9903790951]])
    """
    resolved = resolve_path(poscar, search_dirs)
    in_plane, cell, atom_types, coords = _read_primitive_cached(str(resolved.resolve()))
    # Return copies so that callers cannot corrupt the cache by editing arrays.
    return {
        "in_plane": in_plane.copy(),
        "cell": cell.copy(),
        "atom_types": atom_types.copy(),
        "coords": coords.copy(),
    }


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class MoireStructure:
    """Metadata recipe for one commensurate (multi)layer cell.

    Construct roots with :meth:`from_poscar` and read saved records with
    :meth:`from_yaml`. Records of level >= 1 are produced by
    ``MLM.search.search_pair``. If ``id`` is left empty it is generated from the
    content (see :meth:`make_id`).

    Attributes mirror the YAML keys; see ``docs/DESIGN.md`` §5 for their meaning.
    """

    level: int
    n_layers: int
    material: list[str]
    parents: list[str]
    twist: Optional[float]
    lattice_vectors: np.ndarray
    residual: Optional[float]
    cumulative_residual: Optional[float]
    source_poscar: list[str]
    units: str = "angstrom"
    twist_sequence: list[float] = field(default_factory=list)
    layer_rotations: list[float] = field(default_factory=lambda: [0.0])
    layer_residuals: Optional[list[float]] = None
    coefficients: Optional[dict[str, list[list[int]]]] = None
    id: str = ""

    def __post_init__(self) -> None:
        # Normalise types once, so that the rest of the code can rely on them.
        self.lattice_vectors = np.asarray(self.lattice_vectors, dtype=float).reshape(2, 2)
        self.material = [str(label) for label in self.material]
        self.parents = [str(parent) for parent in self.parents]
        self.source_poscar = [str(path) for path in self.source_poscar]
        self.layer_rotations = [float(angle) for angle in self.layer_rotations]
        self.twist_sequence = [float(angle) for angle in self.twist_sequence]
        if self.layer_residuals is not None:
            self.layer_residuals = [float(value) for value in self.layer_residuals]
        self._validate()
        if not self.id:
            self.id = self.make_id()

    def _validate(self) -> None:
        """Catch inconsistent records early (cheap checks only)."""
        if self.level != self.n_layers - 1:
            raise ValueError(f"level must be n_layers - 1, got level={self.level}, n_layers={self.n_layers}")
        for name in ("material", "source_poscar", "layer_rotations"):
            if len(getattr(self, name)) != self.n_layers:
                raise ValueError(f"{name} must have one entry per layer ({self.n_layers}), got {getattr(self, name)}")
        if self.level == 0:
            not_null = [name for name in ("twist", "residual", "cumulative_residual") if getattr(self, name) is not None]
            if self.parents or not_null:
                raise ValueError(f"a level-0 record must have parents=[] and null {not_null}")

    # --- alternative constructors -------------------------------------------
    @classmethod
    def from_poscar(
        cls,
        file_path: PathLike,
        material: str,
        units: str = "angstrom",
        search_dirs: Sequence[PathLike] = (),
    ) -> "MoireStructure":
        """Build a level-0 (root) record from a primitive-layer POSCAR.

        ``file_path`` is stored exactly as given (it is the pointer used later to
        rebuild atoms), so prefer paths relative to the repository root.

        >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
        >>> root.lattice_vectors.tolist()
        [[3.9903790951, 0.0], [0.0, 3.9903790951]]
        """
        primitive = read_primitive(file_path, search_dirs)
        return cls(
            level=0,
            n_layers=1,
            material=[material],
            parents=[],
            twist=None,
            lattice_vectors=primitive["in_plane"],
            residual=None,
            cumulative_residual=None,
            source_poscar=[str(file_path)],
            units=units,
            twist_sequence=[],
            layer_rotations=[0.0],
            layer_residuals=None,
            coefficients=None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoireStructure":
        """Rebuild a record from the plain dict produced by :meth:`to_dict`.

        ``angle_between_vectors`` is derived from ``lattice_vectors``, so the stored
        value is ignored. Unknown keys raise an error rather than being silently lost.

        >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
        >>> MoireStructure.from_dict(root.to_dict()) == root
        True
        """
        unknown = set(data) - set(SCHEMA_KEYS)
        if unknown:
            raise ValueError(f"unknown keys in structure record: {sorted(unknown)}")
        kwargs = {key: value for key, value in data.items() if key != "angle_between_vectors"}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: PathLike) -> "MoireStructure":
        """Read a record from a YAML recipe (``yaml.safe_load`` only).

        >>> record = MoireStructure.from_yaml("runs/demo/yamls/bto_1_51c2d7.yaml")  # doctest: +SKIP
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)

    # --- serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-Python dict in schema order, ready for ``yaml.safe_dump``.

        >>> MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto").to_dict()["twist"] is None
        True
        """
        data = {
            "id": self.id,
            "level": self.level,
            "n_layers": self.n_layers,
            "material": list(self.material),
            "parents": list(self.parents),
            "twist": self.twist,
            "lattice_vectors": self.lattice_vectors,
            "angle_between_vectors": self.angle_between_vectors,
            "residual": self.residual,
            "cumulative_residual": self.cumulative_residual,
            "source_poscar": list(self.source_poscar),
            "units": self.units,
            "twist_sequence": list(self.twist_sequence),
            "layer_rotations": list(self.layer_rotations),
            "layer_residuals": self.layer_residuals,
            "coefficients": self.coefficients,
        }
        return _to_builtin(data)

    def to_yaml(self, directory: PathLike) -> Path:
        """Write ``<directory>/<id>.yaml`` and return its path.

        >>> path = root.to_yaml("runs/demo/yamls")   # doctest: +SKIP
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.yaml"
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=None)
        return path

    # --- identity -------------------------------------------------------------
    def content_hash(self) -> str:
        """sha1 (first :data:`HASH_LENGTH` hex chars) over the canonical record.

        "Canonical" = every field except ``id``, serialised as JSON with sorted
        keys. Two records with identical content therefore get the same id, and
        two matches that only differ beyond the 2nd decimal of the twist do not.
        """
        content = self.to_dict()
        content.pop("id")
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]

    def make_id(self) -> str:
        """``<material>_<n_layers>[_<angles>]_<hash>``; see DESIGN.md §6.

        The naming is provisional — code must read the ``id`` stored in the file
        and never parse it.

        >>> MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto").make_id()  # doctest: +SKIP
        'bto_1_3f2a9c'
        """
        parts = [material_label(self.material), str(self.n_layers)]
        parts += [f"{angle:.2f}" for angle in self.twist_sequence]
        parts.append(self.content_hash())
        return "_".join(parts)

    # --- derived properties -----------------------------------------------------
    @property
    def angle_between_vectors(self) -> float:
        """Angle between ``a1`` and ``a2`` in degrees (reuses ``MLM.angle_between``)."""
        a1, a2 = self.lattice_vectors
        return angle_between(a1, a2)

    @property
    def cell_area(self) -> float:
        """Area of the in-plane cell, |a1 x a2|, in ``units``²."""
        return float(abs(np.linalg.det(self.lattice_vectors)))

    @property
    def is_root(self) -> bool:
        """True for level-0 (monolayer) records."""
        return self.level == 0

    @property
    def worst_layer_residual(self) -> Optional[float]:
        """max(layer_residuals): the real commensurability error of this cell, in Å."""
        if self.layer_residuals is None:
            return None
        return max(self.layer_residuals)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MoireStructure):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        twist = "None" if self.twist is None else f"{self.twist:.4f}"
        return f"MoireStructure(id={self.id!r}, n_layers={self.n_layers}, twist={twist})"
