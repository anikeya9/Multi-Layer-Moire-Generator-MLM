"""Sanity checks on a rebuilt structure, before it goes into DFT or MD.

:func:`check_built` looks at the atoms produced by :func:`MLM.build.build_atoms`:

- **atom count per layer** = (cell area / primitive area) x atoms per primitive cell;
- **composition** = the sum over layers of the primitive compositions times that factor;
- **overlapping atoms**: any two atoms closer than ``overlap_tol`` (default 0.5 Å) —
  a sign of a folding problem;
- **intra-layer distances**: the closest pair inside each layer must equal the closest
  pair of the (strained) primitive layer, within ``distance_tol`` — catches missing or
  doubled atoms at the cell boundary;
- **inter-layer distance**: the closest pair between neighbouring layers; flagged if
  below ``min_interlayer`` (default 1.0 Å), e.g. after a lateral shift or a small gap;
- **space group** (information only, needs ``spglib``).

The result is a :class:`CheckReport`; ``report.ok`` is False if any check failed.

>>> report = check_built(build_atoms(record))   # doctest: +SKIP
>>> print(report)                               # doctest: +SKIP
checks passed: 1365 atoms, composition Ba390O780Ti195, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from MLM.structure import PathLike, read_primitive


@dataclass
class CheckReport:
    """Outcome of :func:`check_built`: ``problems`` (failed checks) and ``info`` (numbers)."""

    problems: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        head = "checks passed" if self.ok else f"{len(self.problems)} check(s) FAILED"
        lines = [f"{head}: {self.info.get('n_atoms')} atoms, composition {self.info.get('formula')}"]
        lines += [f"  problem: {problem}" for problem in self.problems]
        for key in ("min_distance", "min_interlayer_distance", "space_group"):
            if self.info.get(key) is not None:
                value = self.info[key]
                lines.append(f"  {key}: {value:.4f} Å" if isinstance(value, float) else f"  {key}: {value}")
        return "\n".join(lines)


def to_ase(built: Any) -> Atoms:
    """``BuiltStructure`` → ``ase.Atoms`` (periodic in all directions)."""
    return Atoms(symbols=built.atoms["type"].tolist(), positions=built.atoms[["x", "y", "z"]].to_numpy(),
                 cell=built.cell, pbc=True)


def _closest_pair(atoms: Atoms, cutoff: float) -> Optional[float]:
    if len(atoms) < 2:
        return None
    distances = neighbor_list("d", atoms, cutoff)
    return float(distances.min()) if len(distances) else None


def _primitive_min_distance(poscar: str, strain: np.ndarray, search_dirs: Sequence[PathLike]) -> Optional[float]:
    """Closest pair in one (strained) primitive layer, periodic in-plane only."""
    primitive = read_primitive(poscar, search_dirs)
    cell = primitive["cell"].copy()
    deformation = np.eye(2) + strain
    cell[:2, :2] = cell[:2, :2] @ deformation.T
    positions = primitive["coords"].copy()
    positions[:, :2] = positions[:, :2] @ deformation.T
    layer = Atoms(symbols=list(primitive["atom_types"]), positions=positions, cell=cell, pbc=[True, True, False])
    return _closest_pair(layer, cutoff=6.0)


def space_group(atoms: Atoms, symprec: float = 1e-3) -> Optional[str]:
    """International symbol and number, e.g. ``'P4 (75)'``; ``None`` if spglib is missing."""
    try:
        import spglib
    except ImportError:
        return None
    dataset = spglib.get_symmetry_dataset((atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers),
                                          symprec=symprec)
    if dataset is None:
        return None
    return f"{dataset.international} ({dataset.number})"


def check_built(
    built: Any,
    search_dirs: Sequence[PathLike] = (),
    overlap_tol: float = 0.5,
    min_interlayer: float = 1.0,
    distance_tol: float = 0.02,
    symmetry: bool = True,
) -> CheckReport:
    """Run all checks on a :class:`MLM.build.BuiltStructure`.

    ``distance_tol`` (Å) allows for the cell's commensurability residual when comparing
    intra-layer distances with the primitive layer.
    """
    record = built.record
    report = CheckReport()
    atoms = to_ase(built)
    report.info["n_atoms"] = len(atoms)
    report.info["formula"] = atoms.get_chemical_formula()

    # 1. counts per layer and composition
    if built.layer_counts != built.expected_counts:
        report.problems.append(f"atoms per layer {built.layer_counts} != expected {built.expected_counts}")
    expected_formula: dict[str, int] = {}
    for poscar, expected in zip(record.source_poscar, built.expected_counts):
        types = list(read_primitive(poscar, search_dirs)["atom_types"])
        multiplicity = expected // max(len(types), 1)
        for symbol in types:
            expected_formula[symbol] = expected_formula.get(symbol, 0) + multiplicity
    if built.composition != dict(sorted(expected_formula.items())):
        report.problems.append(f"composition {built.composition} != expected {dict(sorted(expected_formula.items()))}")

    # 2. overlapping atoms anywhere
    report.info["min_distance"] = _closest_pair(atoms, cutoff=max(3.0, overlap_tol))
    if report.info["min_distance"] is not None and report.info["min_distance"] < overlap_tol:
        report.problems.append(f"atoms overlap: closest pair {report.info['min_distance']:.3f} Å < {overlap_tol} Å")

    # 3. inside each layer: same closest pair as the primitive layer
    layer_of_atom = built.atoms["layer"].to_numpy()
    strains = record.strain_tensors()
    report.info["layers"] = []
    for layer in range(record.n_layers):
        mask = layer_of_atom == layer
        inside = _closest_pair(atoms[np.nonzero(mask)[0]], cutoff=6.0)
        reference = _primitive_min_distance(record.source_poscar[layer], strains[layer], search_dirs)
        report.info["layers"].append({"layer": layer + 1, "min_distance": inside, "primitive": reference})
        if inside is not None and reference is not None and abs(inside - reference) > distance_tol:
            report.problems.append(f"layer {layer + 1}: closest pair {inside:.4f} Å differs from the primitive "
                                   f"layer's {reference:.4f} Å (missing or doubled atoms?)")

    # 4. between neighbouring layers
    closest = None
    if record.n_layers > 1:
        first, second, distances = neighbor_list("ijd", atoms, max(min_interlayer, 6.0))
        between = np.abs(layer_of_atom[first] - layer_of_atom[second]) == 1
        if between.any():
            closest = float(distances[between].min())
        if closest is not None and closest < min_interlayer:
            report.problems.append(f"layers too close: {closest:.3f} Å < {min_interlayer} Å between neighbouring layers")
    report.info["min_interlayer_distance"] = closest

    # 5. symmetry (information only)
    if symmetry:
        report.info["space_group"] = space_group(atoms)
    return report
