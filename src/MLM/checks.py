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
- **PBC seam** (:func:`environment_errors`): every atom must have the same in-layer
  neighbours (species and distances, across the periodic boundaries) as some atom of
  its own primitive layer, strained and with the Poisson thickness change as built.
  Broken or stretched bonds at the cell boundary, missing or doubled boundary atoms and
  layers that do not really fit the cell all show up here; works for hetero- and
  N-layer stacks;
- **space group** (information only, needs ``spglib``; slow for very large cells).

All distance checks share one neighbour list. The result is a :class:`CheckReport`;
``report.ok`` is False if any check failed.

>>> report = check_built(build_atoms(record))   # doctest: +SKIP
>>> print(report)                               # doctest: +SKIP
checks passed: 1365 atoms, composition Ba390O780Ti195, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
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
        for key in ("min_distance", "min_interlayer_distance", "seam_max_error", "space_group"):
            if self.info.get(key) is not None:
                value = self.info[key]
                lines.append(f"  {key}: {value:.4f} Å" if isinstance(value, float) else f"  {key}: {value}")
        return "\n".join(lines)


def to_ase(built: Any) -> Atoms:
    """``BuiltStructure`` → ``ase.Atoms`` (periodic in all directions)."""
    return Atoms(symbols=built.atoms["type"].tolist(), positions=built.atoms[["x", "y", "z"]].to_numpy(),
                 cell=built.cell, pbc=True)


def _with_ghosts(frac: np.ndarray, cell: np.ndarray, pbc: Sequence[bool], cutoff: float):
    """Cartesian positions of atoms at fractional ``frac`` (wrapped into the cell), then the
    same atoms followed by their periodic images within ``cutoff`` of the cell ("ghosts"),
    and for each of those the index of the atom it is an image of."""
    from itertools import product

    positions = frac @ cell
    faces = np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]])  # normals of the face pairs (a2xa3, a3xa1, a1xa2)
    margin = cutoff * np.linalg.norm(faces, axis=1) / abs(np.linalg.det(cell))  # cutoff / face spacing
    periodic = np.asarray(pbc, dtype=bool)
    reach = [int(np.ceil(m)) if p else 0 for m, p in zip(margin, periodic)]
    images: list[np.ndarray] = [positions]  # the atoms themselves come first
    owners: list[np.ndarray] = [np.arange(len(frac))]
    for shift in product(*(range(-n, n + 1) for n in reach)):
        if not any(shift):
            continue
        shifted = frac + shift
        near = np.all(((shifted > -margin) & (shifted < 1 + margin))[:, periodic], axis=1)
        images.append(positions[near] + np.asarray(shift) @ cell)
        owners.append(np.nonzero(near)[0])
    return positions, np.vstack(images), np.concatenate(owners)


def neighbor_pairs(atoms: Atoms, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pairs of ``ase.neighborlist.neighbor_list("ijd", atoms, cutoff)``, via a scipy
    KD-tree on the atoms and their periodic images near the cell: ~25x faster and ~15x
    less memory on large cells."""
    from scipy.spatial import cKDTree

    positions, extended, owner = _with_ghosts(atoms.get_scaled_positions(wrap=True), np.array(atoms.cell),
                                              atoms.pbc, cutoff)
    pairs = cKDTree(positions).sparse_distance_matrix(cKDTree(extended), cutoff, output_type="ndarray")
    del extended
    keep = pairs["j"] != pairs["i"]  # an atom and itself (same image) is not a pair
    first, ghost, distances = pairs["i"][keep], pairs["j"][keep], pairs["v"][keep]
    del pairs, keep
    return first.astype(np.intp, copy=False), owner[ghost], distances


def closest_between(atoms: Atoms, lower: np.ndarray, upper: np.ndarray, cutoff: float) -> Optional[float]:
    """Closest distance (periodic) between the atoms selected by masks ``lower`` and
    ``upper``; ``None`` if none closer than ``cutoff``. One nearest-neighbour query per
    atom, so memory stays O(N) however large ``cutoff`` is."""
    from scipy.spatial import cKDTree

    frac = atoms.get_scaled_positions(wrap=True)
    _, extended, _ = _with_ghosts(frac[upper], np.array(atoms.cell), atoms.pbc, cutoff)
    distances, _ = cKDTree(extended).query(frac[lower] @ np.array(atoms.cell), k=1, distance_upper_bound=cutoff)
    finite = distances[np.isfinite(distances)]
    return float(finite.min()) if len(finite) else None


def _closest_pair(atoms: Atoms, cutoff: float) -> Optional[float]:
    if len(atoms) < 2:
        return None
    distances = neighbor_list("d", atoms, cutoff)
    return float(distances.min()) if len(distances) else None


def reference_layer(poscar: str, strain: Optional[np.ndarray] = None, poisson: Optional[float] = None,
                    search_dirs: Sequence[PathLike] = ()) -> Atoms:
    """One primitive layer exactly as :func:`MLM.build.build_atoms` makes it: in-plane strain
    ``(I + ε)`` on cell and positions, then the Poisson thickness change. Periodic in-plane only."""
    from MLM.build import _poisson_z  # same thickness formula as the builder

    primitive = read_primitive(poscar, search_dirs)
    strain = np.zeros((2, 2)) if strain is None else np.asarray(strain, dtype=float)
    deformation = np.eye(3)
    deformation[:2, :2] += strain
    cell = primitive["cell"] @ deformation.T
    positions = primitive["coords"] @ deformation.T
    positions[:, 2] = _poisson_z(pd.DataFrame(positions, columns=["x", "y", "z"]), strain, poisson)["z"]
    return Atoms(symbols=list(primitive["atom_types"]), positions=positions, cell=cell, pbc=[True, True, False])


def _sorted_environments(atoms: Atoms, first: np.ndarray, second: np.ndarray, distances: np.ndarray):
    """Pairs sorted by (atom, neighbour species, distance), and where each atom's block starts."""
    numbers = atoms.numbers
    order = np.lexsort((distances, numbers[second], first))
    first, species, distances = first[order], numbers[second][order], distances[order]
    counts = np.bincount(first, minlength=len(atoms))
    return species, distances, counts, np.cumsum(counts) - counts


def environment_errors(atoms: Atoms, layer_of_atom: np.ndarray, references: Sequence[Atoms],
                       cutoff: float = 4.5, pairs: Optional[tuple] = None) -> np.ndarray:
    """Per atom, the largest error (Å) of its in-layer neighbour distances against the
    best-matching atom of ``references[layer]`` (``inf``: neighbour species or counts match
    no atom of the primitive layer, e.g. a missing or doubled neighbour).

    A layer of a moiré cell is a rotated (and maybe strained) supercell of its primitive
    layer, so under PBC every one of its atoms must see exactly a primitive environment.
    ``pairs``: a precomputed ``neighbor_list("ijd", atoms, r)`` with ``r >= cutoff``.
    """
    first, second, distances = neighbor_list("ijd", atoms, cutoff) if pairs is None else pairs
    keep = (layer_of_atom[first] == layer_of_atom[second]) & (distances <= cutoff)
    species, distances, counts, starts = _sorted_environments(atoms, first[keep], second[keep], distances[keep])
    errors = np.full(len(atoms), np.inf)
    for layer, reference in enumerate(references):
        ref_species, ref_distances, ref_counts, ref_starts = _sorted_environments(
            reference, *neighbor_list("ijd", reference, cutoff))
        in_layer = layer_of_atom == layer
        for site in range(len(reference)):
            n = ref_counts[site]
            site_species = ref_species[ref_starts[site]:ref_starts[site] + n]
            site_distances = ref_distances[ref_starts[site]:ref_starts[site] + n]
            members = np.nonzero(in_layer & (atoms.numbers == reference.numbers[site]) & (counts == n))[0]
            if not len(members):
                continue
            block = starts[members, None] + np.arange(n)  # (members, n): each atom's sorted neighbours
            same_species = (species[block] == site_species).all(axis=1)
            error = np.abs(distances[block] - site_distances).max(axis=1, initial=0.0)
            errors[members] = np.minimum(errors[members], np.where(same_species, error, np.inf))
    return errors


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
    poisson_ratios: Optional[Sequence[Optional[float]]] = None,
    seam: bool = True,
    seam_cutoff: float = 4.5,
) -> CheckReport:
    """Run all checks on a :class:`MLM.build.BuiltStructure`.

    ``distance_tol`` (Å) allows for the cell's commensurability residual when comparing
    intra-layer distances with the primitive layer (also the seam tolerance).
    ``poisson_ratios``: the ones the structure was built with (as in ``build_atoms``).
    ``seam``: run the PBC seam check, comparing neighbours up to ``seam_cutoff`` Å.
    """
    record = built.record
    report = CheckReport()
    atoms = to_ase(built)
    report.info["n_atoms"] = len(atoms)
    report.info["formula"] = atoms.get_chemical_formula()
    poisson_ratios = list(poisson_ratios) if poisson_ratios is not None else [None] * record.n_layers
    strains = record.strain_tensors()
    references = [reference_layer(poscar, strain, nu, search_dirs)
                  for poscar, strain, nu in zip(record.source_poscar, strains, poisson_ratios)]

    # One neighbour list for the overlap, intra-layer and seam checks (the most expensive step).
    layer_of_atom = built.atoms["layer"].to_numpy()
    first, second, distances = neighbor_pairs(atoms, max(seam_cutoff, overlap_tol, 3.0))
    same_layer = layer_of_atom[first] == layer_of_atom[second]

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
    report.info["min_distance"] = float(distances.min()) if len(distances) else None
    if report.info["min_distance"] is not None and report.info["min_distance"] < overlap_tol:
        report.problems.append(f"atoms overlap: closest pair {report.info['min_distance']:.3f} Å < {overlap_tol} Å")

    # 3. inside each layer: same closest pair as the primitive layer, and (seam check)
    #    every atom with a primitive environment
    errors = (environment_errors(atoms, layer_of_atom, references, seam_cutoff, pairs=(first, second, distances))
              if seam else None)
    report.info["layers"] = []
    for layer in range(record.n_layers):
        in_layer = same_layer & (layer_of_atom[first] == layer)
        inside = float(distances[in_layer].min()) if in_layer.any() else None
        reference = _closest_pair(references[layer], cutoff=6.0)
        info: dict[str, Any] = {"layer": layer + 1, "min_distance": inside, "primitive": reference}
        report.info["layers"].append(info)
        if inside is not None and reference is not None and abs(inside - reference) > distance_tol:
            report.problems.append(f"layer {layer + 1}: closest pair {inside:.4f} Å differs from the primitive "
                                   f"layer's {reference:.4f} Å (missing or doubled atoms?)")
        if errors is not None:
            layer_errors = errors[layer_of_atom == layer]
            finite = layer_errors[np.isfinite(layer_errors)]
            info["seam_max_error"] = float(finite.max()) if len(finite) else None
            info["seam_bad_atoms"] = int((layer_errors > distance_tol).sum())
            if info["seam_bad_atoms"]:
                wrong_count = int(np.isinf(layer_errors).sum())
                report.problems.append(
                    f"layer {layer + 1}: {info['seam_bad_atoms']} of {len(layer_errors)} atoms have neighbours unlike "
                    f"the primitive layer ({wrong_count} with missing/extra neighbours, largest distance error "
                    f"{info['seam_max_error'] or 0:.4f} Å): broken bonds at the cell boundary, or a layer that "
                    f"does not fit the cell")
    if errors is not None:
        finite = errors[np.isfinite(errors)]
        report.info["seam_max_error"] = float(finite.max()) if len(finite) else None

    del first, second, distances, same_layer  # free the pair list before the next step

    # 4. between neighbouring layers (nearest-neighbour queries: up to 6 Å without a 6 Å pair list)
    closest = None
    if record.n_layers > 1:
        gaps = [closest_between(atoms, layer_of_atom == layer, layer_of_atom == layer + 1, max(min_interlayer, 6.0))
                for layer in range(record.n_layers - 1)]
        closest = min((gap for gap in gaps if gap is not None), default=None)
        if closest is not None and closest < min_interlayer:
            report.problems.append(
                f"layers too close: {closest:.3f} Å < {min_interlayer} Å between neighbouring layers")
    report.info["min_interlayer_distance"] = closest

    # 5. symmetry (information only)
    if symmetry:
        report.info["space_group"] = space_group(atoms)
    return report
