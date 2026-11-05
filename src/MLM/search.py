"""Pairwise and hierarchical (tournament-tree) moiré search.

This module *wraps* ``MLM.match`` — it never changes it. The building block is

    search_pair(lower, upper, params, filters) -> list[MoireStructure]

which sweeps the twist of ``upper`` relative to ``lower`` with
``match.run_and_filter`` and turns every commensurate cell it finds into a new
:class:`~MLM.structure.MoireStructure` with ``parents = [lower.id, upper.id]``.

``build_stack`` (further down) only orchestrates repeated calls of the same
function over the pairing plan produced by :func:`plan_rounds`.

Layout of this file
-------------------
1. ``SearchParams`` / ``Filters``   – what to search and what to keep
2. raw matches → candidate cells    – conversion + de-duplication
3. ``find_candidates`` / ``search_pair``
4. pairing planner                  – pure functions, no I/O
5. ``build_stack``                  – the tree driver (writes YAMLs + index)
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import yaml

from MLM.index import SPEC_FILENAME, RunStore, get
from MLM.runinfo import RunInfo
from MLM import fastscan
from MLM.match import run_and_filter
from MLM.strain import STRAIN_MODES, Elastic, fit_strain, scan_tolerance, strain_energy, strain_magnitude
from MLM.structure import MoireStructure, PathLike, read_primitive

# ===========================================================================
# 1. Parameters and filters
# ===========================================================================

#: Allowed values of ``SearchParams.merge_equivalent_twists``.
MERGE_MODES = ("none", "rotation", "rotation+mirror")

#: Per-angle match cap used by ``max_matches_per_angle="auto"`` in strained searches.
STRAINED_MATCHES_PER_ANGLE = 200

#: Allowed values of ``SearchParams.scanner``.
SCANNERS = ("fast", "match")


@dataclass(frozen=True)
class SearchParams:
    """Parameters of one pairwise search (one call of ``match.run_and_filter``).

    The first seven fields are passed straight to ``run_and_filter``:

    - ``degmin``, ``degmax``, ``degstep`` — twist sweep in degrees (``degmax`` excluded,
      as in ``np.arange``). The twist is the rotation of the upper block's bottom layer
      relative to the lower block's **top** layer (the layer directly below it); see
      :meth:`window_for` for how this becomes the absolute angle ``match.run`` sweeps.
    - ``dtol`` — tolerance on the *fractional* coordinates of a match (``match.scan``'s
      ``tol``); the Å residual of each match is reported separately as ``delvec``.
    - ``nmin``, ``nmax`` — range of integer coefficients searched on the lower lattice.
    - ``goodangles`` — target angles (deg) between the two moiré vectors.

    The remaining fields control how the raw matches are post-processed here:

    - ``dedupe`` — merge matches that describe the same cell (see :func:`deduplicate`).
    - ``keep_supercells`` — keep non-minimal cells (integer multiples) found at the
      same twist. ``False`` keeps only the smallest cell, like the notebooks did.
    - ``angle_cluster_tol`` — matches with the same lattice at twists closer than this
      (deg) are the same structure seen at neighbouring grid angles; only the
      lowest-residual one is kept. ``None`` → ``1.5 * degstep``.
    - ``strain_mode`` — which strain tensors are allowed when a layer has a strain limit
      (see :mod:`MLM.strain`): ``"isotropic"`` (default; ε = e·I), ``"axial"``
      (ε = diag(e_xx, e_yy)) or ``"full"`` (any symmetric ε). Ignored without strain.
    - ``scanner`` — ``"fast"`` (default): :mod:`MLM.fastscan`, a vectorised version of
      ``match.run`` that returns identical matches 10–70× faster; ``"match"``: call
      ``match.run_and_filter`` itself (the reference implementation).
    - ``include_axes`` — also try lattice points with a zero coefficient (``n = (k, 0)``),
      which ``match.scan`` skips (``"fast"`` scanner only).
    - ``max_matches_per_angle`` — ``"auto"`` (default): 25 like ``match.scan`` for rigid
      searches, :data:`STRAINED_MATCHES_PER_ANGLE` (200) for strained ones; or an integer,
      or ``None`` for unlimited (``"fast"`` scanner only). With strain ε *any* cell larger
      than about a/ε can be matched, so the list per angle grows without bound with the n
      range; 25 would crowd out the longer, less strained cells, unlimited explodes in
      later rounds (PTO/STO/PTO round 2: 1.5 s with 200, ~2 min unlimited). In round 1,
      200 finds every cell at 1 %/1 % limits and ~80 % at 2 %/2 %.
    - ``merge_equivalent_twists`` — ``"none"`` (default), ``"rotation"`` or
      ``"rotation+mirror"``: treat twists related by the upper layer's lattice symmetry
      as one structure (see :func:`merge_equivalent_twists`).
    - ``strain_residual_tol`` — Å; a strained cell is accepted only if straining closes
      it to within this (the point of straining is a perfect match). Mainly matters for
      ``isotropic``/``axial``, which cannot fix a mismatch in cell *shape*.

    >>> SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    SearchParams(degmin=22.0, degmax=23.0, degstep=0.01, dtol=0.0005, nmin=-10, nmax=10, ...)
    """

    degmin: float = 1.0
    degmax: float = 45.0
    degstep: float = 0.01
    dtol: float = 5e-4
    nmin: int = -10
    nmax: int = 10
    goodangles: tuple[float, ...] = (90.0,)
    dedupe: bool = True
    keep_supercells: bool = False
    angle_cluster_tol: Optional[float] = None
    strain_mode: str = "isotropic"
    strain_residual_tol: float = 1e-6
    scanner: str = "fast"
    merge_equivalent_twists: str = "none"
    include_axes: bool = False
    max_matches_per_angle: Union[int, None, str] = "auto"

    def __post_init__(self) -> None:
        # Accept lists from YAML but store an (immutable, hashable) tuple.
        object.__setattr__(self, "goodangles", tuple(float(a) for a in self.goodangles))
        if self.degmax <= self.degmin or self.degstep <= 0:
            raise ValueError(f"need degmin < degmax and degstep > 0, got {self.degmin}, {self.degmax}, {self.degstep}")
        if self.merge_equivalent_twists not in MERGE_MODES:
            raise ValueError(f"merge_equivalent_twists must be one of {MERGE_MODES}, got {self.merge_equivalent_twists!r}")
        if self.scanner not in SCANNERS:
            raise ValueError(f"scanner must be one of {SCANNERS}, got {self.scanner!r}")
        if self.max_matches_per_angle is not None and self.max_matches_per_angle != "auto" \
                and not (isinstance(self.max_matches_per_angle, int) and self.max_matches_per_angle > 0):
            raise ValueError(f"max_matches_per_angle must be 'auto', None or a positive integer, "
                             f"got {self.max_matches_per_angle!r}")
        if self.scanner == "match" and (self.include_axes or self.max_matches_per_angle not in ("auto", 25)):
            raise ValueError("include_axes / max_matches_per_angle need scanner='fast' "
                             "(match.scan has them built in: no axes, 25 matches per angle)")
        if self.strain_mode not in STRAIN_MODES:
            raise ValueError(f"strain_mode must be one of {STRAIN_MODES}, got {self.strain_mode!r}")
        if self.nmax <= self.nmin:
            raise ValueError(f"need nmin < nmax, got {self.nmin}, {self.nmax}")

    @property
    def cluster_tol(self) -> float:
        """Effective ``angle_cluster_tol`` in degrees."""
        if self.angle_cluster_tol is None:
            return 1.5 * self.degstep
        return self.angle_cluster_tol

    def window_for(self, lower: MoireStructure) -> tuple[float, float]:
        """The absolute (degmin, degmax) that ``match.run`` sweeps for this ``lower`` block.

        ``match.scan`` rotates the upper lattice relative to the lower block's *frame*
        (its bottom layer). Twists are defined relative to the lower block's *top* layer,
        so the window is shifted by that layer's rotation, :func:`twist_offset`.
        For a monolayer the shift is 0; for a 22.62° bilayer a twist window of
        1–60.01° is swept as 23.62–82.63°, which is the window of
        ``trilayer_candidate_finder.ipynb`` (``degmin = degree + 1``).

        >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
        >>> SearchParams(degmin=1.0, degmax=60.01).window_for(root)
        (1.0, 60.01)
        """
        offset = twist_offset(lower)
        return self.degmin + offset, self.degmax + offset

    def estimated_cost(self) -> int:
        """Rough work estimate: number of (angle, lattice point) pairs tested.

        ``match.scan`` tests every lower-lattice point with both coefficients
        non-zero at every angle. Useful to compare rounds before running them.

        >>> SearchParams(degmin=1, degmax=2, degstep=0.01, nmin=-10, nmax=10).estimated_cost()
        36100
        """
        n_angles = len(np.arange(self.degmin, self.degmax, self.degstep))
        n_side = self.nmax - self.nmin - 1  # range(nmin, nmax) without the 0
        return int(n_angles * n_side * n_side)


@dataclass(frozen=True)
class Filters:
    """Thresholds deciding which candidates move on to the next round.

    Filters never delete anything: rejected candidates are still written to YAML
    and appear in ``index.csv`` with ``passed_filter = False``.

    - ``max_residual`` — max ``residual`` (Å) of the combination step itself.
    - ``max_cumulative_residual`` — max ``cumulative_residual`` (Å).
    - ``max_layer_residual`` — max of ``layer_residuals`` (Å): the cell's true error
      against every layer. Unlike ``cumulative_residual`` it sees errors that grow when
      an imperfect cell is multiplied up in later rounds (important for heterostructures).
    - ``max_length_mismatch`` — max | |a1| − |a2| | (Å); the notebooks used 0.1 to keep
      cells with equal-length vectors. ``None`` disables it.
    - ``max_strain_energy`` — meV/Å²; only applied where elastic constants are known.

    >>> Filters(max_residual=1e-3).passes(candidate)   # doctest: +SKIP
    True
    """

    max_residual: Optional[float] = None
    max_cumulative_residual: Optional[float] = None
    max_layer_residual: Optional[float] = None
    max_length_mismatch: Optional[float] = None
    max_strain_energy: Optional[float] = None

    def passes(self, record: MoireStructure) -> bool:
        """True if ``record`` satisfies every threshold that is set."""
        return not self.reasons_rejected(record)

    def reasons_rejected(self, record: MoireStructure) -> list[str]:
        """Human-readable list of the thresholds ``record`` violates (empty = passes)."""
        reasons = []
        if record.is_root:
            return reasons  # roots have no residual; they always pass
        if self.max_residual is not None and record.residual > self.max_residual:
            reasons.append(f"residual {record.residual:.3g} > {self.max_residual:.3g}")
        if (
            self.max_cumulative_residual is not None
            and record.cumulative_residual > self.max_cumulative_residual
        ):
            reasons.append(
                f"cumulative_residual {record.cumulative_residual:.3g} > {self.max_cumulative_residual:.3g}"
            )
        worst = record.worst_layer_residual
        if self.max_layer_residual is not None and worst is not None and worst > self.max_layer_residual:
            reasons.append(f"max layer residual {worst:.3g} > {self.max_layer_residual:.3g}")
        energy = record.strain_energy
        if self.max_strain_energy is not None and energy is not None and energy > self.max_strain_energy:
            reasons.append(f"strain energy {energy:.3g} meV/Å² > {self.max_strain_energy:.3g}")
        if self.max_length_mismatch is not None:
            a1, a2 = record.lattice_vectors
            mismatch = abs(np.linalg.norm(a1) - np.linalg.norm(a2))
            if mismatch > self.max_length_mismatch:
                reasons.append(f"length mismatch {mismatch:.3g} > {self.max_length_mismatch:.3g}")
        return reasons


# ===========================================================================
# 2. Raw matches -> candidate cells
# ===========================================================================
@dataclass
class CellMatch:
    """One commensurate cell as returned by ``match.filtermatches``, tidied up.

    ``lower_coeffs`` (rows n, n') and ``upper_coeffs`` (rows m, m') are the integer
    coordinates of the two cell vectors in the lower lattice and in the *rotated*
    upper lattice.
    """

    twist: float
    goodangle: float
    vectors: np.ndarray  # 2x2, rows a1, a2, in the lower block's frame
    lower_coeffs: np.ndarray  # 2x2 int
    upper_coeffs: np.ndarray  # 2x2 int
    residual: float  # max(delvec of a1, delvec of a2), in Å (after straining, if strained)
    lower_strain: Optional[np.ndarray] = None  # 2x2 strain (fraction) of a strained lower layer
    upper_strain: Optional[np.ndarray] = None  # 2x2 strain (fraction) of a strained upper layer

    @property
    def upper_lattice_key(self) -> tuple[int, ...]:
        """HNF of the upper coefficients: with large strain tolerances the same lower
        cell can meet different upper lattice points, which are different structures."""
        return tuple(hermite_normal_form(self.upper_coeffs).ravel().tolist())

    @property
    def strain(self) -> float:
        """Largest absolute principal strain of both layers (fraction; 0 = unstrained)."""
        return max(strain_magnitude(self.lower_strain), strain_magnitude(self.upper_strain))

    @property
    def index(self) -> int:
        """|det(lower_coeffs)|: cell area in units of the lower block's cell area."""
        return int(round(abs(np.linalg.det(self.lower_coeffs))))

    @property
    def lattice_key(self) -> tuple[int, ...]:
        """Basis-independent label of the lattice spanned by the cell (its HNF)."""
        return tuple(hermite_normal_form(self.lower_coeffs).ravel().tolist())

    def size_key(self) -> float:
        """|a1| + |a2|: prefer short, near-reduced bases among equivalent ones."""
        return float(np.linalg.norm(self.vectors, axis=1).sum())


def _row_to_parts(row: dict[str, Any]) -> tuple[np.ndarray, list[int], list[int], float]:
    vector = np.asarray(row["matchpoint"], dtype=float)
    return vector, [int(row["n1"]), int(row["n2"])], [int(row["m1"]), int(row["m2"])], float(row["delvec"])


def twist_offset(lower: MoireStructure) -> float:
    """Rotation (deg) of ``lower``'s top layer in ``lower``'s frame.

    ``absolute scan angle = twist_offset(lower) + twist``: the twist of the upper
    block is measured from the layer directly below it.
    """
    return float(lower.layer_rotations[-1])


def cell_from_pair(
    row_a: dict[str, Any], row_b: dict[str, Any], goodangle: float, offset: float = 0.0
) -> CellMatch:
    """Convert one ``filtermatches`` pair into a right-handed :class:`CellMatch`.

    ``match.filtermatches`` returns both (a, b) and (b, a); swapping to make
    ``a1 x a2 > 0`` maps both orderings onto the same right-handed cell, which is
    also what VASP prefers. ``offset`` (see :func:`twist_offset`) is subtracted from
    the scan angle so that ``CellMatch.twist`` is relative to the layer below.
    """
    v1, n1, m1, d1 = _row_to_parts(row_a)
    v2, n2, m2, d2 = _row_to_parts(row_b)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        v1, v2, n1, n2, m1, m2 = v2, v1, n2, n1, m2, m1
    return CellMatch(
        twist=float(row_a["angle"]) - offset,
        goodangle=float(goodangle),
        vectors=np.array([v1, v2]),
        lower_coeffs=np.array([n1, n2], dtype=int),
        upper_coeffs=np.array([m1, m2], dtype=int),
        residual=max(d1, d2),
    )


def cells_from_run(results: list[dict[str, Any]], offset: float = 0.0) -> list[CellMatch]:
    """Flatten the output of ``match.run_and_filter`` into :class:`CellMatch` objects."""
    cells = []
    for block in results:
        for row_a, row_b, _extra in block["results"]:
            cells.append(cell_from_pair(row_a, row_b, block["goodangle"], offset))
    return cells


def hermite_normal_form(matrix: np.ndarray) -> np.ndarray:
    """Row-style Hermite normal form of a non-singular 2x2 integer matrix.

    Two integer bases (rows = basis vectors) span the same lattice exactly when
    they have the same HNF, so the HNF is a canonical "name" for a lattice.
    The result is ``[[g, x], [0, h]]`` with ``g, h > 0`` and ``0 <= x < h``.

    >>> hermite_normal_form(np.array([[3, -2], [2, 3]]))
    array([[ 1,  8],
           [ 0, 13]])
    """
    (a, b), (c, d) = (int(v) for v in matrix[0]), (int(v) for v in matrix[1])
    g, u, v = _extended_gcd(a, c)  # u*a + v*c = g = gcd(a, c)
    if g == 0:
        raise ValueError(f"singular coefficient matrix {matrix.tolist()}")
    top = (g, u * b + v * d)
    bottom_second = (a * d - b * c) // g  # = det / g, exact
    h = abs(bottom_second)
    if h == 0:
        raise ValueError(f"singular coefficient matrix {matrix.tolist()}")
    x = top[1] % h
    return np.array([[g, x], [0, h]], dtype=int)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, u, v) with u*a + v*b = g = gcd(a, b) >= 0."""
    old_r, r = a, b
    old_u, u = 1, 0
    old_v, v = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_u, u = u, old_u - quotient * u
        old_v, v = v, old_v - quotient * v
    if old_r < 0:
        old_r, old_u, old_v = -old_r, -old_u, -old_v
    return old_r, old_u, old_v


def _best(cells: list[CellMatch]) -> CellMatch:
    """Deterministic choice among equivalent cells: lowest residual, then shortest."""
    return min(cells, key=lambda c: (c.residual, c.size_key(), c.vectors.round(6).ravel().tolist()))


def deduplicate(cells: list[CellMatch], keep_supercells: bool, cluster_tol: float) -> list[CellMatch]:
    """Collapse the many raw matches that describe the same structure.

    ``match.run_and_filter`` reports one cell many times. Three steps, each only
    removing *equivalent* descriptions (not physically different structures):

    1. Same twist, same good angle, same lattice (same HNF): sign flips, swapped
       vectors and alternative bases of one cell → keep one representative.
    2. Same twist and good angle, larger cell (integer multiple / supercell of a
       smaller cell found at the same twist) → dropped unless ``keep_supercells``.
    3. Same lattice at neighbouring grid twists (gap <= ``cluster_tol``): the same
       commensurate angle approximated by several grid points → keep the lowest
       residual one. (The notebooks did this with a 0.05° Theta clustering.)

    Returns the survivors sorted by twist.
    """
    # Step 1 + 2: work twist by twist.
    by_twist: dict[tuple[float, float], list[CellMatch]] = {}
    for cell in cells:
        by_twist.setdefault((cell.twist, cell.goodangle), []).append(cell)

    per_twist_survivors: list[CellMatch] = []
    for group in by_twist.values():
        if not keep_supercells:
            smallest = min(cell.index for cell in group)
            group = [cell for cell in group if cell.index == smallest]
        by_lattice: dict[tuple[int, ...], list[CellMatch]] = {}
        for cell in group:
            by_lattice.setdefault(cell.lattice_key, []).append(cell)
        per_twist_survivors.extend(_best(same) for same in by_lattice.values())

    # Step 3: neighbouring twists with the same lattice.
    by_lattice_all: dict[tuple[Any, ...], list[CellMatch]] = {}
    for cell in per_twist_survivors:
        by_lattice_all.setdefault((cell.goodangle, cell.lattice_key), []).append(cell)

    survivors: list[CellMatch] = []
    for group in by_lattice_all.values():
        for run in _split_into_angle_runs(group, cluster_tol):
            survivors.append(_best(run))
    return sorted(survivors, key=lambda c: (c.twist, c.goodangle, c.size_key()))


def lattice_rotation_order(lattice_vectors: np.ndarray, tol: float = 1e-4) -> int:
    """Largest n in (6, 4, 3, 2) such that a rotation by 360°/n maps the 2D lattice onto itself.

    A rotation maps the lattice onto itself if the rotated basis vectors have integer
    coordinates in the original basis. Every 2D lattice has n ≥ 2 (inversion).

    >>> lattice_rotation_order(np.eye(2)), lattice_rotation_order(np.array([[1.0, 0.0], [0.5, 3 ** 0.5 / 2]]))
    (4, 6)
    """
    basis = np.asarray(lattice_vectors, dtype=float).T  # columns
    for order in (6, 4, 3):
        rotated = rotation_matrix_2d(360.0 / order) @ basis
        coordinates = np.linalg.solve(basis, rotated)
        if np.allclose(coordinates, np.round(coordinates), atol=tol):
            return order
    return 2


def canonical_twist(twist: float, order: int, mirror: bool) -> float:
    """Representative of a twist modulo 360°/order (and modulo sign if ``mirror``).

    >>> canonical_twist(112.62, 4, False), canonical_twist(67.38, 4, True)
    (22.62, 22.62)
    """
    period = 360.0 / order
    reduced = twist % period
    if mirror:
        reduced = min(reduced, period - reduced)
    return round(reduced, 6) + 0.0


def merge_equivalent_twists(
    cells: list[CellMatch], upper: MoireStructure, mode: str, period_lattice: Optional[np.ndarray] = None
) -> list[CellMatch]:
    """Keep one cell per set of symmetry-equivalent twists (``mode`` = :data:`MERGE_MODES`).

    Convention: for an upper **single layer** whose lattice has an n-fold rotation, twists
    θ and θ + 360°/n give the same structure up to an in-plane translation of that layer
    (a different stacking registry); with ``"rotation+mirror"`` θ and −θ are also merged —
    mirror images (enantiomers), equal in energy but of opposite handedness. The lattice
    symmetry is used, not the atomic basis: if the basis breaks it, do not merge. Upper
    *blocks* are never merged (their symmetry can be lower than their cell's). Among
    equivalent cells of the same size the one with the smallest |twist| (then the lowest
    residual) is kept.
    """
    if mode == "none" or not upper.is_root or not cells:
        return cells
    order = lattice_rotation_order(upper.lattice_vectors if period_lattice is None else period_lattice)
    mirror = mode == "rotation+mirror"
    groups: dict[tuple[Any, ...], list[CellMatch]] = {}
    for cell in cells:
        key = (cell.goodangle, canonical_twist(cell.twist, order, mirror), round(abs(np.linalg.det(cell.vectors)), 4))
        groups.setdefault(key, []).append(cell)
    kept = [min(group, key=lambda c: (round(abs(c.twist), 6), c.residual)) for group in groups.values()]
    return sorted(kept, key=lambda c: (c.twist, c.goodangle, c.size_key()))


def _split_into_angle_runs(cells: list[CellMatch], cluster_tol: float) -> list[list[CellMatch]]:
    """Split cells into runs of consecutive twists (gaps <= ``cluster_tol``)."""
    ordered = sorted(cells, key=lambda c: c.twist)
    runs = [[ordered[0]]]
    for cell in ordered[1:]:
        if cell.twist - runs[-1][-1].twist <= cluster_tol + 1e-9:
            runs[-1].append(cell)
        else:
            runs.append([cell])
    return runs


# ===========================================================================
# 3. One pairwise search
# ===========================================================================
def rotation_matrix_2d(angle_deg: float) -> np.ndarray:
    """Anticlockwise 2x2 rotation by ``angle_deg`` (same convention as ``match.scan``)."""
    angle = np.deg2rad(angle_deg)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def layer_residuals(
    cell: np.ndarray,
    poscars: Sequence[str],
    rotations: Sequence[float],
    search_dirs: Sequence[PathLike] = (),
    strains: Optional[Sequence[np.ndarray]] = None,
) -> list[float]:
    """True commensurability error (Å) of ``cell`` with each primitive layer.

    For layer *i* with primitive vectors ``P_i``, strained by ``strains[i]`` (2x2,
    fractions, in the layer's own frame; ``None`` = unstrained) and rotated by
    ``rotations[i]``, each
    cell vector ``v`` is written in that layer's basis, rounded to integers and
    mapped back; the distance to ``v`` is the error. The result per layer is the
    worse of the two cell vectors.

    Unlike ``cumulative_residual`` (a max over the steps), this measures the
    final cell directly against every layer, so errors that grow when a
    cell is multiplied up in later rounds are included.
    """
    if strains is None:
        strains = [np.zeros((2, 2))] * len(poscars)
    residuals = []
    for poscar, rotation, strain in zip(poscars, rotations, strains):
        primitive = read_primitive(poscar, search_dirs)["in_plane"]
        # columns = strained, then rotated a, b
        basis = rotation_matrix_2d(rotation) @ (np.eye(2) + strain) @ primitive.T
        worst = 0.0
        for vector in cell:
            coords = np.linalg.solve(basis, vector)
            error = np.linalg.norm(basis @ np.round(coords) - vector)
            worst = max(worst, float(error))
        residuals.append(worst)
    return residuals


def _max_ignoring_none(*values: Optional[float]) -> float:
    return max(value for value in values if value is not None)


def _clean(strain: np.ndarray) -> np.ndarray:
    """Set float noise (|ε| < 1e-9, i.e. 1e-7 %) to exactly 0, so an unneeded strain reads as 0."""
    strain = np.array(strain, dtype=float)
    strain[np.abs(strain) < 1e-9] = 0.0
    return strain


def record_from_cell(
    lower: MoireStructure,
    upper: MoireStructure,
    cell: CellMatch,
    search_dirs: Sequence[PathLike] = (),
    layer_elastic: Optional[Sequence[Optional[Elastic]]] = None,
) -> MoireStructure:
    """Build the child record for ``lower`` + ``upper`` combined at ``cell``.

    Lists are concatenated lower-first. The upper block is rotated as a rigid unit:
    its bottom layer ends up at (rotation of the lower block's top layer + twist),
    and its other layers keep their rotations relative to that bottom layer.
    A strained parent (only possible for a level-0 layer) contributes its new strain;
    otherwise the parent's ``layer_strains`` are inherited unchanged.
    ``layer_elastic`` (one entry per layer of the child) gives ``strain_energy``.
    """
    upper_base = twist_offset(lower) + cell.twist
    rotations = list(lower.layer_rotations) + [upper_base + rotation for rotation in upper.layer_rotations]
    poscars = list(lower.source_poscar) + list(upper.source_poscar)
    n_layers = lower.n_layers + upper.n_layers
    lower_strains = lower.strain_tensors() if cell.lower_strain is None else [_clean(cell.lower_strain)]
    upper_strains = upper.strain_tensors() if cell.upper_strain is None else [_clean(cell.upper_strain)]
    strains = lower_strains + upper_strains
    return MoireStructure(
        level=n_layers - 1,
        n_layers=n_layers,
        material=list(lower.material) + list(upper.material),
        parents=[lower.id, upper.id],
        twist=cell.twist,
        lattice_vectors=cell.vectors,
        residual=cell.residual,
        cumulative_residual=_max_ignoring_none(cell.residual, lower.cumulative_residual, upper.cumulative_residual),
        source_poscar=poscars,
        units=lower.units,
        twist_sequence=list(lower.twist_sequence) + list(upper.twist_sequence) + [cell.twist],
        layer_rotations=rotations,
        layer_residuals=layer_residuals(cell.vectors, poscars, rotations, search_dirs, strains),
        coefficients={"lower": cell.lower_coeffs.tolist(), "upper": cell.upper_coeffs.tolist()},
        layer_strains=[(100.0 * strain).tolist() for strain in strains],  # stored in percent
        strain_energy=None if layer_elastic is None else strain_energy(strains, list(layer_elastic)),
    )


def strain_cells(
    cells: list[CellMatch],
    lower: MoireStructure,
    upper: MoireStructure,
    params: SearchParams,
    limits: tuple[float, float],
    elastic: tuple[Optional[Elastic], Optional[Elastic]] = (None, None),
) -> list[CellMatch]:
    """Turn near-coincidences into exactly commensurate, strained cells.

    ``limits`` are the strain limits (fractions) of the lower and the upper block.

    1. Group the raw matches by (good angle, lower lattice, upper lattice) and keep
       one per run of neighbouring grid angles (lowest unstrained residual).
    2. Solve for the strain of each (:func:`MLM.strain.fit_strain`); drop it if a
       layer would exceed its limit — such cells are not commensurate within the
       job's rules, so they are not candidates at all.
       The fit also corrects the twist; a cell whose exact twist falls outside the
       requested window ``[degmin, degmax)`` is dropped as well, and so is a cell that
       the allowed strain cannot close (residual > ``strain_residual_tol``).
    3. The same structure reached from several runs: keep the smallest strain.
    4. Unless ``keep_supercells``, keep only the smallest cell at each exact twist.
    """
    offset = twist_offset(lower)
    groups: dict[tuple[Any, ...], list[CellMatch]] = {}
    for cell in cells:
        groups.setdefault((cell.goodangle, cell.lattice_key, cell.upper_lattice_key), []).append(cell)

    best_per_structure: dict[tuple[Any, ...], CellMatch] = {}
    for key, group in groups.items():
        for run in _split_into_angle_runs(group, params.cluster_tol):
            start = _best(run)
            fit = fit_strain(start.vectors, upper.lattice_vectors, start.upper_coeffs,
                             offset + start.twist, params.strain_mode, *limits, *elastic)
            twist = start.twist + fit.twist_correction
            closed = fit.residual <= params.strain_residual_tol
            if not closed or not fit.within(*limits) or not params.degmin <= twist < params.degmax:
                continue
            strained = replace(start, twist=twist, vectors=fit.cell,
                               residual=fit.residual, lower_strain=fit.lower_strain, upper_strain=fit.upper_strain)
            current = best_per_structure.get(key)
            if current is None or strained.strain < current.strain:
                best_per_structure[key] = strained

    survivors = list(best_per_structure.values())
    if not params.keep_supercells:
        by_twist: dict[tuple[float, float], list[CellMatch]] = {}
        for cell in survivors:
            by_twist.setdefault((round(cell.twist, 6), cell.goodangle), []).append(cell)
        survivors = []
        for group in by_twist.values():
            smallest = min(cell.index for cell in group)
            survivors.extend(cell for cell in group if cell.index == smallest)
    return sorted(survivors, key=lambda c: (c.twist, c.goodangle, c.strain, c.size_key()))


def find_candidates(
    lower: MoireStructure,
    upper: MoireStructure,
    params: SearchParams,
    search_dirs: Sequence[PathLike] = (),
    strain_limits: tuple[float, float] = (0.0, 0.0),
    layer_elastic: Optional[Sequence[Optional[Elastic]]] = None,
) -> list[MoireStructure]:
    """All commensurate cells of ``lower`` + twisted ``upper`` (no residual filters).

    The only selection applied is the good-angle match inside
    ``match.run_and_filter`` (plus de-duplication if ``params.dedupe``).

    ``strain_limits`` = (lower, upper) maximum strain in **percent** (0 = rigid). Only
    level-0 layers can be strained (blocks built in earlier rounds stay rigid). With a
    limit, the scan tolerance is widened so that near-coincidences needing up to that
    strain are found, and each is then strained to an exact match (:func:`strain_cells`).

    ``layer_elastic``: optional :class:`~MLM.strain.Elastic` constants, one per layer of the
    combined stack (lower's layers, then upper's). They make the strain sharing minimise
    the elastic energy and fill in ``strain_energy`` (meV/Å²).

    >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    >>> params = SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    >>> [round(c.twist, 2) for c in find_candidates(root, root, params)]   # doctest: +SKIP
    [22.62]
    """
    if lower.units != upper.units:
        raise ValueError(f"unit mismatch: {lower.units} vs {upper.units}")
    limits = (strain_limits[0] / 100.0, strain_limits[1] / 100.0)  # percent -> fraction
    for block, limit in ((lower, limits[0]), (upper, limits[1])):
        if limit < 0:
            raise ValueError(f"strain limits must be >= 0, got {strain_limits}")
        if limit > 0 and not block.is_root:
            raise ValueError(f"only level-0 layers can be strained; {block.id} is a {block.n_layers}-layer block")
    strained = limits[0] > 0 or limits[1] > 0

    degmin, degmax = params.window_for(lower)
    a1, a2 = lower.lattice_vectors
    g1, g2 = upper.lattice_vectors
    tolerance, relative = params.dtol, 0.0
    if strained and params.scanner == "match":
        # match.scan only knows one global tolerance: widen it to the largest |m| searched.
        tolerance = scan_tolerance(params.dtol, sum(limits), max(abs(params.nmin), abs(params.nmax)),
                                   lower.lattice_vectors, upper.lattice_vectors, params.degstep)
    elif strained:
        # A strain ε shifts the upper coordinates by ≈ ε·|m|; the grid adds up to Δθ/2 rad·|m|.
        # A tolerance proportional to |m| (with a 25 % margin) keeps the per-angle match list
        # free of junk, so the real candidates are not cut by the 25-per-angle cap.
        relative = 1.25 * (sum(limits) + np.deg2rad(params.degstep) / 2)
    arguments = dict(degmin=degmin, degmax=degmax, degstep=params.degstep, dtol=tolerance,
                     nmin=params.nmin, nmax=params.nmax, goodangles=list(params.goodangles))
    if params.scanner == "match":
        results = run_and_filter(a1, a2, g1, g2, **arguments)
    else:
        cap = params.max_matches_per_angle
        if cap == "auto":
            cap = STRAINED_MATCHES_PER_ANGLE if strained else 25
        results = fastscan.run_and_filter(a1, a2, g1, g2, include_axes=params.include_axes,
                                          max_per_angle=cap, relative_tol=relative, **arguments)
    cells = cells_from_run(results, offset=twist_offset(lower))
    if layer_elastic is not None and len(layer_elastic) != lower.n_layers + upper.n_layers:
        raise ValueError(f"layer_elastic needs {lower.n_layers + upper.n_layers} entries, got {len(layer_elastic)}")
    roots_elastic = (None, None)
    if layer_elastic is not None:
        roots_elastic = (layer_elastic[0] if lower.is_root else None,
                         layer_elastic[lower.n_layers] if upper.is_root else None)
    if strained and cells:
        cells = strain_cells(cells, lower, upper, params, limits, roots_elastic)
    elif params.dedupe and cells:
        cells = deduplicate(cells, params.keep_supercells, params.cluster_tol)
    cells = merge_equivalent_twists(cells, upper, params.merge_equivalent_twists)
    return [record_from_cell(lower, upper, cell, search_dirs, layer_elastic) for cell in cells]


def search_pair(
    lower: MoireStructure,
    upper: MoireStructure,
    params: SearchParams,
    filters: Optional[Filters] = None,
    search_dirs: Sequence[PathLike] = (),
    strain_limits: tuple[float, float] = (0.0, 0.0),
    layer_elastic: Optional[Sequence[Optional[Elastic]]] = None,
) -> list[MoireStructure]:
    """Commensurate cells of ``lower`` + twisted ``upper`` that pass ``filters``.

    This is the atomic operation of the whole N-layer search. ``upper`` is rotated
    rigidly by the swept twist, measured from ``lower``'s top layer (the layer
    directly below ``upper``'s bottom layer). Each returned record
    has ``parents = [lower.id, upper.id]``.

    >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    >>> params = SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    >>> bilayers = search_pair(root, root, params, Filters(max_residual=1e-3))   # doctest: +SKIP
    >>> bilayers[0].twist, bilayers[0].residual                                  # doctest: +SKIP
    (22.620000000000097, 3.39e-05)
    """
    filters = filters or Filters()
    candidates = find_candidates(lower, upper, params, search_dirs, strain_limits, layer_elastic)
    return [candidate for candidate in candidates if filters.passes(candidate)]


# ===========================================================================
# 4. Pairing planner (pure functions, no I/O)
# ===========================================================================
#: A block of consecutive layers, as a half-open range [start, stop) of layer indices.
Span = tuple[int, int]


@dataclass(frozen=True)
class Step:
    """One entry of a round: combine ``lower`` and ``upper``, or carry ``lower`` forward.

    >>> Step((0, 2), (2, 3)).result
    (0, 3)
    >>> Step((2, 3), None).is_carry
    True
    """

    lower: Span
    upper: Optional[Span] = None

    @property
    def is_carry(self) -> bool:
        """True if this block has no partner in this round (odd leftover)."""
        return self.upper is None

    @property
    def result(self) -> Span:
        """The block this step produces."""
        return self.lower if self.upper is None else (self.lower[0], self.upper[1])

    @property
    def interface(self) -> Optional[int]:
        """Which pair of adjacent layers this step joins: ``k`` = layer k | layer k+1 (1-based).

        Because a twist is measured from the layer directly below, every merge sets
        exactly one interface: the lower block's top layer against the upper block's
        bottom layer. Each of the N-1 interfaces is set by exactly one step.

        >>> Step((0, 2), (2, 3)).interface   # L1-2 + L3: layer 2 | layer 3
        2
        >>> Step((2, 3), None).interface is None
        True
        """
        return None if self.upper is None else self.upper[0]

    def __str__(self) -> str:
        def name(span: Span) -> str:
            first, last = span[0] + 1, span[1]  # 1-based layer numbers for humans
            return f"L{first}" if first == last else f"L{first}-{last}"

        if self.is_carry:
            return f"carry {name(self.lower)}"
        return f"{name(self.lower)} + {name(self.upper)} -> {name(self.result)}"


def plan_rounds(n_layers: int) -> list[list[Step]]:
    """Tournament-tree plan: pair adjacent blocks each round, carry an odd leftover.

    >>> [[str(step) for step in round_] for round_ in plan_rounds(3)]
    [['L1 + L2 -> L1-2', 'carry L3'], ['L1-2 + L3 -> L1-3']]
    >>> len(plan_rounds(1)), len(plan_rounds(2)), len(plan_rounds(5))
    (0, 1, 3)
    """
    if n_layers < 1:
        raise ValueError(f"need at least one layer, got {n_layers}")
    blocks: list[Span] = [(i, i + 1) for i in range(n_layers)]
    rounds: list[list[Step]] = []
    while len(blocks) > 1:
        steps = []
        for position in range(0, len(blocks) - 1, 2):
            steps.append(Step(blocks[position], blocks[position + 1]))
        if len(blocks) % 2 == 1:
            steps.append(Step(blocks[-1], None))
        rounds.append(steps)
        blocks = [step.result for step in steps]
    return rounds


def format_plan(plan: list[list[Step]]) -> str:
    """Multi-line description of a plan, one round per line."""
    lines = []
    for number, steps in enumerate(plan, start=1):
        lines.append(f"round {number}: " + ", ".join(str(step) for step in steps))
    return "\n".join(lines) if lines else "(single layer: nothing to combine)"


# ===========================================================================
# 5. Per-round parameters and the tree driver
# ===========================================================================
@dataclass
class SearchSchedule:
    """Search parameters: a ``default``, overrides per round, and overrides per interface.

    Overrides are applied in this order (later wins):

    1. ``default``
    2. ``rounds[r]`` — e.g. a larger n range for rounds >= 2, where the lower "lattice"
       is a moiré cell;
    3. ``interfaces[k]`` — settings for the twist between layer k and layer k+1
       (1-based), typically the angle window ``degmin``/``degmax``.

    >>> schedule = SearchSchedule(SearchParams(nmin=-10, nmax=10), {2: {"nmin": -40, "nmax": 40}},
    ...                           interfaces={2: {"degmin": 36.0, "degmax": 38.0}})
    >>> schedule.params_for_round(1).nmax, schedule.params_for_round(2).nmax
    (10, 40)
    >>> p = schedule.params_for_round(2, interface=2)
    >>> p.nmax, p.degmin, p.degmax
    (40, 36.0, 38.0)
    """

    default: SearchParams = field(default_factory=SearchParams)
    rounds: dict[int, dict[str, Any]] = field(default_factory=dict)
    interfaces: dict[int, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rounds = {int(number): dict(overrides) for number, overrides in self.rounds.items()}
        self.interfaces = {int(number): dict(overrides) for number, overrides in self.interfaces.items()}
        # Validate every override early, so that a typo fails before any searching.
        for number, overrides in self.rounds.items():
            self._apply(self.default, overrides, f"round {number}")
        for number, overrides in self.interfaces.items():
            if number < 1:
                raise ValueError(f"interface numbers start at 1 (layer 1 | layer 2), got {number}")
            self._apply(self.default, overrides, f"interface {number}")

    @staticmethod
    def _apply(params: SearchParams, overrides: dict[str, Any], where: str) -> SearchParams:
        unknown = set(overrides) - set(SearchParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown search parameter(s) for {where}: {sorted(unknown)}")
        return replace(params, **overrides)

    def params_for_round(self, round_number: int, interface: Optional[int] = None) -> SearchParams:
        """``default`` + this round's overrides (+ this interface's overrides, if given)."""
        params = self._apply(self.default, self.rounds.get(round_number, {}), f"round {round_number}")
        if interface is not None:
            params = self._apply(params, self.interfaces.get(interface, {}), f"interface {interface}")
        return params

    def params_for_step(self, round_number: int, step: Step) -> SearchParams:
        """Parameters for one step of the plan (round + the interface it joins)."""
        return self.params_for_round(round_number, step.interface)

    @classmethod
    def notebook_defaults(cls, goodangles: Sequence[float] = (90.0,), max_round: int = 10) -> "SearchSchedule":
        """The settings of the notebooks: round 1 at n = ±10, every later round at ±40."""
        default = SearchParams(degmin=1.0, degmax=45.0, degstep=0.01, dtol=5e-4, nmin=-10, nmax=10,
                               goodangles=tuple(goodangles))
        later = {number: {"nmin": -40, "nmax": 40} for number in range(2, max_round + 1)}
        return cls(default, later)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in the stack-spec layout (``default`` + ``rounds`` + ``interfaces``)."""
        default = asdict(self.default)
        default["goodangles"] = list(default["goodangles"])
        return {
            "default": default,
            "rounds": {number: dict(o) for number, o in self.rounds.items()},
            "interfaces": {number: dict(o) for number, o in self.interfaces.items()},
        }


@dataclass
class StepReport:
    """What happened (or, in a dry run, what would happen) for one step."""

    round_number: int
    step: Step
    n_lower: Optional[int]  # candidates of the lower block going into this step
    n_upper: Optional[int]
    n_found: Optional[int] = None  # candidates produced (all, incl. rejected)
    n_passed: Optional[int] = None  # of which passed the filters
    seconds: Optional[float] = None
    cost_per_search: Optional[int] = None

    @property
    def n_searches(self) -> Optional[int]:
        """Number of ``search_pair`` calls = product of the two candidate counts."""
        if self.step.is_carry:
            return 0
        if self.n_lower is None or self.n_upper is None:
            return None
        return self.n_lower * self.n_upper

    def __str__(self) -> str:
        if self.step.is_carry:
            return f"round {self.round_number}: {self.step} ({self.n_lower} candidates)"
        def count(value: Optional[int]) -> str:
            return "?" if value is None else str(value)

        if self.n_searches is None:
            searches = "? searches (known once the previous round has run)"
        else:
            searches = f"{self.n_searches} searches"
        text = f"round {self.round_number}: {self.step}: {count(self.n_lower)} x {count(self.n_upper)} = {searches}"
        if self.n_found is not None:
            text += f" -> {self.n_found} candidates, {self.n_passed} passed filters ({self.seconds:.1f} s)"
        return text


@dataclass
class StackResult:
    """Outcome of :func:`build_stack`.

    ``final`` holds the full-stack candidates that passed the filters; every
    candidate (passing or not, all levels) is on disk under ``run_dir``.
    """

    run_dir: Path
    plan: list[list[Step]]
    reports: list[StepReport]
    final: list[MoireStructure]
    dry_run: bool = False
    status: str = "finished"  # "finished", or "stopped" if should_stop() ended the run early

    def summary(self) -> str:
        """Readable multi-line summary (plan + per-step counts)."""
        header = "DRY RUN — nothing searched" if self.dry_run else f"run directory: {self.run_dir}"
        lines = [header, format_plan(self.plan)] + [str(report) for report in self.reports]
        if not self.dry_run:
            lines.append(f"full-stack candidates passing filters: {len(self.final)}")
        return "\n".join(lines)


def job_to_spec_dict(
    name: str,
    layers: Sequence[MoireStructure],
    schedule: SearchSchedule,
    filters: Filters,
    output_dir: PathLike,
    strain_limits: Optional[Sequence[float]] = None,
    elastic: Optional[Sequence[Optional[Elastic]]] = None,
) -> dict[str, Any]:
    """The stack-spec dict equivalent to a ``build_stack`` call (for ``stack_spec.yaml``).

    A layer's ``strain`` (percent) is written only when it is non-zero.
    """
    limits = strain_limits if strain_limits is not None else [0.0] * len(layers)
    layer_entries = []
    constants = list(elastic) if elastic is not None else [None] * len(layers)
    for layer, limit, layer_constants in zip(layers, limits, constants):
        entry = {"poscar": layer.source_poscar[0], "material": layer.material[0]}
        if limit:
            entry["strain"] = float(limit)
        if layer_constants is not None:
            entry["young"] = float(layer_constants.young)
            entry["poisson"] = float(layer_constants.poisson)
        layer_entries.append(entry)
    return {
        "name": name,
        "output_dir": str(output_dir),
        "layers": layer_entries,
        "search": schedule.to_dict(),
        "filters": asdict(filters),
    }


def build_stack(
    layers: Sequence[MoireStructure],
    schedule: Optional[SearchSchedule] = None,
    filters: Optional[Filters] = None,
    run_dir: PathLike = "runs/stack",
    dry_run: bool = False,
    max_candidates: Optional[int] = None,
    search_dirs: Sequence[PathLike] = (),
    spec: Optional[dict[str, Any]] = None,
    log: Optional[Callable[[str], None]] = print,
    strain_limits: Optional[Sequence[float]] = None,
    workers: Optional[int] = 1,
    resume: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    index_format: str = "csv",
    elastic: Optional[Sequence[Any]] = None,
) -> StackResult:
    """Build every commensurate N-layer cell by a tournament-tree reduction.

    ``layers`` are level-0 records in stacking order (bottom first). Round 1
    pairs adjacent layers (1,2), (3,4), …; later rounds pair adjacent blocks,
    using each block's moiré cell as the effective lattice; an odd leftover is
    carried forward. *Every* candidate that passes ``filters`` goes on to the next
    round (no top-K cap), and every candidate — rejected ones included — is
    written to ``<run_dir>/yamls`` and listed in ``<run_dir>/index.csv``.

    - ``dry_run=True`` returns the plan and the number of searches known up front
      without searching or writing anything.
    - ``max_candidates`` is a deliberately unimplemented hook for a future top-K
      cap (DESIGN.md §10); passing a value raises ``NotImplementedError``.
    - ``spec`` is written as ``stack_spec.yaml``; by default one is generated from
      the arguments. Pass ``spec={}`` to skip writing it.
    - ``strain_limits``: maximum strain of each layer in **percent** (one value per
      layer, 0 = rigid; default: all rigid). A layer is strained only in the first
      search it takes part in, i.e. while it is still a single layer; blocks built in
      earlier rounds stay rigid. The strain mode comes from the search parameters.
    - ``workers``: processes for the searches of one step (they are independent);
      ``None`` or 0 = all CPU cores. Results are identical to ``workers=1``. Works in
      notebooks; in a script, call ``build_stack`` under ``if __name__ == "__main__":``.
    - ``resume=True`` continues an interrupted run in ``run_dir``: steps recorded in
      ``progress.yaml`` are not searched again. Refused if the job changed.
    - ``should_stop``: called before every search; returning True stops the run cleanly
      (finished steps are kept, ``status = "stopped"``; resume later).
    - ``progress_callback``: called after every search with
      ``{"round", "step", "done", "total"}``.
    - ``elastic``: optional per-layer elastic constants (:class:`~MLM.strain.Elastic`,
      a dict ``{"young": N/m, "poisson": ν}``, or ``None``). With constants for both layers
      of a strained pair the strain is shared to minimise elastic energy, and every
      structure whose layers all have constants gets ``strain_energy`` (meV/Å²).
    - ``index_format``: ``"csv"`` (default, human-readable) or ``"sqlite"`` (for very
      large runs; ``find`` then filters in SQL).
    - Writes ``run_info.yaml`` (versions, git commit, timings; see :mod:`MLM.runinfo`).

    >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    >>> schedule = SearchSchedule(SearchParams(degmin=22, degmax=23), {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -40, "nmax": 40}})
    >>> result = build_stack([root] * 3, schedule, Filters(max_residual=1e-3), run_dir="runs/demo")   # doctest: +SKIP
    >>> print(result.summary())                                                                     # doctest: +SKIP
    """
    if max_candidates is not None:
        raise NotImplementedError(
            "max_candidates (a top-K cap per block) is intentionally not implemented yet: "
            "the current design explores the full landscape (DESIGN.md §8 #10, §10)."
        )
    schedule = schedule or SearchSchedule.notebook_defaults()
    filters = filters or Filters()
    say = log or (lambda message: None)
    for layer in layers:
        if not layer.is_root:
            raise ValueError(f"build_stack expects level-0 layers, got {layer!r}")

    strain_limits = _check_strain_limits(strain_limits, len(layers))
    elastic = _check_elastic(elastic, len(layers))
    plan = plan_rounds(len(layers))
    if dry_run:
        return _dry_run(layers, plan, schedule, run_dir, say)

    run_dir = Path(run_dir)
    store = RunStore(run_dir, index_format=index_format)
    job = job_to_spec_dict(run_dir.name, layers, schedule, filters, run_dir, strain_limits, elastic)
    progress = _Progress(run_dir, job)
    if resume:
        progress.load_for_resume()
        store.load_existing()
        say(f"resuming: {len(progress.completed)} step(s) already done")
    elif progress.path.exists():
        say(f"note: {run_dir} already holds a run; starting a new one (resume=True continues it)")
    if spec is None:
        spec = job
    if spec:
        with open(store.run_dir / SPEC_FILENAME, "w", encoding="utf-8") as handle:
            yaml.safe_dump(spec, handle, sort_keys=False)
    info = RunInfo(run_dir)
    info.start({"workers": _resolve_workers(workers), "resumed": bool(resume)})
    progress.write()

    # Round 0: the roots. ``blocks`` maps a span to its candidates that passed.
    blocks: dict[Span, list[MoireStructure]] = {}
    for position, layer in enumerate(layers):
        store.add(layer, round_number=0, passed_filter=True)
        blocks[(position, position + 1)] = [layer]
    store.write_index()
    say(format_plan(plan))

    reports: list[StepReport] = []
    stopped = False
    for round_number, steps in enumerate(plan, start=1):
        new_blocks: dict[Span, list[MoireStructure]] = {}
        for step in steps:
            params = schedule.params_for_step(round_number, step)
            report = StepReport(round_number, step, len(blocks[step.lower]),
                                None if step.is_carry else len(blocks[step.upper]),
                                cost_per_search=None if step.is_carry else params.estimated_cost())
            if step.is_carry:
                new_blocks[step.result] = blocks[step.lower]
                say(str(report))
                reports.append(report)
                continue
            done = progress.lookup(round_number, step)
            if done is not None:  # resume: this step finished in an earlier session
                new_blocks[step.result] = [get(run_dir, structure_id) for structure_id in done["passed"]]
                report.n_found, report.n_passed, report.seconds = done["found"], len(done["passed"]), done["seconds"]
                say(f"{report} [from the earlier session]")
                reports.append(report)
                continue
            limits = (_block_strain_limit(step.lower, strain_limits), _block_strain_limit(step.upper, strain_limits))
            strain_note = f", strain ≤ {limits[0]:g}% / {limits[1]:g}% ({params.strain_mode})" if any(limits) else ""
            say(f"round {round_number}: {step}: {report.n_searches} searches, twist window "
                f"{params.degmin:g}–{params.degmax:g}°{strain_note} "
                f"(~{params.estimated_cost():.2e} lattice-point tests each)")
            started = time.perf_counter()
            outcome = _run_step(blocks[step.lower], blocks[step.upper], params, filters, store, round_number,
                                search_dirs, limits, workers, should_stop,
                                lambda done_, total_, r=round_number, s=step: _notify(progress_callback, r, s, done_, total_),
                                _elastic_for(step, elastic))
            store.write_index()  # keep the index current after every step
            if outcome is None:  # stopped by the caller
                stopped = True
                break
            passed, n_found = outcome
            report.n_found, report.n_passed = n_found, len(passed)
            report.seconds = time.perf_counter() - started
            new_blocks[step.result] = passed
            progress.mark_done(round_number, step, [record.id for record in passed], n_found, report.seconds)
            info.add_step({"round": round_number, "step": str(step), "searches": report.n_searches,
                           "found": n_found, "passed": len(passed), "seconds": round(report.seconds, 3)})
            say(str(report))
            reports.append(report)
        if stopped:
            break
        blocks = new_blocks

    if stopped:
        say("stopped; run again with resume=True to continue")
        info.finish("stopped")
        return StackResult(run_dir, plan, reports, [], dry_run=False, status="stopped")
    final = blocks[(0, len(layers))]
    info.finish("finished", final_candidates=len(final))
    return StackResult(run_dir, plan, reports, final, dry_run=False)


def _notify(callback: Optional[Callable[[dict[str, Any]], None]], round_number: int, step: Step,
            done: int, total: int) -> None:
    """Forward search progress to the caller (e.g. a GUI progress bar)."""
    if callback is not None:
        callback({"round": round_number, "step": str(step), "done": done, "total": total})


def _resolve_workers(workers: Optional[int]) -> int:
    """``None``/0 → all CPU cores; otherwise the given number (at least 1)."""
    if not workers:
        return os.cpu_count() or 1
    return max(1, int(workers))


#: Name of the file that records finished steps (used by ``resume=True``).
PROGRESS_FILENAME = "progress.yaml"


class _Progress:
    """Finished steps of a run, so that an interrupted run can be resumed.

    ``progress.yaml`` holds the job (to refuse resuming a *different* job in the same
    directory) and, for every finished step, the ids that passed the filters.
    """

    def __init__(self, run_dir: Path, job: dict[str, Any]) -> None:
        self.path = Path(run_dir) / PROGRESS_FILENAME
        self.job = _comparable_job(job)
        self.completed: list[dict[str, Any]] = []

    def load_for_resume(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"cannot resume: no {PROGRESS_FILENAME} in {self.path.parent}")
        with open(self.path, "r", encoding="utf-8") as handle:
            saved = yaml.safe_load(handle) or {}
        if _comparable_job(saved.get("job", {})) != self.job:
            raise ValueError(f"cannot resume: {self.path.parent} holds a different job "
                             "(layers, search parameters or filters changed); use another run_dir")
        self.completed = list(saved.get("completed", []))

    def lookup(self, round_number: int, step: Step) -> Optional[dict[str, Any]]:
        for entry in self.completed:
            if entry["round"] == round_number and entry["step"] == str(step):
                return entry
        return None

    def mark_done(self, round_number: int, step: Step, passed_ids: list[str], found: int, seconds: float) -> None:
        self.completed.append({"round": round_number, "step": str(step), "passed": passed_ids,
                               "found": found, "seconds": round(seconds, 3)})
        self.write()

    def write(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            yaml.safe_dump({"job": self.job, "completed": self.completed}, handle, sort_keys=False)


def _comparable_job(job: dict[str, Any]) -> dict[str, Any]:
    """The parts of a job that decide its results (not its name or output directory)."""
    return yaml.safe_load(yaml.safe_dump({key: value for key, value in job.items()
                                          if key not in ("name", "output_dir")}))


def _check_elastic(elastic: Optional[Sequence[Any]], n_layers: int) -> Optional[list[Optional[Elastic]]]:
    """Per-layer elastic constants as :class:`Elastic` objects (``None`` = not given)."""
    if elastic is None:
        return None
    values = list(elastic)
    if len(values) != n_layers:
        raise ValueError(f"elastic needs one entry per layer ({n_layers}), got {len(values)}")
    return [value if value is None or isinstance(value, Elastic) else Elastic(**value) for value in values]


def _elastic_for(step: Step, elastic: Optional[list[Optional[Elastic]]]) -> Optional[list[Optional[Elastic]]]:
    """Elastic constants of the layers of the block a step produces."""
    if elastic is None:
        return None
    start, stop = step.result
    return elastic[start:stop]


def _check_strain_limits(strain_limits: Optional[Sequence[float]], n_layers: int) -> list[float]:
    """Validate per-layer strain limits (percent); ``None`` means all layers rigid."""
    if strain_limits is None:
        return [0.0] * n_layers
    limits = [float(limit) for limit in strain_limits]
    if len(limits) != n_layers:
        raise ValueError(f"strain_limits needs one value per layer ({n_layers}), got {len(limits)}")
    if any(limit < 0 for limit in limits):
        raise ValueError(f"strain limits must be >= 0 (percent), got {limits}")
    return limits


def _block_strain_limit(span: Span, strain_limits: Sequence[float]) -> float:
    """Strain limit (percent) of a block: its layer's limit if it is a single layer, else 0."""
    first, stop = span
    return strain_limits[first] if stop - first == 1 else 0.0


def _search_task(task: tuple[Any, ...]) -> list[MoireStructure]:
    """One search, as a top-level function so that worker processes can run it."""
    lower, upper, params, search_dirs, strain_limits, layer_elastic = task
    return find_candidates(lower, upper, params, search_dirs, strain_limits, layer_elastic)


def _run_step(
    lowers: list[MoireStructure],
    uppers: list[MoireStructure],
    params: SearchParams,
    filters: Filters,
    store: RunStore,
    round_number: int,
    search_dirs: Sequence[PathLike],
    strain_limits: tuple[float, float] = (0.0, 0.0),
    workers: Optional[int] = 1,
    should_stop: Optional[Callable[[], bool]] = None,
    on_search_done: Optional[Callable[[int, int], None]] = None,
    layer_elastic: Optional[Sequence[Optional[Elastic]]] = None,
) -> Optional[tuple[list[MoireStructure], int]]:
    """All lower x upper searches of one step; store everything, return the passing ones.

    Searches run in ``workers`` processes when there are several; candidates are always
    stored in the same (lower, upper) order, so results do not depend on ``workers``.
    Returns ``None`` if ``should_stop()`` ended the step early.
    """
    tasks = [(lower, upper, params, tuple(search_dirs), strain_limits, layer_elastic)
             for lower in lowers for upper in uppers]
    n_workers = min(_resolve_workers(workers), len(tasks))

    def results():
        if n_workers <= 1:
            for task in tasks:
                if should_stop is not None and should_stop():
                    return
                yield _search_task(task)
            return
        # "spawn", not "fork": forking after polars/numpy have started their thread pools
        # can deadlock. Spawned workers import MLM afresh (~1 s start-up each).
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_search_task, task) for task in tasks]
            for future in futures:  # in submission order: deterministic output
                if should_stop is not None and should_stop():
                    for pending in futures:
                        pending.cancel()
                    return
                try:
                    yield future.result()
                except BrokenProcessPool as error:
                    raise RuntimeError(
                        "parallel search workers could not start. In a Python script, put the "
                        "code that calls build_stack under `if __name__ == \"__main__\":` (needed by "
                        "multiprocessing); scripts piped through stdin cannot use workers. "
                        "Or use workers=1."
                    ) from error

    passed: dict[str, MoireStructure] = {}  # keyed by id: identical recipes count once
    n_found = 0
    n_done = 0
    for candidates in results():
        for candidate in candidates:
            ok = filters.passes(candidate)
            store.add(candidate, round_number=round_number, passed_filter=ok)
            n_found += 1
            if ok:
                passed.setdefault(candidate.id, candidate)
        n_done += 1
        if on_search_done is not None:
            on_search_done(n_done, len(tasks))
    if n_done < len(tasks):
        return None
    return list(passed.values()), n_found


def _dry_run(
    layers: Sequence[MoireStructure],
    plan: list[list[Step]],
    schedule: SearchSchedule,
    run_dir: PathLike,
    say: Callable[[str], None],
) -> StackResult:
    """Describe the work without searching: only round-1 counts are known (1 x 1)."""
    reports = []
    known: dict[Span, Optional[int]] = {(i, i + 1): 1 for i in range(len(layers))}
    for round_number, steps in enumerate(plan, start=1):
        next_known: dict[Span, Optional[int]] = {}
        for step in steps:
            params = schedule.params_for_step(round_number, step)
            report = StepReport(round_number, step, known[step.lower],
                                None if step.is_carry else known[step.upper],
                                cost_per_search=None if step.is_carry else params.estimated_cost())
            # After a search the candidate count is unknown; a carried block keeps its count.
            next_known[step.result] = known[step.lower] if step.is_carry else None
            reports.append(report)
        known = next_known
    result = StackResult(Path(run_dir), plan, reports, final=[], dry_run=True)
    say(result.summary())
    return result
