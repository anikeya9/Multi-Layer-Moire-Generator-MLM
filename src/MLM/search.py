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

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import yaml

from MLM.index import SPEC_FILENAME, RunStore
from MLM.match import run_and_filter
from MLM.structure import MoireStructure, PathLike, read_primitive

# ===========================================================================
# 1. Parameters and filters
# ===========================================================================

#: Allowed values of ``SearchParams.window_relative_to``.
WINDOW_MODES = ("frame", "lower_top_layer")


@dataclass(frozen=True)
class SearchParams:
    """Parameters of one pairwise search (one call of ``match.run_and_filter``).

    The first seven fields are passed straight to ``run_and_filter``:

    - ``degmin``, ``degmax``, ``degstep`` — twist sweep in degrees (``degmax`` excluded,
      as in ``np.arange``).
    - ``dtol`` — tolerance on the *fractional* coordinates of a match (``match.scan``'s
      ``tol``); the Å residual of each match is reported separately as ``delvec``.
    - ``nmin``, ``nmax`` — range of integer coefficients searched on the lower lattice.
    - ``goodangles`` — target angles (deg) between the two moiré vectors.

    The remaining fields control how the raw matches are post-processed here:

    - ``window_relative_to`` — ``"frame"`` (default): the window is the twist of the
      upper block relative to the lower block's frame. ``"lower_top_layer"``: the
      window is shifted by the absolute rotation of the lower block's top layer,
      which reproduces the trilayer notebook (``degmin = degree + 1``).
    - ``dedupe`` — merge matches that describe the same cell (see :func:`deduplicate`).
    - ``keep_supercells`` — keep non-minimal cells (integer multiples) found at the
      same twist. ``False`` keeps only the smallest cell, like the notebooks did.
    - ``angle_cluster_tol`` — matches with the same lattice at twists closer than this
      (deg) are the same structure seen at neighbouring grid angles; only the
      lowest-residual one is kept. ``None`` → ``1.5 * degstep``.

    >>> SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    SearchParams(degmin=22.0, degmax=23.0, degstep=0.01, ...)
    """

    degmin: float = 1.0
    degmax: float = 45.0
    degstep: float = 0.01
    dtol: float = 5e-4
    nmin: int = -10
    nmax: int = 10
    goodangles: tuple[float, ...] = (90.0,)
    window_relative_to: str = "frame"
    dedupe: bool = True
    keep_supercells: bool = False
    angle_cluster_tol: Optional[float] = None

    def __post_init__(self) -> None:
        # Accept lists from YAML but store an (immutable, hashable) tuple.
        object.__setattr__(self, "goodangles", tuple(float(a) for a in self.goodangles))
        if self.window_relative_to not in WINDOW_MODES:
            raise ValueError(f"window_relative_to must be one of {WINDOW_MODES}, got {self.window_relative_to!r}")
        if self.degmax <= self.degmin or self.degstep <= 0:
            raise ValueError(f"need degmin < degmax and degstep > 0, got {self.degmin}, {self.degmax}, {self.degstep}")
        if self.nmax <= self.nmin:
            raise ValueError(f"need nmin < nmax, got {self.nmin}, {self.nmax}")

    @property
    def cluster_tol(self) -> float:
        """Effective ``angle_cluster_tol`` in degrees."""
        if self.angle_cluster_tol is None:
            return 1.5 * self.degstep
        return self.angle_cluster_tol

    def window_for(self, lower: MoireStructure) -> tuple[float, float]:
        """The (degmin, degmax) actually swept when ``lower`` is the lower block.

        >>> SearchParams(degmin=1.0, degmax=60.01, window_relative_to="lower_top_layer").window_for(bilayer_at_22_62)
        (23.62, 82.63)                                                 # doctest: +SKIP
        """
        if self.window_relative_to == "frame":
            return self.degmin, self.degmax
        # The notebook used np.round(theta, 3) of the bilayer twist as offset.
        offset = round(lower.layer_rotations[-1], 3)
        return self.degmin + offset, self.degmax + offset

    def estimated_cost(self) -> int:
        """Rough work estimate: number of (angle, lattice point) pairs tested.

        ``match.scan`` tests every lower-lattice point with both coefficients
        non-zero at every angle. Useful to compare rounds before running them.

        >>> SearchParams(degmin=1, degmax=2, degstep=0.01, nmin=-10, nmax=10).estimated_cost()
        40000
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

    >>> Filters(max_residual=1e-3).passes(candidate)   # doctest: +SKIP
    True
    """

    max_residual: Optional[float] = None
    max_cumulative_residual: Optional[float] = None
    max_layer_residual: Optional[float] = None
    max_length_mismatch: Optional[float] = None

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
    residual: float  # max(delvec of a1, delvec of a2), in Å

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


def cell_from_pair(row_a: dict[str, Any], row_b: dict[str, Any], goodangle: float) -> CellMatch:
    """Convert one ``filtermatches`` pair into a right-handed :class:`CellMatch`.

    ``match.filtermatches`` returns both (a, b) and (b, a); swapping to make
    ``a1 x a2 > 0`` maps both orderings onto the same right-handed cell, which is
    also what VASP prefers.
    """
    v1, n1, m1, d1 = _row_to_parts(row_a)
    v2, n2, m2, d2 = _row_to_parts(row_b)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        v1, v2, n1, n2, m1, m2 = v2, v1, n2, n1, m2, m1
    return CellMatch(
        twist=float(row_a["angle"]),
        goodangle=float(goodangle),
        vectors=np.array([v1, v2]),
        lower_coeffs=np.array([n1, n2], dtype=int),
        upper_coeffs=np.array([m1, m2], dtype=int),
        residual=max(d1, d2),
    )


def cells_from_run(results: list[dict[str, Any]]) -> list[CellMatch]:
    """Flatten the output of ``match.run_and_filter`` into :class:`CellMatch` objects."""
    cells = []
    for block in results:
        for row_a, row_b, _extra in block["results"]:
            cells.append(cell_from_pair(row_a, row_b, block["goodangle"]))
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
) -> list[float]:
    """True commensurability error (Å) of ``cell`` with each primitive layer.

    For layer *i* with primitive vectors ``P_i`` rotated by ``rotations[i]``, each
    cell vector ``v`` is written in that layer's basis, rounded to integers and
    mapped back; the distance to ``v`` is the error. The result per layer is the
    worse of the two cell vectors.

    Unlike ``cumulative_residual`` (a max over the steps), this measures the
    final cell directly against every layer, so errors that grow when a
    cell is multiplied up in later rounds are included.
    """
    residuals = []
    for poscar, rotation in zip(poscars, rotations):
        primitive = read_primitive(poscar, search_dirs)["in_plane"]
        basis = rotation_matrix_2d(rotation) @ primitive.T  # columns = rotated a, b
        worst = 0.0
        for vector in cell:
            coords = np.linalg.solve(basis, vector)
            error = np.linalg.norm(basis @ np.round(coords) - vector)
            worst = max(worst, float(error))
        residuals.append(worst)
    return residuals


def _max_ignoring_none(*values: Optional[float]) -> float:
    return max(value for value in values if value is not None)


def record_from_cell(
    lower: MoireStructure,
    upper: MoireStructure,
    cell: CellMatch,
    search_dirs: Sequence[PathLike] = (),
) -> MoireStructure:
    """Build the child record for ``lower`` + ``upper`` combined at ``cell``.

    Lists are concatenated lower-first; the upper block's layer rotations are
    shifted by the twist (block-rigid rotation).
    """
    rotations = list(lower.layer_rotations) + [rotation + cell.twist for rotation in upper.layer_rotations]
    poscars = list(lower.source_poscar) + list(upper.source_poscar)
    n_layers = lower.n_layers + upper.n_layers
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
        layer_residuals=layer_residuals(cell.vectors, poscars, rotations, search_dirs),
        coefficients={"lower": cell.lower_coeffs.tolist(), "upper": cell.upper_coeffs.tolist()},
    )


def find_candidates(
    lower: MoireStructure,
    upper: MoireStructure,
    params: SearchParams,
    search_dirs: Sequence[PathLike] = (),
) -> list[MoireStructure]:
    """All commensurate cells of ``lower`` + twisted ``upper`` (no residual filters).

    The only selection applied is the good-angle match inside
    ``match.run_and_filter`` (plus de-duplication if ``params.dedupe``).

    >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    >>> params = SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    >>> [round(c.twist, 2) for c in find_candidates(root, root, params)]   # doctest: +SKIP
    [22.62]
    """
    if lower.units != upper.units:
        raise ValueError(f"unit mismatch: {lower.units} vs {upper.units}")
    degmin, degmax = params.window_for(lower)
    a1, a2 = lower.lattice_vectors
    g1, g2 = upper.lattice_vectors
    results = run_and_filter(
        a1, a2, g1, g2,
        degmin=degmin, degmax=degmax, degstep=params.degstep,
        dtol=params.dtol, nmin=params.nmin, nmax=params.nmax,
        goodangles=list(params.goodangles),
    )
    cells = cells_from_run(results)
    if params.dedupe and cells:
        cells = deduplicate(cells, params.keep_supercells, params.cluster_tol)
    return [record_from_cell(lower, upper, cell, search_dirs) for cell in cells]


def search_pair(
    lower: MoireStructure,
    upper: MoireStructure,
    params: SearchParams,
    filters: Optional[Filters] = None,
    search_dirs: Sequence[PathLike] = (),
) -> list[MoireStructure]:
    """Commensurate cells of ``lower`` + twisted ``upper`` that pass ``filters``.

    This is the atomic operation of the whole N-layer search. ``upper`` is rotated
    rigidly by the swept twist relative to ``lower``'s frame. Each returned record
    has ``parents = [lower.id, upper.id]``.

    >>> root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")
    >>> params = SearchParams(degmin=22.0, degmax=23.0, nmin=-10, nmax=10, goodangles=(90.0,))
    >>> bilayers = search_pair(root, root, params, Filters(max_residual=1e-3))   # doctest: +SKIP
    >>> bilayers[0].twist, bilayers[0].residual                                  # doctest: +SKIP
    (22.620000000000097, 3.39e-05)
    """
    filters = filters or Filters()
    candidates = find_candidates(lower, upper, params, search_dirs)
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
    """Search parameters per round: a ``default`` plus overrides keyed by round number.

    >>> schedule = SearchSchedule(SearchParams(nmin=-10, nmax=10), {2: {"nmin": -40, "nmax": 40}})
    >>> schedule.params_for_round(1).nmax, schedule.params_for_round(2).nmax
    (10, 40)
    """

    default: SearchParams = field(default_factory=SearchParams)
    rounds: dict[int, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rounds = {int(number): dict(overrides) for number, overrides in self.rounds.items()}
        for number in self.rounds:
            self.params_for_round(number)  # validate overrides early

    def params_for_round(self, round_number: int) -> SearchParams:
        """``default`` with this round's overrides applied."""
        overrides = self.rounds.get(round_number, {})
        unknown = set(overrides) - set(SearchParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown search parameter(s) for round {round_number}: {sorted(unknown)}")
        return replace(self.default, **overrides)

    @classmethod
    def notebook_defaults(cls, goodangles: Sequence[float] = (90.0,), max_round: int = 10) -> "SearchSchedule":
        """The settings of the notebooks: round 1 at n = ±10, every later round at ±40."""
        default = SearchParams(degmin=1.0, degmax=45.0, degstep=0.01, dtol=5e-4, nmin=-10, nmax=10,
                               goodangles=tuple(goodangles))
        later = {number: {"nmin": -40, "nmax": 40} for number in range(2, max_round + 1)}
        return cls(default, later)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in the stack-spec layout (``default`` + ``rounds``)."""
        default = asdict(self.default)
        default["goodangles"] = list(default["goodangles"])
        return {"default": default, "rounds": {number: dict(o) for number, o in self.rounds.items()}}


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
) -> dict[str, Any]:
    """The stack-spec dict equivalent to a ``build_stack`` call (for ``stack_spec.yaml``)."""
    return {
        "name": name,
        "output_dir": str(output_dir),
        "layers": [{"poscar": layer.source_poscar[0], "material": layer.material[0]} for layer in layers],
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

    plan = plan_rounds(len(layers))
    if dry_run:
        return _dry_run(layers, plan, schedule, run_dir, say)

    store = RunStore(Path(run_dir))
    if spec is None:
        spec = job_to_spec_dict(Path(run_dir).name, layers, schedule, filters, run_dir)
    if spec:
        with open(store.run_dir / SPEC_FILENAME, "w", encoding="utf-8") as handle:
            yaml.safe_dump(spec, handle, sort_keys=False)

    # Round 0: the roots. ``blocks`` maps a span to its candidates that passed.
    blocks: dict[Span, list[MoireStructure]] = {}
    for position, layer in enumerate(layers):
        store.add(layer, round_number=0, passed_filter=True)
        blocks[(position, position + 1)] = [layer]
    store.write_index()
    say(format_plan(plan))

    reports: list[StepReport] = []
    for round_number, steps in enumerate(plan, start=1):
        params = schedule.params_for_round(round_number)
        new_blocks: dict[Span, list[MoireStructure]] = {}
        for step in steps:
            report = StepReport(round_number, step, len(blocks[step.lower]),
                                None if step.is_carry else len(blocks[step.upper]),
                                cost_per_search=None if step.is_carry else params.estimated_cost())
            if step.is_carry:
                new_blocks[step.result] = blocks[step.lower]
                say(str(report))
                reports.append(report)
                continue
            say(f"round {round_number}: {step}: {report.n_searches} searches "
                f"(~{params.estimated_cost():.2e} lattice-point tests each)")
            started = time.perf_counter()
            passed, n_found = _run_step(blocks[step.lower], blocks[step.upper], params, filters,
                                        store, round_number, search_dirs)
            report.n_found, report.n_passed = n_found, len(passed)
            report.seconds = time.perf_counter() - started
            new_blocks[step.result] = passed
            store.write_index()  # keep the index current after every step
            say(str(report))
            reports.append(report)
        blocks = new_blocks

    full_span = (0, len(layers))
    return StackResult(Path(run_dir), plan, reports, blocks[full_span], dry_run=False)


def _run_step(
    lowers: list[MoireStructure],
    uppers: list[MoireStructure],
    params: SearchParams,
    filters: Filters,
    store: RunStore,
    round_number: int,
    search_dirs: Sequence[PathLike],
) -> tuple[list[MoireStructure], int]:
    """All lower x upper searches of one step; store everything, return the passing ones."""
    passed: dict[str, MoireStructure] = {}  # keyed by id: identical recipes count once
    n_found = 0
    for lower in lowers:
        for upper in uppers:
            for candidate in find_candidates(lower, upper, params, search_dirs):
                ok = filters.passes(candidate)
                store.add(candidate, round_number=round_number, passed_filter=ok)
                n_found += 1
                if ok:
                    passed.setdefault(candidate.id, candidate)
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
        params = schedule.params_for_round(round_number)
        next_known: dict[Span, Optional[int]] = {}
        for step in steps:
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
