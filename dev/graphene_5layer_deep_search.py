"""Deep search for 5-layer twisted graphene with adaptive n shells (prototype).

Usage (from the repository root, in the ``mlm`` environment)::

    python dev/graphene_5layer_deep_search.py --quick            # ~1 min smoke test
    nohup python dev/graphene_5layer_deep_search.py > deep.log 2>&1 &
    python dev/graphene_5layer_deep_search.py --resume           # continue after an interruption
    python dev/graphene_5layer_deep_search.py --report-only      # redo the summary / files

What it does
------------
The normal tournament (``build_stack``: (1,2) (3,4) -> (12)(34) -> (1234)+5), but every
interface may take *any* twist in 0.5-59.5° (graphene repeats every 60°; 0 and 60 are
the aligned layer), and every pairwise search grows its coefficient range ``n`` in
shells instead of scanning one fixed box:

1. scan the box |n| < n_start (round 1: 10, round 2: 20, round 3: 30);
2. if no cell was found, scan only the next shell (n_start <= |n| < n_start + 10), and
   so on. Raw matches of all shells are kept, so a cell may combine vectors from
   different shells. Nothing is scanned twice;
3. a first match is not yet proof of the smallest cell: shells continue until
   n * h > L, where L is the longest vector of the smallest cell found and h the
   smallest height of the lower cell (a point with max|n_i| >= n is at least n*h long);
4. a search also stops at ``n_limit`` or when even the smallest point of the next
   shell is longer than the atom budget allows (``max_atoms``: cells of the final
   5-layer stack). Cells over the budget are dropped before they are stored.

Because the final cell is a supercell of every block below it, the budget is a valid
bound in every round. With the budget, rounds 2 and 3 are complete (every cell up to
``max_atoms`` at every twist on the grid); round 1 keeps the smallest bilayers
(box |n| < 10), as designed. Set ``--round1-start`` higher to also allow larger bilayers.

Implementation: the script swaps ``MLM.search.find_candidates`` for
:func:`adaptive_find_candidates` at import time. Worker processes (spawn) import this
script again as ``__mp_main__``, so they get the swap too; the settings travel in the
``MLM_DEEP_CONFIG`` environment variable. ``match.py`` and the package are untouched.

Output (``--run-dir``, default ``runs/graphene_5layer_deep``): the usual run directory
(index.csv, yamls/, run_info.yaml), plus
``shells.tsv``         one line per search: blocks, box reached, why it stopped, time
``summary_5layer.csv`` every 5-layer stack: atoms, interface twists, residuals, a key
                       that is the same for stacks related by flipping or mirroring
``structures/``        VASP files of the ``--write-top`` smallest distinct stacks,
                       each checked (PBC seam, overlaps, gaps) before writing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from MLM import fastscan
from MLM import search as _search
from MLM.match import filtermatches
from MLM.search import (
    Filters,
    SearchParams,
    SearchSchedule,
    build_stack,
    cells_from_run,
    deduplicate,
    merge_equivalent_twists,
    record_from_cell,
    twist_offset,
)
from MLM.structure import MoireStructure, read_primitive

REPO = Path(__file__).resolve().parents[1]
POSCAR = "moire_structures/graphene/graphene_monolayer.vasp"
N_LAYERS = 5
GAP_TARGET = 3.35  # Å between graphene sheets (graphite)

DEFAULTS: dict[str, Any] = {
    "max_atoms": 3000,   # budget for the final 5-layer cell (None = no budget, pure "smallest" rule)
    "n_step": 10,        # shell width
    "n_limit": 200,      # never scan beyond |n| < n_limit
    "log": None,         # shells.tsv path (set by main)
}
_original_find_candidates = _search.find_candidates


def config() -> dict[str, Any]:
    return {**DEFAULTS, **json.loads(os.environ.get("MLM_DEEP_CONFIG", "{}"))}


# --- adaptive shells --------------------------------------------------------------
def shell_points(inner: int, outer: int, include_axes: bool) -> np.ndarray:
    """Points of the box range(-outer, outer)^2 not in range(-inner, inner)^2 (2 x P)."""
    points = fastscan.lattice_points(-outer, outer, include_axes)
    if inner <= 0:
        return points
    in_inner = np.all((points >= -inner) & (points < inner), axis=0)
    return points[:, ~in_inner]


def scan_points(A: np.ndarray, G: np.ndarray, angles: np.ndarray, tol: float, points: np.ndarray) -> list[dict]:
    """``fastscan.scan_angles`` for a given set of lattice points, without the per-angle cap."""
    if points.shape[1] == 0:
        return []
    lower_points = A @ points
    chunk_size = int(max(1, min(fastscan.ANGLE_CHUNK, 2_000_000 // points.shape[1])))  # bounded memory
    rows: list[dict] = []
    for start in range(0, len(angles), chunk_size):
        chunk = angles[start:start + chunk_size]
        B = fastscan._rotation_stack(chunk) @ G
        solution = np.linalg.solve(B, np.broadcast_to(lower_points, (len(chunk),) + lower_points.shape))
        rounded = np.round(solution)
        fraction = solution - rounded
        is_match = (np.abs(fraction[:, 0, :]) < tol) & (np.abs(fraction[:, 1, :]) < tol)
        for t in np.nonzero(is_match.any(axis=1))[0]:
            rows.extend(fastscan._rows_for_angle(chunk[t], A, B[t], points, solution[t], rounded[t], fraction[t],
                                                 np.nonzero(is_match[t])[0], None))
    return rows


def capped(rows: list[dict], cap: Optional[int]) -> list[dict]:
    """The ``cap`` shortest-|n| matches per angle of the union (what one big box would keep)."""
    if cap is None:
        return rows
    by_angle: dict[float, list[dict]] = {}
    for row in rows:
        by_angle.setdefault(row["angle"], []).append(row)
    out: list[dict] = []
    for angle in sorted(by_angle):
        out.extend(sorted(by_angle[angle], key=lambda r: math.hypot(r["n1"], r["n2"]))[:cap])
    return out


def commensurate_twists(max_cells: float, degmin: float, degmax: float) -> np.ndarray:
    """Every twist (deg) at which two graphene layers share a cell of <= ``max_cells`` primitive cells.

    Closed form for the hexagonal lattice: coprime 0 < m < n give cos θ = (n²+4nm+m²)/(2S) and
    sin θ = √3(n²−m²)/(2S), S = n²+nm+m²; the cell holds S primitive cells (S/3 if 3 | n−m).
    θ and 60°−θ (mirror) are both commensurate. Matches the grid search exactly (138 twists for
    250 cells, 274 for 500).
    """
    twists = set()
    top = int(math.sqrt(3 * max_cells)) + 3
    for m in range(1, top):
        for n in range(m + 1, m + top):
            if math.gcd(m, n) != 1:
                continue
            size = n * n + n * m + m * m
            cells = size // 3 if (n - m) % 3 == 0 else size
            if cells > max_cells:
                continue
            theta = math.degrees(math.atan2(math.sqrt(3) * (n * n - m * m), n * n + 4 * n * m + m * m))
            for twist in (theta, 60.0 - theta):
                if degmin <= twist < degmax:
                    twists.add(round(twist, 9))  # different (m, n) can give the same twist
    return np.array(sorted(twists))


_twists_cache: dict[str, np.ndarray] = {}


def exact_twists(settings: dict[str, Any]) -> Optional[np.ndarray]:
    path = settings.get("twists_file")
    if not path:
        return None
    if path not in _twists_cache:
        _twists_cache[path] = np.array(json.loads(Path(path).read_text()))
    return _twists_cache[path]


def cell_lengths(vectors: np.ndarray) -> float:
    return float(np.linalg.norm(vectors, axis=1).max())


def adaptive_find_candidates(lower: MoireStructure, upper: MoireStructure, params: SearchParams,
                             search_dirs=(), strain_limits=(0.0, 0.0), layer_elastic=None) -> list[MoireStructure]:
    """``find_candidates`` for rigid searches, with n grown in shells (see the module docstring)."""
    if any(strain_limits) or params.scanner != "fast":
        return _original_find_candidates(lower, upper, params, search_dirs, strain_limits, layer_elastic)
    settings = config()
    started = time.perf_counter()
    A, G = lower.lattice_vectors.T, upper.lattice_vectors.T
    a1, a2 = lower.lattice_vectors
    degmin, degmax = params.window_for(lower)
    angles = np.arange(degmin, degmax, params.degstep)
    twists = exact_twists(settings)
    if twists is not None:
        # Every interface joins two graphene layers, so its twist must be one of the bilayer
        # commensurate twists: test exactly those angles instead of the whole grid.
        angles = twist_offset(lower) + twists[(twists >= params.degmin) & (twists < params.degmax)]
    angles = angles[angles != 0]
    height = abs(np.linalg.det(A)) / np.linalg.norm(lower.lattice_vectors, axis=1).max()
    max_area = settings.get("max_area")
    max_length = None if max_area is None else 1.02 * math.sqrt(max_area / math.sin(math.radians(60.0)))
    n_cap = settings["n_limit"] if max_length is None else min(settings["n_limit"], math.ceil(max_length / height) + 1)
    cap = 25 if params.max_matches_per_angle == "auto" else params.max_matches_per_angle

    inner, outer = 0, max(1, min(params.nmax, n_cap))
    rows: list[dict] = []
    cells: list = []
    while True:
        rows.extend(scan_points(A, G, angles, params.dtol, shell_points(inner, outer, params.include_axes)))
        cells = []
        if rows:
            table = pl.DataFrame(capped(rows, cap))
            results = [{"goodangle": g, "results": filtermatches(a1, a2, table, g)} for g in params.goodangles]
            cells = cells_from_run(results, offset=twist_offset(lower))
            if max_area is not None:
                cells = [c for c in cells if abs(np.linalg.det(c.vectors)) <= max_area * (1 + 1e-9)]
            if params.dedupe and cells:
                cells = deduplicate(cells, params.keep_supercells, params.cluster_tol)
        if cells and outer * height > min(cell_lengths(c.vectors) for c in cells):
            reason = "smallest"  # nothing unscanned can be shorter than the smallest cell found
            break
        if outer >= n_cap:
            reason = "budget" if max_length is not None and n_cap < settings["n_limit"] else "n_limit"
            break
        inner, outer = outer, min(outer + settings["n_step"], n_cap)

    cells = merge_equivalent_twists(cells, upper, params.merge_equivalent_twists)
    records = [record_from_cell(lower, upper, c, search_dirs, layer_elastic) for c in cells]
    if settings.get("log"):
        with open(settings["log"], "a", encoding="utf-8") as handle:  # one short line: safe to append in parallel
            handle.write(f"{lower.id}\t{upper.id}\t{outer}\t{n_cap}\t{reason}\t{len(records)}\t"
                         f"{time.perf_counter() - started:.2f}\n")
    return records


_search.find_candidates = adaptive_find_candidates  # also runs in spawned workers (they re-import this file)


# --- run ----------------------------------------------------------------------------
def atoms_per_area(root: MoireStructure) -> float:
    return len(read_primitive(str(REPO / POSCAR))["coords"]) / abs(np.linalg.det(root.lattice_vectors))


def interface_twists(record: MoireStructure) -> list[float]:
    rotations = np.asarray(record.layer_rotations)
    return [round(float(t) % 60.0, 3) for t in np.diff(rotations)]


def equivalence_key(twists: list[float]) -> str:
    """Same for stacks related by flipping upside down (reversed order) or a mirror (t -> 60 - t)."""
    def norm(seq):
        return tuple(round(t % 60.0, 2) % 60.0 for t in seq)
    variants = [norm(twists), norm(twists[::-1]), norm([-t for t in twists]), norm([-t for t in twists[::-1]])]
    return " ".join(f"{t:.2f}" for t in min(variants))


def write_checked(run_dir: Path, structure_id: str, out: Path) -> dict[str, Any]:
    """Rebuild one stack at a 3.35 Å gap, run the checks (PBC seam, overlaps, gaps), write VASP."""
    from MLM.build import _build_from_run, default_search_dirs, write_built
    from MLM.checks import check_built

    gap = GAP_TARGET - read_primitive(str(REPO / POSCAR))["coords"][:, 2].min()
    search_dirs = default_search_dirs(run_dir)
    built = _build_from_run(run_dir, structure_id, gap, 50.0, search_dirs, None, None)
    checked = check_built(built, search_dirs=search_dirs, symmetry=False)
    path = write_built(built, "vasp", out)
    if not checked.ok:
        print(f"CHECK FAILED {structure_id}: {checked.problems}")
    return {"written": str(path.relative_to(run_dir)), "check_ok": checked.ok,
            "seam_max_error": checked.info.get("seam_max_error"),
            "min_interlayer": checked.info.get("min_interlayer_distance")}


def write_below(run_dir: Path, limit: int) -> None:
    """Write every stack (equivalent ones included) with fewer than ``limit`` atoms, from summary_5layer.csv."""
    table = pd.read_csv(run_dir / "summary_5layer.csv", low_memory=False)
    chosen = table[table["atoms"] < limit].drop(columns=["written", "check_ok", "seam_max_error", "min_interlayer"])
    folder = run_dir / f"structures_lt{limit}"
    results = []
    for number, structure_id in enumerate(chosen["id"], start=1):
        results.append(write_checked(run_dir, structure_id, folder / f"{structure_id}.vasp"))
        if number % 50 == 0:
            print(f"  {number}/{len(chosen)}", flush=True)
    chosen = pd.concat([chosen.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    chosen.to_csv(folder / "index.csv", index=False)
    print(f"{len(chosen)} stacks < {limit} atoms ({chosen['first_of_its_class'].sum()} distinct up to flip/mirror) "
          f"written to {folder}; checks passed: {chosen['check_ok'].sum()}/{len(chosen)}, "
          f"max seam error {chosen['seam_max_error'].max():.1e} Å")


def report(run_dir: Path, final: list[MoireStructure], root: MoireStructure, write_top: int) -> None:
    density = atoms_per_area(root)
    rows = []
    for record in final:
        twists = interface_twists(record)
        area = abs(np.linalg.det(record.lattice_vectors))
        rows.append({"id": record.id, "atoms": int(round(N_LAYERS * density * area)),
                     "atoms_per_layer": int(round(density * area)), "cell_length": cell_lengths(record.lattice_vectors),
                     "twists": " ".join(f"{t:.3f}" for t in twists),
                     "distinct_twists": len(set(round(t, 2) for t in twists)),
                     "equivalence_key": equivalence_key(twists), "max_layer_residual": record.worst_layer_residual,
                     "cumulative_residual": record.cumulative_residual})
    table = pd.DataFrame(rows)
    if table.empty:
        print("no 5-layer structures found")
        return
    table = table.sort_values(["atoms", "distinct_twists", "max_layer_residual"], ascending=[True, False, True])
    table["first_of_its_class"] = ~table["equivalence_key"].duplicated()

    table["written"], table["check_ok"], table["seam_max_error"], table["min_interlayer"] = "", None, np.nan, np.nan
    for index in table.index[table["first_of_its_class"]][:write_top]:
        structure_id = table.at[index, "id"]
        for key, value in write_checked(run_dir, structure_id, run_dir / "structures" / f"{structure_id}.vasp").items():
            table.at[index, key] = value
    table.to_csv(run_dir / "summary_5layer.csv", index=False)

    classes = table[table["first_of_its_class"]]
    print(f"\n5-layer stacks: {len(table)} ({len(classes)} distinct up to flip/mirror), "
          f"{classes['distinct_twists'].eq(4).sum()} distinct classes with four different twists")
    print(f"atoms: {table['atoms'].min()} .. {table['atoms'].max()}")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 40):
        print(classes[["atoms", "twists", "max_layer_residual", "check_ok", "seam_max_error", "written"]]
              .head(min(write_top, 40)).to_string(index=False))
    print(f"\nfull table: {run_dir / 'summary_5layer.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs" / "graphene_5layer_deep")
    parser.add_argument("--max-atoms", type=int, default=DEFAULTS["max_atoms"],
                        help="largest final 5-layer cell kept (0 = no budget)")
    parser.add_argument("--degstep", type=float, default=0.001)
    parser.add_argument("--degmin", type=float, default=0.5)
    parser.add_argument("--degmax", type=float, default=59.5)
    parser.add_argument("--round1-start", type=int, default=10, help="round-1 box |n| < this")
    parser.add_argument("--n-step", type=int, default=DEFAULTS["n_step"])
    parser.add_argument("--n-limit", type=int, default=DEFAULTS["n_limit"])
    parser.add_argument("--max-residual", type=float, default=1e-3, help="Å, for residual and every layer residual")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--write-top", type=int, default=50, help="write + check this many smallest distinct stacks")
    parser.add_argument("--grid", action="store_true",
                        help="scan the whole twist grid in every search (default: only the exact commensurate "
                             "graphene twists, found in closed form; same structures, much faster, ~0 residual)")
    parser.add_argument("--index-format", choices=["csv", "sqlite"], default="sqlite",
                        help="csv for small runs that are kept in git (examples/runs/graphene_5layer)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="only redo summary_5layer.csv and the files")
    parser.add_argument("--write-below", type=int, default=None, metavar="ATOMS",
                        help="only write (and check) every finished stack with fewer atoms, from summary_5layer.csv")
    parser.add_argument("--quick", action="store_true", help="smoke test: 1000-atom budget, own run dir")
    args = parser.parse_args()
    if args.quick:
        args.max_atoms = 1000
        args.run_dir = args.run_dir.with_name(args.run_dir.name + "_quick")
    os.chdir(REPO)  # the run stores POSCAR paths relative to the repository
    run_dir = args.run_dir
    if args.write_below is not None:
        write_below(run_dir, args.write_below)
        return 0
    root = MoireStructure.from_poscar(POSCAR, "graphene")

    max_area = None if args.max_atoms <= 0 else args.max_atoms / (N_LAYERS * atoms_per_area(root))
    settings = {"max_atoms": args.max_atoms or None, "max_area": max_area, "n_step": args.n_step,
                "n_limit": args.n_limit, "log": str(run_dir / "shells.tsv"),
                "twists_file": None if args.grid else str(run_dir / "twists.json")}
    search = {"grid": args.grid,
              "degmin": args.degmin, "degmax": args.degmax, "degstep": args.degstep, "round1_start": args.round1_start,
              "max_residual": args.max_residual}
    saved = run_dir / "deep_search_config.json"
    if args.report_only or args.resume:
        previous = json.loads(saved.read_text())
        if previous != {"adaptive": settings, "search": search}:
            sys.exit(f"{saved} differs from these options:\n  saved: {previous}\n  now:   "
                     f"{ {'adaptive': settings, 'search': search} }\nuse the same options (or another --run-dir)")
    if args.report_only:
        from MLM.index import find, get

        ids = find(run_dir, n_layers=N_LAYERS, passed_filter=True)["id"]
        report(run_dir, [get(run_dir, i) for i in ids], root, args.write_top)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps({"adaptive": settings, "search": search}, indent=1))
    if not args.resume and Path(settings["log"]).exists():
        Path(settings["log"]).unlink()
    if not Path(settings["log"]).exists():
        Path(settings["log"]).write_text("lower\tupper\tn_reached\tn_cap\tstopped_by\tcells\tseconds\n")
    if settings["twists_file"]:
        max_cells = 1e12 if max_area is None else max_area / abs(np.linalg.det(root.lattice_vectors))
        twists = commensurate_twists(max_cells, args.degmin, args.degmax)
        Path(settings["twists_file"]).write_text(json.dumps(twists.tolist()))
        print(f"{len(twists)} commensurate twists with <= {max_cells:.0f} cells per layer", flush=True)
    os.environ["MLM_DEEP_CONFIG"] = json.dumps(settings)  # inherited by the worker processes

    # At exact twists true matches are exact (~1e-9); a tight dtol drops chance near-coincidences, which
    # could otherwise pose as a smaller cell at the same twist and hide the real one in the de-duplication.
    base = SearchParams(degmin=args.degmin, degmax=args.degmax, degstep=args.degstep, nmin=-args.round1_start,
                        nmax=args.round1_start, goodangles=(60.0,), include_axes=True,
                        dtol=5e-4 if args.grid else 1e-6)
    schedule = SearchSchedule(base, rounds={2: {"nmin": -20, "nmax": 20}, 3: {"nmin": -30, "nmax": 30}})
    filters = Filters(max_residual=args.max_residual, max_layer_residual=args.max_residual)
    budget = f"{args.max_atoms} atoms (cell area <= {max_area:.0f} Å²)" if max_area else "none"
    print(f"5-layer graphene, twists {args.degmin}-{args.degmax}° step {args.degstep}°, budget {budget}, "
          f"shells start 10/20/30 step {args.n_step} limit {args.n_limit}, {args.workers} workers", flush=True)

    last = {"t": 0.0}

    def progress(info: dict) -> None:
        now = time.time()
        if info["done"] == info["total"] or now - last["t"] > 60:
            last["t"] = now
            print(f"  {time.strftime('%H:%M:%S')} round {info['round']} {info['step']}: "
                  f"{info['done']}/{info['total']} searches", flush=True)

    started = time.time()
    result = build_stack([root] * N_LAYERS, schedule, filters, run_dir=run_dir, workers=args.workers,
                         resume=args.resume, index_format=args.index_format, progress_callback=progress,
                         log=lambda message: print(message, flush=True))
    print(result.summary())
    print(f"search time: {(time.time() - started) / 3600:.2f} h", flush=True)
    if result.status == "finished":
        report(run_dir, result.final, root, args.write_top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
