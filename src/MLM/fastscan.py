"""A vectorised re-implementation of ``match.run`` / ``match.run_and_filter``.

``match.scan`` loops over every lattice point in Python, once per angle. Here the
lattice points of a whole block of angles are solved in one NumPy call, which gives
the same matches many times faster. ``match.py`` itself is unchanged and stays the
reference: with the default options :func:`run` returns *the same DataFrame* as
``match.run`` (same rows, order and values; checked in ``tests/test_fastscan.py``),
and :func:`run_and_filter` reuses ``match.filtermatches`` unchanged.

Two options lift limits of ``match.scan`` (both off by default, i.e. identical results):

- ``include_axes=True``: also try lattice points with a zero coefficient, ``n = (k, 0)``
  and ``(0, k)``. ``match.scan`` skips them, so e.g. the lower block's own cell can never
  be found in round 2.
- ``max_per_angle``: ``match.scan`` keeps only the 25 shortest matches per angle; pass a
  larger number (or ``None`` for all) to keep more.

>>> from MLM.io import read_vasp
>>> lattice, _, _ = read_vasp("moire_structures/bto/bto_1-layer.vasp")
>>> a1, a2 = lattice["a"][:2], lattice["b"][:2]
>>> table = run(a1, a2, a1, a2, 22.0, 23.0, 0.01, 5e-4, -10, 10)
>>> sorted(set(round(x, 2) for x in table["angle"]))
[22.62]
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import polars as pl

from MLM.match import filtermatches

#: Angles solved per NumPy call; bounds memory to ~ chunk x 2 x (lattice points) floats.
ANGLE_CHUNK = 256


def lattice_points(nmin: int, nmax: int, include_axes: bool = False) -> np.ndarray:
    """2 x P integer coefficients in ``match.scan``'s order (x outer, y inner).

    ``match.scan`` uses ``range(nmin, nmax)`` (``nmax`` excluded) and skips every point
    with a zero coefficient; ``include_axes=True`` only skips the origin.

    >>> lattice_points(-1, 2).tolist()
    [[-1, -1, 1, 1], [-1, 1, -1, 1]]
    """
    points = []
    for x in range(nmin, nmax):
        for y in range(nmin, nmax):
            if include_axes:
                if x == 0 and y == 0:
                    continue
            elif x == 0 or y == 0:
                continue
            points.append([x, y])
    return np.array(points, dtype=np.int64).T


def _rotation_stack(angles_deg: np.ndarray) -> np.ndarray:
    """(T, 2, 2) anticlockwise rotations, built exactly as in ``match.scan``."""
    theta = np.deg2rad(angles_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    rotations = np.empty((len(angles_deg), 2, 2), dtype=np.float64)
    rotations[:, 0, 0], rotations[:, 0, 1] = cos, -sin
    rotations[:, 1, 0], rotations[:, 1, 1] = sin, cos
    return rotations


def scan_angles(
    A: np.ndarray,
    G: np.ndarray,
    angles: Sequence[float],
    tol: float,
    nmin: int,
    nmax: int,
    include_axes: bool = False,
    max_per_angle: Optional[int] = 25,
    relative_tol: float = 0.0,
) -> list[dict[str, Any]]:
    """All matches for every angle in ``angles`` (rows like ``match.scan`` + ``"angle"``).

    ``A``, ``G``: 2x2 matrices whose *columns* are the lower and upper lattice vectors.
    A point matches if both fractional errors are below ``tol + relative_tol · |m|``
    (``relative_tol = 0``: exactly ``match.scan``'s criterion). A strain ε moves the
    upper coordinates by about ε·|m|, so strained searches use ``relative_tol ≈ ε``.
    """
    points = lattice_points(nmin, nmax, include_axes)
    lower_points = A @ points  # all lower lattice points, Cartesian (2 x P)
    angles = np.asarray(angles, dtype=float)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(angles), ANGLE_CHUNK):
        chunk = angles[start:start + ANGLE_CHUNK]
        B = _rotation_stack(chunk) @ G  # rotated upper lattice for every angle (T, 2, 2)
        targets = np.broadcast_to(lower_points, (len(chunk),) + lower_points.shape)
        solution = np.linalg.solve(B, targets)  # coordinates in the rotated upper basis
        rounded = np.round(solution)
        fraction = solution - rounded
        limit = tol if not relative_tol else tol + relative_tol * np.linalg.norm(solution, axis=1)
        is_match = (np.abs(fraction[:, 0, :]) < limit) & (np.abs(fraction[:, 1, :]) < limit)
        for t in np.nonzero(is_match.any(axis=1))[0]:
            rows.extend(_rows_for_angle(chunk[t], A, B[t], points, solution[t], rounded[t], fraction[t],
                                        np.nonzero(is_match[t])[0], max_per_angle))
    return rows


def _rows_for_angle(angle, A, B, points, solution, rounded, fraction, columns, max_per_angle):
    """Build the match dicts of one angle exactly like ``match.scan`` does."""
    matches = []
    for j in columns:
        n1, n2 = points[:, j]
        m1, m2 = rounded[:, j]
        matchpoint = A @ np.array([n1, n2])
        delvec = np.linalg.norm(B @ np.array([m1, m2]) - matchpoint)
        matches.append({
            "n1": n1, "n2": n2, "m1": int(m1), "m2": int(m2),
            "norm_m1m2": np.linalg.norm([m1, m2]),
            "err_m1": fraction[0, j], "err_m2": fraction[1, j],
            "delvec": delvec,
            "m1_true": solution[0, j], "m2_true": solution[1, j],
            "matchpoint": matchpoint.tolist(),
        })
    matches = sorted(matches, key=lambda x: np.linalg.norm([x["n1"], x["n2"]]))
    if max_per_angle is not None:
        matches = matches[:max_per_angle]
    for match in matches:
        match["angle"] = angle
    return matches


def run(a1, a2, g1, g2, degmin, degmax, degstep, dtol, nmin, nmax,
        include_axes: bool = False, max_per_angle: Optional[int] = 25, relative_tol: float = 0.0) -> pl.DataFrame:
    """Drop-in for ``match.run`` (same arguments, same DataFrame), plus the options."""
    A = np.array([a1, a2]).T
    G = np.array([g1, g2]).T
    angles = np.arange(degmin, degmax, degstep)
    angles = angles[angles != 0]  # match.run skips exactly 0
    return pl.DataFrame(scan_angles(A, G, angles, dtol, nmin, nmax, include_axes, max_per_angle, relative_tol))


def run_and_filter(a1, a2, g1, g2, degmin, degmax, degstep, dtol, nmin, nmax, goodangles,
                   include_axes: bool = False, max_per_angle: Optional[int] = 25,
                   relative_tol: float = 0.0) -> list[dict[str, Any]]:
    """Drop-in for ``match.run_and_filter``; the filtering is ``match.filtermatches`` itself."""
    df = run(a1, a2, g1, g2, degmin, degmax, degstep, dtol, nmin, nmax, include_axes, max_per_angle, relative_tol)
    if df.height == 0:
        return []
    return [{"goodangle": good_angle, "results": filtermatches(a1, a2, df, good_angle)}
            for good_angle in goodangles]
