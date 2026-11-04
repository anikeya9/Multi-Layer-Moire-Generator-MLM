"""Heterostrain: the exact strain that turns a near-coincidence into a perfect one.

Two lattices that are almost commensurate (for example PTO, a = 3.880 Å, on STO,
a = 3.913 Å) give moiré cells with a residual: the cell vector built from the lower
layer, ``P_A = A n``, and the one built from the rotated upper layer, ``P_B = R(θ) B m``,
differ slightly. Straining the layers a little makes them coincide exactly.

Instead of scanning a grid of strain values, we take each near-coincidence found by the
ordinary (unstrained) search and *solve* for the strain that closes it:

    (I + ε_A) P_A  =  R(θ + δ) (I + ε_B) B M          (both cell vectors)

- ``ε_A``, ``ε_B``: 2x2 strain tensors of the two layers, each in the layer's **own**
  (POSCAR) frame — the usual small-strain convention: a strained lattice vector is
  ``a' = (I + ε) a``. Rigid-body rotations are not strain; they are absorbed into the
  twist correction ``δ``.
- **Strain modes** (``mode``), i.e. which tensors are allowed:

  ============ ====================================== ===========================
  ``isotropic`` ``ε = e·I``                            in-plane isotropic ("biaxial"
                                                        strain in the thin-film
                                                        literature); keeps cell angles
  ``axial``     ``ε = diag(e_xx, e_yy)``               independent normal strains along
                                                        x and y of the layer's POSCAR
                                                        frame (along a1, a2 for square/
                                                        rectangular cells); no shear
  ``full``      ``ε = [[e_xx, e_xy], [e_xy, e_yy]]``   general symmetric in-plane strain
  ============ ====================================== ===========================

- **Sharing the strain.** Only the *relative* strain is fixed by commensurability.
  Each layer has a limit (``limit_A``, ``limit_B``); the mismatch is shared in
  proportion to the limits, by minimising Σ |ε|² / limit over the exact solutions.
  A limit of 0 means the layer is rigid (the usual "fixed substrate" convention);
  equal limits split the mismatch equally (the usual "average lattice" convention).
  Proportional sharing is also the split that is most likely to respect both limits.
- **Strain magnitude** = the largest absolute principal strain (eigenvalue of ε).

With ``full`` the match is always exact (4 equations, enough unknowns). ``isotropic`` and
``axial`` have fewer degrees of freedom; when the two cells also differ in *shape*, the
best fit leaves a (smaller) residual, which is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

#: Allowed values of ``SearchParams.strain_mode``.
STRAIN_MODES = ("isotropic", "axial", "full")

#: Number of strain parameters per layer in each mode.
_N_PARAMS = {"isotropic": 1, "axial": 2, "full": 3}

#: Weight of the strain-sharing penalty. Small, so that the fit stays (numerically)
#: exact; it only selects one solution among the exact ones.
_SHARING_WEIGHT = 1e-6


def strain_tensor(mode: str, params: np.ndarray) -> np.ndarray:
    """2x2 strain tensor (fractions) from the mode's parameters.

    >>> strain_tensor("isotropic", np.array([0.01])).tolist()
    [[0.01, 0.0], [0.0, 0.01]]
    >>> strain_tensor("full", np.array([0.01, -0.02, 0.005])).tolist()
    [[0.01, 0.005], [0.005, -0.02]]
    """
    if mode == "isotropic":
        (e,) = params
        return np.array([[e, 0.0], [0.0, e]])
    if mode == "axial":
        exx, eyy = params
        return np.array([[exx, 0.0], [0.0, eyy]])
    if mode == "full":
        exx, eyy, exy = params
        return np.array([[exx, exy], [exy, eyy]])
    raise ValueError(f"strain mode must be one of {STRAIN_MODES}, got {mode!r}")


def strain_magnitude(epsilon: Optional[np.ndarray]) -> float:
    """Largest absolute principal strain (fraction); 0 for ``None``.

    >>> strain_magnitude(np.array([[0.01, 0.0], [0.0, -0.02]]))
    0.02
    """
    if epsilon is None:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvalsh(epsilon))))


def _rotation(angle_rad: float) -> np.ndarray:
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos, -sin], [sin, cos]])


@dataclass
class StrainFit:
    """Result of :func:`fit_strain`.

    ``lower_strain``/``upper_strain`` are 2x2 tensors (fractions) in each layer's own
    frame, or ``None`` for a rigid layer. ``twist_correction`` (deg) is added to the
    scan angle. ``cell`` (rows a1, a2, lower frame) is the strained moiré cell, and
    ``residual`` (Å) the remaining mismatch: max over a1, a2 of |lower − upper|.
    """

    lower_strain: Optional[np.ndarray]
    upper_strain: Optional[np.ndarray]
    twist_correction: float
    cell: np.ndarray
    residual: float

    @property
    def magnitude(self) -> float:
        """Largest absolute principal strain over both layers (fraction)."""
        return max(strain_magnitude(self.lower_strain), strain_magnitude(self.upper_strain))

    def within(self, limit_lower: float, limit_upper: float) -> bool:
        """True if each layer's strain magnitude respects its limit (fractions)."""
        slack = 1e-9
        return (strain_magnitude(self.lower_strain) <= limit_lower + slack
                and strain_magnitude(self.upper_strain) <= limit_upper + slack)


def fit_strain(
    lower_cell: np.ndarray,
    upper_lattice: np.ndarray,
    upper_coeffs: np.ndarray,
    theta_deg: float,
    mode: str,
    limit_lower: float,
    limit_upper: float,
) -> StrainFit:
    """Solve for the strains (and twist correction) that make a cell commensurate.

    - ``lower_cell``: 2x2, rows = the cell vectors ``A n``, ``A n'`` (lower frame).
    - ``upper_lattice``: 2x2, rows = the upper lattice vectors in the upper layer's own frame.
    - ``upper_coeffs``: 2x2 integers, rows = ``m``, ``m'``.
    - ``theta_deg``: rotation of the upper lattice in the lower frame (the scan angle).
    - ``limit_lower``, ``limit_upper``: strain limits as fractions (0 = rigid layer).

    >>> square = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> fit = fit_strain(square * 1.0, square * 1.01, np.eye(2, dtype=int), 0.0, "isotropic", 0.0, 0.02)
    >>> round(float(fit.upper_strain[0, 0]), 6), fit.lower_strain is None, fit.residual < 1e-9
    (-0.009901, True, True)
    """
    if mode not in STRAIN_MODES:
        raise ValueError(f"strain mode must be one of {STRAIN_MODES}, got {mode!r}")
    lower_cols = np.asarray(lower_cell, dtype=float).T          # columns = cell vectors
    upper_cols = np.asarray(upper_lattice, dtype=float).T @ np.asarray(upper_coeffs, dtype=float).T
    theta = np.deg2rad(theta_deg)
    scale = float(np.mean(np.linalg.norm(lower_cols, axis=0)))  # makes residuals dimensionless
    k = _N_PARAMS[mode]
    strain_lower, strain_upper = limit_lower > 0, limit_upper > 0

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        position = 0
        eps_lower = np.zeros((2, 2))
        eps_upper = np.zeros((2, 2))
        if strain_lower:
            eps_lower = strain_tensor(mode, x[position:position + k])
            position += k
        if strain_upper:
            eps_upper = strain_tensor(mode, x[position:position + k])
            position += k
        return eps_lower, eps_upper, x[position]

    def cells(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eps_lower, eps_upper, delta = unpack(x)
        lower = (np.eye(2) + eps_lower) @ lower_cols
        upper = _rotation(theta + delta) @ (np.eye(2) + eps_upper) @ upper_cols
        return lower, upper

    def residuals(x: np.ndarray) -> np.ndarray:
        lower, upper = cells(x)
        mismatch = ((lower - upper) / scale).ravel()
        # Sharing penalty: minimise sum(eps^2 / limit) among the exact solutions.
        penalty = []
        position = 0
        for strainable, limit in ((strain_lower, limit_lower), (strain_upper, limit_upper)):
            if strainable:
                penalty.extend(_SHARING_WEIGHT * x[position:position + k] / np.sqrt(limit))
                position += k
        return np.concatenate([mismatch, penalty])

    n_unknowns = k * (strain_lower + strain_upper) + 1
    solution = least_squares(residuals, np.zeros(n_unknowns), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    eps_lower, eps_upper, delta = unpack(solution.x)
    lower, upper = cells(solution.x)
    residual = float(np.max(np.linalg.norm(lower - upper, axis=0)))
    return StrainFit(
        lower_strain=eps_lower if strain_lower else None,
        upper_strain=eps_upper if strain_upper else None,
        twist_correction=float(np.rad2deg(delta)),
        cell=lower.T,
        residual=residual,
    )


def scan_tolerance(
    dtol: float,
    limit_total: float,
    nmax_abs: int,
    lower_lattice: np.ndarray,
    upper_lattice: np.ndarray,
    degstep: float,
) -> float:
    """Fractional tolerance for ``match.scan`` that does not miss strainable matches.

    A match that needs a strain ε shows up in ``match.scan`` with a fractional error of
    about ε·|m| in the upper coefficients, plus a rotational error of up to
    (degstep/2 in rad)·|m| between grid angles. ``|m|`` is estimated from the largest
    lower-lattice vector searched. Capped at 0.45 (beyond that every point "matches").
    """
    longest = nmax_abs * float(np.linalg.norm(lower_lattice, axis=1).sum())
    area = abs(np.linalg.det(upper_lattice))
    height = area / float(np.linalg.norm(upper_lattice, axis=1).max())
    m_max = longest / height
    needed = (limit_total + np.deg2rad(degstep) / 2) * m_max
    return float(min(0.45, max(dtol, needed)))
