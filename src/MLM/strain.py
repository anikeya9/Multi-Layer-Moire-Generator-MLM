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


#: 1 J/m² in meV/Å² (1 J = 6.241509e21 meV, 1 m² = 1e20 Å²).
J_PER_M2_IN_MEV_PER_A2 = 62.41509074

#: d(e_xx, e_yy, e_xy)/d(params) for each mode: the parameters as a strain "vector".
_PARAM_TO_VOIGT = {
    "isotropic": np.array([[1.0], [1.0], [0.0]]),
    "axial": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    "full": np.eye(3),
}


@dataclass(frozen=True)
class Elastic:
    """In-plane elastic constants of one layer (isotropic, plane stress).

    - ``young``: 2D Young's modulus in N/m (= J/m²), e.g. ~340 N/m for graphene;
      for a slab, the 3D modulus times its thickness.
    - ``poisson``: Poisson ratio ν (also used for the thickness response when building).

    Elastic energy per area (the standard isotropic plane-stress form):
    W = Y / (2(1 − ν²)) · (ε_xx² + ε_yy² + 2ν ε_xx ε_yy + 2(1 − ν) ε_xy²).

    >>> round(Elastic(young=100.0, poisson=0.25).energy_density(np.eye(2) * 0.01), 6)   # J/m²
    0.013333
    """

    young: float
    poisson: float = 0.0

    def __post_init__(self) -> None:
        if self.young <= 0 or not -1.0 < self.poisson < 0.5:
            raise ValueError(f"need young > 0 and -1 < poisson < 0.5, got {self.young}, {self.poisson}")

    def hessian(self) -> np.ndarray:
        """3x3 H with W = ½ vᵀ H v for v = (ε_xx, ε_yy, ε_xy)."""
        nu = self.poisson
        return self.young / (1.0 - nu ** 2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 2.0 * (1.0 - nu)]])

    def energy_density(self, epsilon: np.ndarray) -> float:
        """W in J/m² for a 2x2 strain tensor (fractions)."""
        v = np.array([epsilon[0, 0], epsilon[1, 1], epsilon[0, 1]])
        return float(0.5 * v @ self.hessian() @ v)


def strain_energy(strains: list[np.ndarray], elastic: list[Optional[Elastic]]) -> Optional[float]:
    """Total elastic energy of all layers per area of the cell, in meV/Å².

    ``None`` if any layer has no elastic constants. Each layer stores W per area of that
    layer; all layers have the same (moiré cell) area, so the sum is per cell area.
    """
    total = 0.0
    for strain, constants in zip(strains, elastic):
        if constants is None:
            return None
        total += constants.energy_density(strain)
    return float(total * J_PER_M2_IN_MEV_PER_A2)


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
    elastic_lower: Optional[Elastic] = None,
    elastic_upper: Optional[Elastic] = None,
) -> StrainFit:
    """Solve for the strains (and twist correction) that make a cell commensurate.

    - ``lower_cell``: 2x2, rows = the cell vectors ``A n``, ``A n'`` (lower frame).
    - ``upper_lattice``: 2x2, rows = the upper lattice vectors in the upper layer's own frame.
    - ``upper_coeffs``: 2x2 integers, rows = ``m``, ``m'``.
    - ``theta_deg``: rotation of the upper lattice in the lower frame (the scan angle).
    - ``limit_lower``, ``limit_upper``: strain limits as fractions (0 = rigid layer).
    - ``elastic_lower``, ``elastic_upper``: optional :class:`Elastic` constants. If both
      layers can strain and both have constants, the mismatch is shared so that the
      total elastic energy is minimal (the stiffer layer strains less); otherwise in
      proportion to the limits. The limits are checked afterwards either way.

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

    # Sharing penalty: among the exact solutions, minimise either the elastic energy
    # sum(½ pᵀ H p) (both layers with constants) or sum(|p|² / limit). Written as
    # residuals L p with LᵀL = the weight matrix, scaled so only the *ratio* matters.
    energy_pair = None
    if strain_lower and strain_upper and elastic_lower is not None and elastic_upper is not None:
        energy_pair = (elastic_lower, elastic_upper)
    weights = []
    for strainable, limit, constants in ((strain_lower, limit_lower, elastic_lower),
                                         (strain_upper, limit_upper, elastic_upper)):
        if not strainable:
            continue
        if energy_pair is not None and constants is not None:
            projector = _PARAM_TO_VOIGT[mode]
            reference = 0.5 * (energy_pair[0].young + energy_pair[1].young)
            matrix = projector.T @ constants.hessian() @ projector / reference
        else:
            matrix = np.eye(k) / limit
        weights.append(np.linalg.cholesky(matrix).T)  # upper triangular: |L p|² = pᵀ M p

    def residuals(x: np.ndarray) -> np.ndarray:
        lower, upper = cells(x)
        mismatch = ((lower - upper) / scale).ravel()
        penalty = [_SHARING_WEIGHT * (weight @ x[i * k:(i + 1) * k]) for i, weight in enumerate(weights)]
        return np.concatenate([mismatch, *penalty])

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
