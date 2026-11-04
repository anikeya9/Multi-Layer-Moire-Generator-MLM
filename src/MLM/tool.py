"""Small legacy helpers, kept for the notebooks. New code should use MLM.io / MLM.angle_between."""

import numpy as np

from MLM.angle_between import angle_between
from MLM.io import read_vasp


def load_layer_basis(vaspfile):
    """2x2 matrix whose columns are the in-plane lattice vectors a, b of a POSCAR."""
    lattice, _, _ = read_vasp(vaspfile)
    return np.array([lattice["a"][:2], lattice["b"][:2]]).T


def angle_in_degrees(vec1, vec2):
    """Angle between two vectors in degrees (same as MLM.angle_between.angle_between)."""
    return angle_between(np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float))
