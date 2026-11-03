"""Shared pytest fixtures.

All tests run with the repository root as the working directory, so the
relative POSCAR paths used throughout (``moire_structures/...``) resolve.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BTO_POSCAR = "moire_structures/bto/bto_1-layer.vasp"
GRAPHENE_POSCAR = "moire_structures/graphene/graphene_monolayer.vasp"
PTO_POSCAR = "moire_structures/PTO_STO/pto_tetra_monolayer.vasp"
STO_POSCAR = "moire_structures/PTO_STO/STO_centered_unit_cell_cubic_monolayer.vasp"


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def bto_root():
    from MLM.structure import MoireStructure

    return MoireStructure.from_poscar(BTO_POSCAR, "bto")
