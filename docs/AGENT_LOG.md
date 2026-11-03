# Agent log

Running log of the unattended implementation session. Newest entries at the bottom.

## 2026-11-03 — M0 setup

**Done**
- Created `claude-edits` from `origin/dev` (the local `dev` checkout was one commit behind
  `origin/dev`; `origin/dev` holds `docs/DESIGN.md` and the `structure.py` skeleton).
- `requirements.txt`: removed `multiprocessing` (stdlib), added the real dependencies.
  `setup.py`: added `install_requires`. Added a minimal `pyproject.toml` because
  `pip install -e .` failed with the system (Debian) setuptools
  (`AttributeError: install_layout`); with a `[build-system]` table pip builds in an
  isolated env with a current setuptools and the install works.
- Untracked the committed `src/MLM/__pycache__/` and `src/MLM.egg-info/`; added `runs/`
  and `.pytest_cache/` to `.gitignore`.
- Read `match.py`, `structure_writer.py`, `moire_lattice_vector_finder.py`, `tool.py`,
  `angle_between.py` and the four notebooks.

**What `match.py` returns (important for everything that follows)**
- `scan(A, G, theta, ...)` rotates the *upper* lattice `G` by `theta` (anticlockwise)
  and looks for points `n` of the *lower* lattice `A` that are also (within `tol`, a
  tolerance on the fractional coordinates) points `m` of the rotated `G`.
  `matchpoint = A @ n` is expressed in the **lower lattice's (unrotated) frame**;
  `delvec = |R G m - A n|` is in Å. Coefficients with a zero component (`n1 == 0` or
  `n2 == 0`) are skipped by `scan`, and only the 25 shortest matches per angle are kept.
- `filtermatches` returns `[row_a, row_b, {"delgoodangle": ...}]` for every ordered pair
  of match points whose angle is `goodangle` — so both `(a, b)` and `(b, a)`, sign flips
  and integer multiples all appear. A single BTO angle (22.62°) gives 94 such pairs.
- In the trilayer notebook the third layer is searched with `a = (A1, A2)` of the bilayer
  and `g =` layer 3's primitive cell, so `Phi` is the rotation of layer 3 relative to
  layer 1 (the bilayer frame) — i.e. exactly the block-rigid "upper block rotates by the
  swept twist" model.

**Baseline timings** (`run_and_filter`, BTO primitive cell, `degstep=0.01`, `dtol=5e-4`):
| case | window | n range | time | pairs returned |
|---|---|---|---|---|
| bilayer | 22–23° | ±10 | 0.17 s | 94 (one angle, 22.62°) |
| bilayer | 1–45° | ±10 | 7.2 s | 490 (21 distinct angles) |
| trilayer (bilayer 22.62° cell + layer 3) | 36.5–37.5° | ±40 | 2.7 s | 112 (36.87°) |
| trilayer, notebook window (θ+1 … θ+60.01) | 23.62–82.63° | ±40 | see M2 entry | |

So a notebook-style 60° window at ±40 costs ~2.5–3 min per bilayer candidate; tests
use 1° windows.

**Tests:** none yet (no test suite existed).

**Next:** M1 — `MoireStructure`.

## 2026-11-03 — M1 `MoireStructure`

**Done**
- `src/MLM/structure.py` rewritten from the skeleton: a `@dataclass` with
  `from_poscar`, `from_dict`, `from_yaml`, `to_dict`, `to_yaml`, `content_hash`,
  `make_id`, and derived properties `angle_between_vectors` (reuses
  `MLM.angle_between.angle_between`, which already clips the cosine and returns a float),
  `cell_area`, `worst_layer_residual`.
- POSCAR reading reuses `moire_lattice_vector_finder.read_vasp` (wrapped in an
  `lru_cache`d `read_primitive`) — no new parser.
- Added fields (uniform at all levels): `twist_sequence`, `layer_rotations`,
  `layer_residuals`, `coefficients` (documented in DESIGN.md §5.1).
- `resolve_path()` resolves stored relative POSCAR paths (cwd, then given dirs).

**Tests:** `tests/test_structure.py` — 10 passed (round trip incl. nulls and NumPy
scalars, schema uniformity, id format and uniqueness, validation).

**Decisions (agent-decided, logged in DESIGN.md §8):** store `twist_sequence` so the
id can be generated from a record alone (no filename parsing); hash = sha1 of the JSON
of all fields except `id`; partial repeats in heterostructure labels are not collapsed
(`pto_sto_sto`).

**Push status:** `git push` is refused with HTTP 403 ("Claude doesn't have GitHub access
to anikeya9/Multi-Layer-Moire-Generator-MLM for your organization"), and the GitHub API
connector also gets 403 on branch creation. Commits are kept locally on
`claude-edits`; push is retried at every milestone.

**Next:** M2 — `search_pair`, bilayer YAMLs, index.
