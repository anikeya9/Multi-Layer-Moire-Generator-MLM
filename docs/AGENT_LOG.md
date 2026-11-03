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

## 2026-11-03 — M2 `search_pair`, bilayer YAMLs, index

**Done**
- `src/MLM/search.py`: `SearchParams`, `Filters`, `CellMatch`, `cell_from_pair`,
  `deduplicate` (HNF-based), `layer_residuals`, `record_from_cell`, `find_candidates`
  (all cells, no residual filter) and `search_pair` (= `find_candidates` + `Filters`).
  `match.py` is called unchanged via `run_and_filter`.
- `src/MLM/index.py`: `RunStore` (writes YAMLs + index rows), `write_index`,
  `rebuild_index`, `load_index`, `find`, `get`, `lineage` (`LineageNode` tree).

**Reproduction (bilayer)**
- BTO, 1–45°, ±10, goodangle 90°: raw `run_and_filter` returns 490 pairs; after
  de-duplication **19 cells, whose twists are exactly the 19 `TBLBTO_<angle>_2.dat`
  files below 45°** (6.73 … 43.60). 7.2 s.
- BTO 22.62° vs `bto_bilayer_candidates.csv`: same twist (22.62), same lattice (HNF
  equal), residual 3.391e-5 Å vs 3.391e-5 Å. The chosen basis
  `[[-11.971, 7.981], [-7.981, -11.971]]` is identical to the cell of
  `moire_strs_vasp/bto_22.62.vasp`.
- Graphene vs `graphene_bilayer_candidates.csv` (goodangle 60°): same lattice; the new
  best grid angle is 21.79° with 3.66e-4 Å (old O(N⁴) finder on a 0.05° grid: 21.80°,
  1.51e-3 Å).

**Tests:** 27 passed in ~16 s (`test_structure`, `test_search_pair`, `test_index`,
`test_reproduction`).

**Decisions (agent-decided):** de-duplication of equivalent matches (same lattice via
HNF / supercells / neighbouring grid angles) on by default (`dedupe=True`); residual of
a cell = max(delvec of both vectors); cells made right-handed; `layer_residuals`
added (true per-layer error of the cell).

**Next:** M3 — planner + `build_stack` + trilayer reproduction.

## 2026-11-03 — M3 planner, `build_stack`, queries, trilayer reproduction

**Done**
- `search.py`: `Step`, `plan_rounds` (pure), `format_plan`, `SearchSchedule`
  (`default` + per-round overrides, `notebook_defaults()`), `build_stack` (tree driver:
  carry-forward, per-round params, filters, `dry_run`, unimplemented
  `max_candidates` hook), `StepReport`/`StackResult` (counts and timings per step).
- Every candidate (rejected ones too) is written to `yamls/` and `index.csv`; the index is
  rewritten after every step so an interrupted run keeps its progress.

**Reproduction (trilayer)**
- BTO: bilayer 22.62° + layer 3, round 2 window 36.5–37.5°, ±40 → one cell at
  **36.87°, residual 5.7472e-5 Å; old CSV best row: 36.87°, 5.7472e-5 Å**, same lattice
  (HNF), area 1035.0 Å² = 65 primitive cells. `max_layer_residual` = 7.58e-5 Å (the
  cell's true error w.r.t. layer 2, larger than the max-rule `cumulative_residual`).
- Graphene: bilayer 21.79° + layer 3 at 21.79° (aligned with layer 2), same lattice as
  `graphene_trilayer_candidates_21.80.csv` (|A| = 11.307 Å, 3× the bilayer cell),
  residual 6.3e-4 Å (old: 2.6e-3 Å at the 0.05° grid point 21.80°).

**Tests:** 46 passed in 22 s.

**Next:** M4 — stack spec loader and `write_structure`.

## 2026-11-03 — M4 stack spec + `write_structure`

**Done**
- `src/MLM/spec.py`: `parse_stack_spec`/`load_stack_spec` (strict key validation,
  per-round param merging, string round keys accepted), `run_stack_spec(path,
  base_dir, dry_run)` — one call from spec to YAMLs + index; the spec file is copied
  verbatim to `<run>/stack_spec.yaml`.
- `src/MLM/build.py`: `rotations_from_lineage` (walks the lineage; checked against the
  stored `layer_rotations`), `build_atoms` (replicate → stack 3 Å above previous top →
  rotate → `select_atoms_fractional` with `dtol = max(layer_residuals)`), and
  `write_structure(run_dir, id, fmt="vasp"|"lammps", out)` using the existing
  `structure_writer.write_vasp_ase`/`write_lammps_ase`. Per-layer atom counts are checked
  against area ratio × atoms per cell (warning if off).
- Replication patch sized from the cell (`replicas_needed`) instead of the notebooks'
  `8 * (|A1|+|A2|)/|a|`; same folded result, far fewer atoms.
- `examples/bto_trilayer_demo.yaml`.

**Reproduction (structures)** — all built through `write_structure`:
| structure | atoms (new / old) | composition | cell | per-layer in-plane max displacement |
|---|---|---|---|---|
| BTO bilayer 22.62° (from `bto_2-layer.vasp`) vs `moire_strs_vasp/bto_22.62.vasp` | 312 / 312 | Ba78 O182 Ti52 both | identical | 6.8e-6, 1.1e-4 Å |
| STO bilayer 22.62° vs `STO_moire_cubic/sto_cubic_22.62.vasp` | 182 / 182 | O104 Sr52 Ti26 both | identical | 6.6e-6, 1.1e-4 Å |
| BTO trilayer 22.62°/36.87° vs `bto_22.62_36.86.vasp` | 1365 / 1365 | Ba390 O780 Ti195 both | same lattice, other basis | 3e-14, 2.7e-4, **2.29 Å** |

The trilayer's layer 3 differs on purpose: the old file has layer 3 at
22.62 + 36.86 = 59.48° (φ applied relative to layer 2, at the 36.86° grid point), while
`trilayer_structure_creator.ipynb` and the new tree rotate layer 3 by φ relative to
layer 1 (36.87°). Rebuilding the same cell with the old rotation gives ≤ 0.017 Å on all
layers. Both are commensurate (layer residual 5.7e-5 Å at 36.87°; 5.5e-3 Å at 59.48°,
1.3e-4 Å at 59.49°). The z positions differ because the old files used a different
interlayer offset (10.0 Å / 11.2 Å shifts) than the notebook's "3 Å above the top atom"
rule that `build_atoms` follows (`interlayer_gap` is a parameter).

**Tests:** 69 passed in 32 s.

**Next:** M5 demo notebook.
