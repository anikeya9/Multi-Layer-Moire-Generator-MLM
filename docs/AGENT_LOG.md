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

## 2026-11-03 — M5 demo notebook

**Done**
- `dev/yaml_multilayer_demo.ipynb`, executed with outputs saved (nbconvert):
  spec → dry run → `run_stack_spec` → `index.csv` → `find` → `get` → recipe file →
  `lineage` → `write_structure` (VASP + LAMMPS) → atom-count check → top-view plot.
  Runs in ~10 s (windows 20–24° and 35–38°): 1 bilayer, 3 trilayer candidates
  (35.30°, 36.04°, 36.87°), 2 pass `max_residual = 1e-3`.
- `dev/make_yaml_multilayer_demo.py` regenerates the notebook (cells as code, so the
  notebook can be rebuilt after API changes).

**Tests:** 69 passed (unchanged).

**Next:** M6 docs.

## 2026-11-03 — S1–S3 (done before finishing M6, because the docs reference them)

**S3 CLI** — `src/MLM/cli.py` + `__main__.py`: `python -m MLM run|find|show|write`
(thin argparse wrappers). `tests/test_cli.py`.

**S1 examples** — `examples/bto_4layer_demo.yaml`, `bto_5layer_demo.yaml`,
`pto_sto_pto_demo.yaml`, executed in `dev/yaml_multilayer_stretch.ipynb`:
- 4 layers: (L1+L2),(L3+L4) → identical bilayer recipe stored once; round 2 (30–40°,
  ±10) → 4 cells, 2 pass; best 36.87°, rotations [0, 22.62, 36.87, 59.49].
- 5 layers: L5 carried through two rounds; round 3 (20–25°, ±40, 12.9 s) → 5 cells,
  2 pass `max_layer_residual ≤ 1e-3`; rebuilt 5-layer VASP with 910 atoms/layer exactly.
- PTO/STO bilayer (1–45°, ±10, dtol 9e-3) reproduces `PTO_STO/tblstopto.pkl` exactly:
  the same 6 twists (7.13, 17.74, 28.99, 29.74, 35.02, 40.36) and residuals
  (0.0238, 0.0288, 0.0083, 0.0238, 0.0081, 0.0288 Å).
- PTO/STO/PTO trilayer: 21 candidates, 5 pass. **Finding:** the per-step residual of the
  trilayer step is tiny (1e-4 Å) but the true error vs the STO layer is 0.02–0.3 Å
  (median 5× the `cumulative_residual`). Added `Filters.max_layer_residual`
  (agent-decided) so runs can filter on the true error.

**S2 benchmark** — `dev/benchmark_yaml_vs_index.py` (2000 synthetic trilayer recipes):
YAML 783 B/file (1.57 MB total), index 286 B/row (0.57 MB); filtering by reading all
YAMLs 4.95 s (2.5 ms/file) vs `load_index` 15 ms (~325×) vs `find` 26 ms.

Also committed a tiny example run, `examples/runs/bto_trilayer_demo/` (YAMLs + index +
spec, ~20 kB, no atom files); `python -m MLM write examples/runs/bto_trilayer_demo
<id>` works from the repo root.

**Tests:** 70 passed in 33 s.

## 2026-11-03 — M6 docs

**Done**
- `docs/IMPLEMENTATION_NOTES.md`: module map, API with examples, data-flow walkthrough,
  test + reproduction tables with numbers, failures/limitations, agent-decided items.
- `docs/DESIGN.md`: §2 no longer calls the search model / entry point undecided; new
  module table; schema example with the added fields; §5.1 added fields; §6 filename with
  hash; §7 concrete conventions (frame, twist, residual, dedupe, paths, stacking, index);
  §8 decisions #14–#31 marked agent-decided; §10 new open points.
- `tests/test_docstrings.py` runs the docstring examples of the new modules (3 examples
  were wrong and are fixed).
- Measured the PyYAML C loader for the notes: 6.6× faster than `safe_load` (not used).

**Tests:** 76 passed in 32 s.

**Next:** S4 polish pass.

## 2026-11-03 — S4 polish pass + wrap-up

**Done**
- Reviewed the diff vs `origin/dev`: `match.py`, `structure_writer.py`,
  `angle_between.py`, `moire_lattice_vector_finder.py`, `tool.py` and `notebooks/` are
  untouched (empty `git diff --stat`). pyflakes clean on all new modules and tests
  (one unused import removed). README gained a short section on the YAML workflow.
- The old notebooks were not re-executed (their full sweeps take tens of minutes); they
  only import unchanged modules.

**Tests:** 76 passed in 34 s. No xfails.

**Still blocked:** push to `origin/claude-edits` (HTTP 403 for the whole session). A git
bundle of the branch was produced as a fallback (see the final session message).

## 2026-11-04 — Owner review of the agent-decided items

The owner reviewed the decisions (DESIGN.md §8) and chose:
- **Twist reference → measured from the layer directly below** (new decision #32). A twist
  is now the rotation of the upper block's bottom layer relative to the lower block's top
  layer. `match.run` is unchanged: the search sweeps the window shifted by
  `lower.layer_rotations[-1]` and stores `twist = scan angle − shift`
  (`search.twist_offset`). `layer_rotations` and `build.rotations_from_lineage` use
  `lower[-1] + twist + r`. The `window_relative_to` option was removed (it is now the only
  behaviour; round-2 window 1–60.01° = the notebooks' θ+1 … θ+60.01).
- **Hash rounded values** (#18): floats are rounded to 8 decimals (and `-0.0` → `0.0`)
  before hashing; ids no longer depend on the angle grid origin (new test).
- Everything else kept: dedupe on, both residuals kept, max of the two delvecs, all four
  added fields, label rule, #25–#31.

**Effect on the reproduction**
- `bto_22.62_36.86.vasp` now matches on **all three layers** (3e-14, 2.7e-4, 7.5e-3 Å; the
  last is the 36.87° vs 36.86° grid step). Previously layer 3 was off by 2.29 Å.
- The old trilayer CSV (layer 3 at 36.87° from layer 1) is now found as twist 14.25°,
  residual 5.7472e-5 Å (unchanged); the graphene trilayer CSV as twist 0.00°.
- Example specs re-run; `bto_5layer_demo.yaml` windows changed to round 2 = 30–31°,
  round 3 = 35–40° (1 of 4 five-layer cells passes). Both notebooks re-executed; the
  committed example run `examples/runs/bto_trilayer_demo` regenerated (new ids).
- DESIGN.md (§5 example, §5.1, §6, §7, §8 #16/#18/#23/#32, §9, §10) and
  IMPLEMENTATION_NOTES.md updated.

**Tests:** 77 passed in 21 s.

## 2026-11-04 — Stack explorer notebook + per-interface twist windows

**Request:** a notebook where the user gives the input structures and the range of angles
between each pair of layers.

**Done**
- Since a twist is measured from the layer below, each merge of the tree sets exactly one
  interface. Added `Step.interface` and `SearchSchedule.interfaces` (overrides per
  interface k = layer k | layer k+1, applied after `default` and `rounds`);
  `build_stack` and the dry run resolve parameters per step (`params_for_step`); the spec
  loader accepts `search.interfaces`. 5 new tests.
- `notebooks/moire_stack_explorer.ipynb` (generated by `dev/make_moire_stack_explorer.py`,
  executed): list of available primitive POSCARs → one input cell (`LAYERS`,
  `TWIST_RANGES`, `CELL_ANGLES`, residual limits) → advanced settings → input check, layer
  table and plan with a time estimate per search → run → results table (twist per
  interface, cell size, atom count, residuals) → lineage and top-view plot of a chosen
  structure → VASP/LAMMPS output. The saved `stack_spec.yaml` re-runs with
  `python -m MLM run`.
- Checked with three inputs: BTO trilayer (default, ~8 s), graphene 4-layer with a different
  window per interface (1 cell, 392 atoms), and an empty result (hints, no errors).

**Tests:** 81 passed.

## 2026-11-04 — Heterostructure worked example in the explorer notebook

Added "Worked example: a heterostructure (PTO / STO / PTO)" at the end of
`notebooks/moire_stack_explorer.ipynb` (independent of the input cell). It runs
`examples/pto_sto_pto_demo.yaml` with `run_stack_spec` and shows: the round-1 bilayers
against the old `tblstopto.pkl` (28.99° 0.008259 Å, 29.74° 0.023813 Å — identical);
the 19 trilayers with step residual vs max layer residual (median 161× the last step,
5.0× the cumulative residual); and the smallest passing trilayer (29.74° / 7.13°, layers at
0 / 29.74 / 36.87°, 2716 atoms, 910/896/910 per layer as expected) written as VASP and
LAMMPS. Notebook executed, 0 errors.
