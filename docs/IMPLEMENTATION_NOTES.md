# Implementation notes — YAML metadata layer and N-layer search

Handoff document for the owner. It describes what was built on branch `claude-edits`,
how the pieces fit together, how well the new path reproduces the old notebook results,
and everything that was decided without you (marked **agent-decided**).

Companion documents:
- `docs/DESIGN.md` — the design (updated: §2, §3, §5.1, §6, §7 conventions, §8 decisions 14–31, §10).
- `docs/AGENT_LOG.md` — chronological log of the session.
- `docs/AGENT_PLAN.md` — the milestone checklist.
- `dev/yaml_multilayer_demo.ipynb`, `dev/yaml_multilayer_stretch.ipynb` — executed demos.

---

## 1. Module map

```
src/MLM/
  match.py                     (unchanged)  numeric core: scan → run → filtermatches → run_and_filter
  structure_writer.py          (unchanged)  replicate_atoms, select_atoms_fractional, write_*_ase
  moire_lattice_vector_finder  (unchanged)  read_vasp is reused for POSCAR reading
  angle_between.py             (unchanged on this branch; dev already had the np.clip fix)
  ─── new ───────────────────────────────────────────────────────────────────────────
  structure.py   MoireStructure record: schema, YAML round trip, id + hash, POSCAR reading
  search.py      search_pair (wraps run_and_filter), de-duplication, planner, build_stack
  index.py       run directory store, index.csv, load_index / find / get / lineage
  spec.py        stack-spec loader, run_stack_spec
  build.py       atoms from a recipe's lineage, write_structure (VASP / LAMMPS)
  cli.py         python -m MLM run | find | show | write   (+ __main__.py)
tests/           76 tests, ~35 s
examples/        stack specs + one tiny committed run (examples/runs/bto_trilayer_demo)
dev/             demo notebooks (+ the scripts that generate them), YAML-vs-index benchmark
```

Why this split: each module has one job and the dependency arrows only point one way:

```
structure  ←  index  ←  search  ←  spec  ←  cli
    ↑           ↑                    ↑
    └───────────┴──────  build  ─────┘
```

`structure` knows nothing about searching; `search` knows nothing about YAML files on disk
except through `index.RunStore`; `build` only needs recipes and POSCARs. That makes every
layer testable on its own.

`match.py` is untouched: `git diff origin/dev -- src/MLM/match.py` is empty.

---

## 2. Public API, one example each

All examples assume the repository root as working directory.

### `MLM.structure`

```python
from MLM.structure import MoireStructure

root = MoireStructure.from_poscar("moire_structures/bto/bto_1-layer.vasp", "bto")   # level-0 record
path = root.to_yaml("runs/scratch/yamls")        # writes runs/scratch/yamls/<id>.yaml
same = MoireStructure.from_yaml(path)            # yaml.safe_load only
root.to_dict()                                   # plain-Python dict, schema order
root.angle_between_vectors, root.cell_area       # derived properties (90.0, 15.92)
```

### `MLM.search`

```python
from MLM.search import SearchParams, Filters, SearchSchedule, search_pair, find_candidates, build_stack, plan_rounds

params = SearchParams(degmin=22.0, degmax=23.0, degstep=0.01, dtol=5e-4, nmin=-10, nmax=10, goodangles=(90.0,))
bilayers = search_pair(root, root, params, Filters(max_residual=1e-3))   # → [MoireStructure(id='bto_2_22.62_…')]
everything = find_candidates(root, root, params)                         # same, without residual filters

plan_rounds(5)        # [[L1+L2, L3+L4, carry L5], [L1-2 + L3-4, carry L5], [L1-4 + L5]]

schedule = SearchSchedule(params, {2: {"degmin": 36.5, "degmax": 37.5, "nmin": -40, "nmax": 40}})
result = build_stack([root] * 3, schedule, Filters(max_residual=1e-3), run_dir="runs/bto3")
print(result.summary())
build_stack([root] * 5, schedule, dry_run=True)    # plan + search counts, nothing searched
```

### `MLM.index`

```python
from MLM.index import load_index, find, get, lineage

table = load_index("runs/bto3")                                   # pandas DataFrame, one row per recipe
hits = find("runs/bto3", n_layers=3, max_cumulative_residual=1e-3, angle_between_vectors=90.0, material="bto")
trilayer = get("runs/bto3", hits["id"].iloc[0])                  # → MoireStructure
print(lineage("runs/bto3", trilayer.id))                         # tree back to the roots
```

### `MLM.spec`

```python
from MLM.spec import load_stack_spec, run_stack_spec

spec = load_stack_spec("examples/bto_trilayer_demo.yaml")         # validated StackSpec
result = run_stack_spec("examples/bto_trilayer_demo.yaml")        # one call: roots → … → index.csv
run_stack_spec("examples/bto_trilayer_demo.yaml", base_dir="..")  # from a notebook in dev/
```

### `MLM.build`

```python
from MLM.build import write_structure, build_atoms

write_structure("runs/bto3", trilayer.id, fmt="vasp")      # runs/bto3/structures/<id>.vasp
write_structure("runs/bto3", trilayer.id, fmt="lammps", out="tri.data")
built = build_atoms(trilayer)                              # atoms DataFrame + cell, no file
built.layer_counts, built.expected_counts, built.composition
```

### CLI

```bash
python -m MLM run examples/bto_trilayer_demo.yaml
```
```bash
python -m MLM find runs/bto_trilayer_demo --n-layers 3 --passed yes
```
```bash
python -m MLM show runs/bto_trilayer_demo bto_3_22.62_36.87_c35ae7
```
```bash
python -m MLM write runs/bto_trilayer_demo bto_3_22.62_36.87_c35ae7 --fmt lammps
```

---

## 3. Data flow, step by step (for a learner)

Take the BTO trilayer job `examples/bto_trilayer_demo.yaml`.

1. **Spec → roots.** `spec.load_stack_spec` reads the YAML with `yaml.safe_load` and checks
   every key (a typo such as `nmaxx` is an error, not silently ignored). For each layer,
   `MoireStructure.from_poscar` reads the POSCAR (`read_vasp`, cached) and makes a level-0
   record: in-plane lattice vectors, `parents: []`, `twist: null`. Its id is
   `bto_1_<hash>`; the three identical layers have the same id.

2. **Plan.** `search.plan_rounds(3)` is a pure function: round 1 = (L1 + L2) and
   "carry L3"; round 2 = (L1-2) + L3. Each `Step` names two *spans* of layers.

3. **Round 1 = one pairwise search.** `find_candidates(L1, L2, params)` calls
   `match.run_and_filter(a1, a2, g1, g2, …)` with `a` = the lower block's lattice vectors
   and `g` = the upper block's. `match.scan` rotates `g` by every angle on the grid and
   records lattice points of `a` that are (almost) lattice points of the rotated `g`;
   `filtermatches` pairs those points into cells whose two vectors make a "good angle".

4. **Raw matches → cells.** `run_and_filter` reports the same cell many times (both
   orders, ± signs, integer multiples, neighbouring grid angles: 94 entries for one BTO
   angle). `cell_from_pair` makes each one right-handed; `deduplicate` groups them by the
   lattice they span (Hermite normal form of the integer coefficients), keeps the
   smallest cell per angle and the best grid angle per run of neighbouring angles.

5. **Cells → records.** `record_from_cell` builds the child: `parents = [lower.id,
   upper.id]`, lists concatenated lower-first, `twist`, the moiré `lattice_vectors`,
   `residual` (Å), `cumulative_residual = max(residual, parents')`, `layer_rotations`
   (upper block rotated rigidly by `twist`), `layer_residuals` (true error per layer).

6. **Store everything.** `build_stack` passes each candidate to `Filters.passes`, writes
   **every** candidate with `RunStore.add` (YAML in `yamls/`, a row in memory), and
   rewrites `index.csv` after each step. Passing candidates go to the next round.

7. **Round 2.** The bilayer's moiré cell is now the lower "lattice" and layer 3's primitive
   cell the upper one; the same `find_candidates` runs again. Number of searches =
   (bilayer candidates) × 1.

8. **Jump to a structure.** `find` filters `index.csv` (no YAML parsing), `get` opens one
   recipe via the index's `path` column (and checks the stored `id`), `lineage` follows
   `parents` recursively.

9. **Rebuild atoms.** `write_structure` → `rotations_from_lineage` (re-derives each layer's
   rotation from the tree and compares it with the stored `layer_rotations`) →
   `build_atoms`: replicate each primitive layer over a patch that covers the cell, shift
   it up (3 Å above the previous layer's top atom), rotate it, concatenate, fold into the
   moiré cell with `select_atoms_fractional`, and check the atom count per layer against
   (cell area / primitive area) × atoms per cell → `write_vasp_ase` / `write_lammps_ase`.

---

## 4. Tests and reproduction results

`python -m pytest` → **76 passed in ~35 s** (no xfails). `-m "not slow"` skips the one 7 s
full-window bilayer test.

| file | covers |
|---|---|
| `test_structure.py` | YAML round trip (incl. `null`s, NumPy scalars), uniform keys across levels, id format/uniqueness/determinism, validation |
| `test_search_pair.py` | `search_pair` on 22–23°, child fields, max rule for `cumulative_residual`, filters, window modes, HNF, dedupe |
| `test_index.py` | index columns/rows, `find` filters, `get`, `lineage` tree, `rebuild_index` |
| `test_build_stack.py` | planner for N = 1…7, schedule merging, dry run, `max_candidates` hook, trilayer tree, run layout, rejected candidates kept, 4 layers |
| `test_spec.py` | parsing, per-round merging, invalid specs, one-call run, dry run |
| `test_build.py` | lineage rotations, replication size, atom counts/composition/stacking, VASP + LAMMPS output |
| `test_cli.py` | `run`, `find`, `show`, `write` |
| `test_reproduction.py` | the comparisons below |
| `test_docstrings.py` | every non-skipped docstring example of the new modules |

### 4.1 Search results vs the old notebooks / CSVs

| case | parameters (new path) | old result | new result |
|---|---|---|---|
| BTO bilayer, all angles | 1–45°, step 0.01, dtol 5e-4, n ±10, 90° | 19 `TBLBTO_<θ>_2.dat` files below 45° | **the same 19 twists** (490 raw pairs → 19 cells) |
| BTO bilayer 22.62° | 22–23° | CSV: 22.62°, n = (3,−2),(−2,−3), vec_del 3.391e-5 Å | 22.62°, same lattice, **3.391e-5 Å**; cell identical to `moire_strs_vasp/bto_22.62.vasp` |
| graphene bilayer | 21.5–22°, 60° | CSV: 21.80°, 1.51e-3 Å (0.05° grid) | 21.79°, same lattice, 3.66e-4 Å |
| PTO/STO bilayer | 1–45°, dtol 9e-3, n ±10 | `tblstopto.pkl`: 7.13, 17.74, 28.99, 29.74, 35.02, 40.36° | **same 6 twists and residuals** (0.0238 … 0.0081 Å) |
| BTO trilayer (22.62° + L3) | round 2: 36.5–37.5°, n ±40 | CSV: 36.87°, vec_del 5.7472e-5 Å | **36.87°, 5.7472e-5 Å**, same lattice, 65 primitive cells |
| graphene trilayer (21.79° + L3) | round 2: 21.7–21.9°, n ±40 | CSV: 21.80°, 3× bilayer cell, 2.6e-3 Å | 21.79°, same 3× cell, 6.3e-4 Å |

Why the graphene numbers differ slightly: the CSVs came from the older O(N⁴) finder on a
0.05° grid with an absolute residual criterion; `match.py` scans a 0.01° grid, so the new
best grid point is closer to the exact commensurate angle.

### 4.2 Structures rebuilt through `write_structure` vs the old `.vasp` files

| structure | atoms new / old | composition | in-plane cell | max in-plane displacement per layer |
|---|---|---|---|---|
| BTO bilayer 22.62° (from `bto_2-layer.vasp`) vs `moire_strs_vasp/bto_22.62.vasp` | 312 / 312 | Ba78 O182 Ti52 | identical | 6.8e-6, 1.1e-4 Å |
| STO bilayer 22.62° vs `STO_moire_cubic/sto_cubic_22.62.vasp` | 182 / 182 | O104 Sr52 Ti26 | identical | 6.6e-6, 1.1e-4 Å |
| BTO trilayer vs `bto_22.62_36.86.vasp` | 1365 / 1365 | Ba390 O780 Ti195 | same lattice, different basis | 3e-14, 2.7e-4, **2.29 Å** |

(1e-4 Å is the 5-decimal precision of the old files.) z positions are not compared: the
old files used other interlayer offsets than the notebooks' "3 Å above the top atom"
rule, which `build_atoms` follows.

**The trilayer's layer 3 is a real difference, not a bug.** In `bto_22.62_36.86.vasp`
layer 3 is rotated by 22.62 + 36.86 = 59.48° (φ applied relative to layer 2, at the 36.86°
grid point). In `trilayer_candidate_finder.ipynb` / `trilayer_structure_creator.ipynb`, and
therefore in the new tree, φ is the rotation of layer 3 relative to layer 1 (36.87°). Both
are commensurate with the same cell (layer error 5.7e-5 Å at 36.87°, 5.5e-3 Å at 59.48°).
Rebuilding the new cell with the old rotation reproduces all three layers within 0.017 Å
(asserted in `test_trilayer_structure_matches_old_vasp`). **Please check which convention
the old file was meant to have.**

### 4.3 Timings

| what | time |
|---|---|
| bilayer, 1–45°, n ±10 | 7.2 s |
| one round-2 search (bilayer cell + layer), 1° window, n ±40 | 2.7 s |
| same, notebook window θ+1 … θ+60.01°, n ±40 | **153 s** (2052 raw pairs) |
| rebuild + write the 1365-atom trilayer | 0.12 s |

A full notebook-style BTO trilayer scan (19 bilayer candidates × ~150 s) would take about
50 minutes, so the committed examples use narrow windows. `SearchParams.estimated_cost()`
and the dry run make this visible before running.

### 4.4 Benchmark: 2000 YAML files vs `index.csv` (open question)

`python dev/benchmark_yaml_vs_index.py 2000` (synthetic trilayer recipes):

| | |
|---|---|
| YAML file size | 783 B mean, 1.57 MB total |
| `index.csv` | 0.57 MB (286 B/row) |
| filter by reading all 2000 YAMLs | 4.95 s (2.5 ms/file) |
| filter via `load_index` | 15 ms (~325× faster) |
| filter via `find` | 26 ms |

Conclusion: keeping one YAML per candidate is cheap on disk; the index is what makes
querying thousands of candidates instant. (PyYAML's C loader, `yaml.CSafeLoader`, reads
the same files 6.6× faster — measured on 2000 files: 4.0 s → 0.6 s — but that is still far
slower than one CSV; it was not used because the design mandates `yaml.safe_load`.)

---

## 5. What failed, was skipped, or behaves unexpectedly

- **Pushing failed for the whole session.** `git push` returns HTTP 403 ("Claude doesn't
  have GitHub access to anikeya9/Multi-Layer-Moire-Generator-MLM for your organization"),
  and the GitHub connector cannot create branches either (403 "Resource not accessible by
  integration"). All work is committed **locally** on `claude-edits`; see the final
  message for how to recover it.
- **The trilayer `.vasp` layer-3 orientation** differs from the old file (§4.2).
- **`match.scan` skips coefficients with a zero component** and keeps only the 25 shortest
  matches per angle. In round 2+ a cell equal to the lower block's cell (n = (1,0),(0,1))
  can therefore never be found (graphene trilayer: the smallest cell found is 3× the
  bilayer cell). `match.py` was not modified, so this stays.
- **`cumulative_residual` (max rule) underestimates.** BTO trilayer: 5.7e-5 vs 7.6e-5 Å
  true; PTO/STO/PTO: true error typically 5× larger (0.02–0.3 Å). Use
  `max_layer_residual` (index column and filter) for quality decisions.
- **Ids depend on the angle grid origin.** The hash covers full-precision floats, and
  `np.arange(1, 45, 0.01)` and `np.arange(22, 23, 0.01)` give 22.620000000000097 vs
  22.62000000000041, so the same physical cell found in two runs can get different hashes.
  Within one run ids are unique and deterministic.
- `read_vasp` ignores the POSCAR scale factor (line 2) and needs Cartesian coordinates;
  all repository POSCARs satisfy this. `MoireStructure.from_poscar` refuses cells whose
  a, b are not in-plane.
- Docstring examples that need a run directory are marked `# doctest: +SKIP`; all others
  are executed by `tests/test_docstrings.py`.
- Not done: no performance work on `match.py` (not allowed), no parallelism over the
  searches of a round (each search is independent, so this would be easy to add).

---

## 6. Agent-decided items (review these)

Full list with reasoning in `docs/DESIGN.md` §8 (#14–#31). The ones that matter most:

1. **De-duplication is on by default** (#21). Without it every cell appears dozens of
   times and round 2 explodes (94 "candidates" for one BTO angle). Only equivalent
   descriptions are merged (same lattice, supercells at the same twist, neighbouring grid
   angles), but it *is* a judgement call: `dedupe: false` / `keep_supercells: true` turn it off.
2. **`twist` of the upper block is relative to the lower block's frame**, i.e. the
   trilayer's twist is the notebook's φ (relative to layer 1), not relative to layer 2
   (§4.2 shows the old `.vasp` file used the other convention).
3. **Added fields** `twist_sequence`, `layer_rotations`, `layer_residuals`,
   `coefficients` (#14–#17), plus the filter `max_layer_residual`.
4. **Step residual = max(delvec of a1, a2)** (#20).
5. **Id = `<material>_<n>_<angles>_<sha1[:6] of all fields except id>`**; a homostructure
   collapses to one label, a heterostructure joins all labels (`pto_sto_pto`) (#18, #19).
6. **Paths are stored as written** and resolved against `base_dir` / cwd at search time and
   against the run directory's ancestors at build time (#25).
7. **Fold tolerance = max(layer_residuals)** and a minimal replication patch (#26, #27).
8. `window_relative_to: lower_top_layer` reproduces the notebook's `degmin = θ + 1`
   window; the default `frame` uses the window as a plain relative twist (#23).
9. Index: one row per id; `round` is the first round that produced it (#28).

---

## 7. Limitations and next steps

Left alone on purpose (still open in DESIGN.md §10): heterostrain to drive residuals to
zero, a top-K beam cap (`max_candidates` raises `NotImplementedError`), tuning per-round
parameters, the final naming scheme.

Suggested next steps:
- Decide whether filters should default to `max_layer_residual` rather than
  `max_residual` / `cumulative_residual`.
- Parallelise the `lower × upper` searches of a step (`concurrent.futures`): they are
  independent, and the round-2 sweeps dominate the run time.
- Stable ids across runs: hash rounded values (e.g. twist to 1e-8°) instead of raw floats.
- Allow `n1 = 0` / `n2 = 0` coefficients in a copy of `scan` if the owner wants
  "the lower cell itself" as a candidate in later rounds.
- Store the interlayer gap / vacuum used for a written structure next to the file.
