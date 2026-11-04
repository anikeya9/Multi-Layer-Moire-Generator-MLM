# MLM — Software Design Document

**Project:** MLM (Multi-Layer Moiré) — building commensurate supercells of twisted
multilayer 2D materials.
**Status:** living document. Captures the design of the YAML-driven, multi-layer
search/storage workflow.
**Last updated:** 2026-11-04 (branch `claude-edits`; implementation of §5–§9 plus the owner's review of §8 — see
`docs/IMPLEMENTATION_NOTES.md`).

---

## 1. Motivation

Twisting two or more atomically thin layers produces *moiré* superlattices — the
substrate for flat bands, unconventional superconductivity, correlated insulators,
and ferroelectric polarization textures. Before such systems can be simulated with
DFT or MD, one must find a **commensurate supercell**: a finite, periodic box in
which the twisted pattern closes back on itself.

MLM finds these cells quickly, for any Bravais lattice, and — unlike existing
bilayer-focused tools — for **three or more layers** with independent twist angles.

The original workflow was notebook-driven and *chained* (bilayer → trilayer), with
results stored as opaque Python pickles (`.pkl`). This document defines a
**structured, human-readable metadata layer based on YAML** that:

1. tracks the properties of every structure (lattice vectors, angles, layer count,
   twists, provenance), and
2. supports building up multi-layer stacks level by level, with each structure
   referencing the structures it was derived from.

---

## 2. Goals and scope

**In scope**

- Use **VASP POSCAR** files as the input format for *layer primitives* (the atom
  basis and primitive lattice vectors of each material).
- Introduce **one YAML file per structure** to store structure *metadata* (not atom
  coordinates).
- Encode a **lineage / provenance tree**: roots (single layers) → bilayers →
  trilayers → … Each structure references its parents.
- Keep a **single uniform schema** across all levels so the data is trivial to read
  back.

**Out of scope (for the YAML layer)**

- Storing atom coordinates. Atoms live in POSCAR / `.dat` files; YAML stores only a
  pointer to the primitive POSCAR(s). The YAML is a *recipe*, the atom file is the
  *dish*.

**Decided** (§8 #12, #13): the multi-layer search is a **hierarchical tournament tree
of pairwise searches with block-rigid twists** (§9), and the primary entry point is the
**Python API** (`MLM.search.build_stack`), with a thin **YAML stack-spec** loader
(`MLM.spec.run_stack_spec`) and a small CLI (`python -m MLM`) on top.

---

## 3. Architecture overview

```
   VASP POSCAR (per material)         <-- input: atom basis + primitive lattice
            │
            ▼
   ┌─────────────────────┐
   │  Level-0 reader      │  POSCAR  -> level-0 YAML (root)
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │  Coincidence search  │  match.py: solve-and-round bilayer search
   │  (match.py et al.)   │  -> moiré lattice vectors, twist, residual
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │  YAML writer         │  emit level-N YAML referencing parents
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │  Structure writer    │  read YAML + lineage -> follow source_poscar(s)
   │  (structure_writer)  │  -> build & export real atom file (VASP / LAMMPS)
   └─────────────────────┘
```

The YAML layer sits **between** the search and the structure writer. The search
*produces* YAML; the structure writer *consumes* it. Atom coordinates are
reconstructed on demand from the primitive POSCAR(s) named in the lineage.

### Existing core modules (`src/MLM/`)

| Module | Role |
|---|---|
| `match.py` | Solve-and-round bilayer coincidence search (`scan`→`run`→`filtermatches`→`run_and_filter`). O(N²). |
| `moire_lattice_vector_finder.py` | Older O(N⁴) Numba + multiprocessing brute force; has `read_vasp`. |
| `structure_writer.py` | `replicate_atoms`, `select_atoms_fractional` (fractional-fold dedup), VASP/LAMMPS export via ASE. |
| `tool.py` | `load_layer_basis(vaspfile)`, `angle_in_degrees`. |
| `angle_between.py` | Angle helper. |

### New modules (YAML layer)

| Module | Role |
|---|---|
| `structure.py` | `MoireStructure` record: schema, YAML round trip, id/hash, POSCAR reading (cached). |
| `search.py` | `search_pair` (wraps `match.run_and_filter`), de-duplication, pairing planner, `build_stack` tree driver. |
| `index.py` | Run directory store, `index.csv`, query helpers `load_index` / `find` / `get` / `lineage`. |
| `spec.py` | Stack-spec loader and `run_stack_spec`. |
| `build.py` | Rebuild atoms from a recipe's lineage; `write_structure` (VASP / LAMMPS). |
| `cli.py`, `__main__.py` | `python -m MLM run|find|show|write`. |

---

## 4. The lineage (level) model

Each structure has a **level**, equal to (number of layers − 1):

| Level | Layers | Meaning |
|---|---|---|
| 0 | 1 | Root / monolayer, read straight from a POSCAR |
| 1 | 2 | Bilayer moiré cell (output of one coincidence search) |
| 2 | 3 | Trilayer — the bilayer cell used as the reference lattice for a second search against layer 3 |
| N | N+1 | … and so on |

Provenance is encoded by **reference, not by embedding**: a child stores the **ids**
of its parents, never a copy of their data. The full stack (all layers, all twists)
is reconstructed by walking the `parents` links back to the roots. This is the
*normalized* choice — small files, no duplicated data, a single source of truth for
every fact.

**Consequence — angles:** the *filename* carries the whole twist sequence
(e.g. `..._22.62_36.87_<hash>`), and each file's `twist` is its own step relative to its
lower parent. The implementation additionally stores `twist_sequence` and
`layer_rotations` (§5.1) so the id and the atom rebuild do not depend on reading other
files; the lineage remains the source of truth and is cross-checked at build time.

---

## 5. The uniform YAML schema

Every file — level 0 through N — carries the **same keys**, using `null` where a
field does not apply at that level. A single shape means the loader never has to
branch on level to know which keys exist.

```yaml
id: bto_3_22.62_36.87_015688   # unique; matches filename stem
level: 2                       # 0 = root/monolayer, 1 = bilayer, ...
n_layers: 3                    # = level + 1 (stored explicitly for readability)
material: [bto, bto, bto]      # list of material labels, one per layer

parents: [bto_2_22.62_1fdb6f, bto_1_406932]  # [lower, upper] parent ids; [] at level 0
twist: 36.87                   # upper block vs the layer directly below it (deg); null at level 0

lattice_vectors:               # uniform key at every level (a1, a2 as rows)
  - [1.2336, -2.1367]
  - [1.2336,  2.1367]
angle_between_vectors: 60.0    # angle between a1 and a2 (deg)

residual: 5.0e-4               # commensurability error (delvec) at THIS step; null at level 0
cumulative_residual: 5.0e-4    # max residual over the whole lineage; null at level 0
source_poscar:                 # always a list, one entry per primitive layer
  - moire_structures/bto/bto_1-layer.vasp
  - moire_structures/bto/bto_1-layer.vasp
  - moire_structures/bto/bto_1-layer.vasp
units: angstrom
# --- added fields (§5.1) ---
twist_sequence: [22.62, 36.87]           # twists along the lineage (post-order); [] at level 0
layer_rotations: [0.0, 22.62, 59.49]     # absolute rotation of each layer in this frame (deg): 22.62 + 36.87
layer_residuals: [1.8e-15, 7.6e-05, 1.3e-04]  # true error of this cell vs each layer (Å); null at level 0
coefficients:                            # integer coordinates of a1, a2; null at level 0
  lower: [[1, 2], [-2, 1]]               #   in the lower parent's lattice
  upper: [[-7, 4], [-4, -7]]             #   in the (rotated) upper parent's lattice
```

### Field semantics by level

| Field | Level 0 (root) | Level ≥ 1 |
|---|---|---|
| `parents` | `[]` | list of parent ids |
| `twist` | `null` | rotation of the upper parent (as a rigid block) vs the layer directly below it, i.e. the lower parent's top layer |
| `lattice_vectors` | primitive cell (from POSCAR) | moiré commensurate cell |
| `angle_between_vectors` | angle of the primitive cell | angle of the moiré cell |
| `residual` | `null` | `delvec` from the coincidence search at this step |
| `cumulative_residual` | `null` | `max(this residual, parents' cumulative_residual)` |
| `source_poscar` | the input POSCAR (list of 1) | primitive POSCAR(s) the stack is built from |

### Field reference

- **`id`** — unique identifier; equals the filename stem.
- **`level`** — integer, `n_layers − 1`.
- **`n_layers`** — stored explicitly even though derivable, for human readability.
- **`material`** — list of labels, one per layer. Homostructures repeat the label;
  heterostructures (e.g. PTO/STO) list each. Joined with `_` to form the filename
  material field.
- **`parents`** — list of parent **ids**. Homobilayer lists the same root id twice.
- **`twist`** — degrees, anticlockwise rotation of the upper parent block relative to the
  layer directly below it (the lower parent's top layer). `null` at level 0.
- **`lattice_vectors`** — 2×2, rows are `a1`, `a2`, in `units`.
- **`angle_between_vectors`** — degrees, between `a1` and `a2`.
- **`residual`** — commensurability error (`delvec`) at this combination step. `null` at level 0.
- **`cumulative_residual`** — worst residual over the entire lineage,
  `max(residual, parentA.cumulative_residual, parentB.cumulative_residual)`. The
  field to filter on for final structure quality. `null` at level 0.
- **`source_poscar`** — list of paths to primitive POSCAR(s). Homobilayer lists the
  same file twice (uniformity over brevity).
- **`units`** — length unit for `lattice_vectors` (e.g. `angstrom`).

### 5.1 Added fields (implementation)

These were added uniformly at every level (agent-decided — needs review, §8 #14–#17):

| Field | Level 0 | Level ≥ 1 | Why |
|---|---|---|---|
| `twist_sequence` | `[]` | `lower.twist_sequence + upper.twist_sequence + [twist]` | The id's `<angles>` part must be computable from a record alone, without parsing parent filenames. |
| `layer_rotations` | `[0.0]` | `lower.layer_rotations + [lower.layer_rotations[-1] + twist + r for r in upper.layer_rotations]` | Absolute orientation of every layer, needed to rebuild atoms; also the physically readable angles. `build.write_structure` re-derives it from the lineage and refuses to build if they disagree. |
| `layer_residuals` | `null` | per layer: max over a1, a2 of the distance from the vector to the nearest point of that layer's rotated primitive lattice (Å) | `cumulative_residual` (max over steps) can **under**-estimate the real error, because a later round's cell is an integer combination of an earlier cell and the error grows with the coefficients. This is the direct measurement. Used as the fold tolerance when rebuilding atoms. |
| `coefficients` | `null` | `{lower: [[n1,n2],[n1',n2']], upper: [[m1,m2],[m1',m2']]}` | The integer solution behind the cell (from `match.scan`): lets anyone check commensurability by hand. |

`angle_between_vectors` is stored for readability but is derived: on load it is
recomputed from `lattice_vectors`.

---

## 6. Filename convention

```
<material>_<n_layers>[_<angles>]_<hash>.yaml
```

- `<material>` — a homostructure collapses to one label (`bto`); otherwise all labels
  joined with `_` in stacking order (`pto_sto`, `pto_sto_sto`; partial repeats are not
  collapsed).
- `<n_layers>` — integer layer count.
- `<angles>` — `twist_sequence`, each rounded to **2 decimal places**, joined with
  `_` (absent at level 0). Rounding is for the *name only*; the file keeps full precision.
- `<hash>` — first 6 hex characters of sha1 over the canonical record: the JSON of every
  field except `id`, keys sorted. Prevents collisions between matches that share a
  rounded angle; identical recipes get identical ids (so a root used for three layers is
  stored once).

Examples: `bto_1_406932.yaml` (root), `bto_2_22.62_1fdb6f.yaml` (bilayer),
`bto_3_22.62_36.87_015688.yaml` (trilayer).

Before hashing, every float is rounded to 8 decimals (1e-8 Å / 1e-8°; `-0.0` → `0.0`), so
the id does not depend on floating-point noise from the angle grid: the same cell found with
`np.arange(1, 45, 0.01)` (twist 22.620000000000097) and with `np.arange(22, 23, 0.01)`
(22.62000000000041) gets the same id (§8 #18). The file keeps full precision.

> The naming scheme is provisional and may be refined; the schema does not depend on
> it (the authoritative `id` lives inside the file).

---

## 7. Design rationale and conventions

- **Reference, not embed.** Children name parents by id. Small files, no duplication,
  single source of truth. Reading a deep stack means following links — an acceptable
  trade for normalization.
- **Metadata only, no atom coordinates.** YAML is text: verbose and slow for large
  numeric arrays, and float round-tripping is needless risk for DFT coordinates.
  Atoms stay in POSCAR/`.dat`; YAML points to them. The YAML is a recipe.
- **Uniform schema with `null`s.** One shape for all levels keeps the reader simple.
- **`null` for inapplicable fields** (`twist`, `residual` at level 0) rather than
  `0.0`, which is a real angle and could be mistaken for an aligned bilayer.
- **`safe_load` / `safe_dump` only.** Never plain `yaml.load`/`yaml.dump`, which can
  execute arbitrary embedded Python — a security footgun. `safe_*` restricts to basic
  types, which is all this schema needs.
- **NumPy must be converted before dumping.** `yaml.safe_dump` cannot serialize
  `np.ndarray`, `np.float64`, `np.int64`. Convert first:
  `arr.tolist()`, `float(x)`, `int(x)`. Always hand `safe_dump` plain
  `dict`/`list`/`float`/`int`/`str`.

### Concrete conventions (implementation)

- **Frame.** Every record lives in the frame of its bottom layer (unrotated there).
  `lattice_vectors` and `layer_rotations` are in that frame. `match.scan` rotates the
  *upper* lattice anticlockwise by `theta` and reports `matchpoint = A @ n` in the lower
  lattice's frame, so a child record inherits the lower parent's frame.
- **Twist** (owner decision, §8 #32). `twist` = anticlockwise rotation (deg) of the upper
  block, as a rigid unit, measured from the **layer directly below it** (the lower block's
  top layer). For a trilayer it is layer 3 relative to layer 2; layer 3's absolute angle is
  `layer_rotations[2] = θ₁₂ + twist`. `match.scan` still rotates relative to the lower
  block's frame, so the search sweeps `[degmin, degmax) + lower.layer_rotations[-1]` and
  stores `twist = scan angle − lower.layer_rotations[-1]`. A round-2 window of 1–60.01° is
  therefore the trilayer notebook's θ+1 … θ+60.01 window. (The notebook's `Phi` is the
  absolute angle of layer 3, i.e. `Phi = θ₁₂ + twist`.)
- **Residual of a step** = max(`delvec` of a1, `delvec` of a2), in Å (the notebooks used
  the `delvec` of a1 only; for the symmetric cells found they are equal).
- **Cell orientation.** Cells are stored right-handed (`a1 × a2 > 0`).
- **Equivalent matches.** `match.run_and_filter` reports each cell many times (both
  orderings, sign flips, integer multiples, neighbouring grid angles). They are
  collapsed by `search.deduplicate` (same lattice via Hermite normal form; only the
  smallest cell per twist; one per run of consecutive grid angles). `dedupe: false` in the
  search params turns this off.
- **Paths.** `source_poscar` stores paths exactly as written in the stack spec (normally
  relative to the repository root). The spec's relative paths and `output_dir` are
  resolved against `base_dir` (default: cwd). When rebuilding atoms, relative paths are
  looked up in the run directory and its ancestors.
- **Stacking.** `build_atoms` uses the notebook rule: layer k+1 is shifted up by
  `interlayer_gap` (3 Å) + the highest z of layer k; c = (top − bottom) + 50 Å vacuum.
- **Index.** `index.csv` has one row per id with every scalar field, joined lists
  (`material` with `,`, `parents` with `;`), plus `max_layer_residual`, `cell_area`,
  `a1_length`, `a2_length`, `round`, `passed_filter`, `path` (relative to the run dir).

---

## 8. Decisions log

| # | Decision | Choice |
|---|---|---|
| 1 | Parents: reference vs embed | **Reference by id** |
| 2 | Atom coordinates in YAML? | **No** — pointer to POSCAR only |
| 3 | Do level-0 roots get a YAML? | **Yes** |
| 4 | Schema across levels | **Uniform** (same keys + `null`s) |
| 5 | `twist`/`residual` at level 0 | **`null`** |
| 6 | `source_poscar` | **Always a list** (homobilayer repeats the file) |
| 7 | `material` representation | **List** of labels; joined with `_` for filenames |
| 8 | Angle precision in filename | **2 dp** in name; full precision in file |
| 9 | Residual accumulation metric | **`cumulative_residual = max(...)`** over lineage |
| 10 | Beam cap (top-K per block) | **None for now** — explore the full landscape |
| 11 | `id` uniqueness | descriptive stem **+ short content hash** |
| 12 | Multi-layer DOF model | **block-rigid** twists (see §9) |
| 13 | Entry point | **Python API first**, thin YAML stack-spec wrapper later |

### Implementation decisions (agent-decided, reviewed by the owner on 2026-11-04)

Each was taken during the unattended implementation session with the simplest option
consistent with the design and the notebooks. The owner reviewed them on 2026-11-04:
**#32 (twist reference) was changed** to "measured from the layer below", **#18 (hash) was
changed** to hash rounded floats; all other items were accepted as they are (#16/#22 with
the filter choice left to the user per run, #21 dedupe on by default, #20 max of the two
delvecs, #14–#17 all four added fields kept, #19 label rule kept, #25–#31 kept).

| # | Decision | Choice | Reasoning |
|---|---|---|---|
| 14 | Store `twist_sequence` | **yes**, list at every level | id must be computable from a record without parsing parent filenames (§6 forbids relying on names). |
| 15 | Store `layer_rotations` | **yes** | needed to rebuild atoms; lineage-derived value is cross-checked at build time. |
| 16 | Store `layer_residuals` | **yes** (null at level 0) | the max rule for `cumulative_residual` under-estimates compound errors (BTO trilayer at twist 14.25°: 5.7e-5 vs 7.6e-5 Å). Owner: keep both; each run chooses which to filter on. |
| 17 | Store `coefficients` | **yes** (null at level 0) | makes the integer solution inspectable. |
| 18 | Id hash input | sha1 of JSON of all fields except `id`, **floats rounded to 8 decimals**, sorted keys | deterministic, content-addressed; identical recipes share one file; ids are stable across runs with different angle grids (owner decision 2026-11-04; before, raw floats were hashed). |
| 19 | Heterostructure label | join all labels, collapse only if all equal | simplest unambiguous rule (`pto_sto_sto`). |
| 20 | Step residual | max(delvec of both vectors) | conservative; equals the notebooks' value for symmetric cells. |
| 21 | De-duplicate raw matches | **on by default** (`dedupe`, `keep_supercells`, `angle_cluster_tol = 1.5·degstep`) | without it a single BTO angle yields 94 "candidates" that are one cell; the notebooks did the same reduction by hand (norm filter + 0.05° Theta clustering + min delvec). Only equivalent descriptions are merged. |
| 22 | Extra filter `max_length_mismatch` | available, off by default | the notebooks filtered `|norm_A1 − norm_A2| ≤ 0.1`. |
| 23 | Notebook-style window | *(superseded by #32)* there is no separate window mode any more: windows are always twists from the layer below, so `degmin: 1, degmax: 60.01` in round 2 is the notebook's θ+1 … θ+60.01 window. The `window_relative_to` parameter was removed. | one convention, one meaning of `degmin`/`degmax`. |
| 24 | `search_pair` vs filters | `find_candidates` returns all cells; `search_pair` = those passing `Filters`; the driver saves both | "nothing is deleted" and `passed_filter` need the rejected ones too. |
| 25 | Path resolution | store as written; resolve against `base_dir`/cwd at search time and against the run dir's ancestors at build time | keeps recipes readable and portable inside the repo. |
| 26 | Fold tolerance when rebuilding | max(`layer_residuals`) | the notebooks used the step's `delvec`; the per-layer max is the real image mismatch. |
| 27 | Replication patch | `ceil((|A1|+|A2|)/h_min) + 1` cells each way | covers the cell exactly; the notebooks' `×8` rule gives the same folded atoms with ~50× more atoms. |
| 28 | Index rows | one per id, first round it appeared in | identical recipes produced in two slots are one structure. |
| 29 | Dry run | exact search counts only where known (round 1); later rounds report `?` plus the per-search cost estimate | counts depend on the previous round's results; no guessing. |
| 30 | `max_candidates` | raises `NotImplementedError` | clear hook for a future top-K cap (decision 10). |
| 31 | Index / table library | pandas | the notebooks use pandas; `match.py`'s polars output is converted internally. |
| 32 | Twist reference | **measured from the layer directly below** (lower block's top layer) — owner decision 2026-11-04 | matches how a stack is described layer by layer; reproduces `bto_22.62_36.86.vasp` (layer 3 at 22.62 + 36.86°). Initially the agent used "relative to the lower block's bottom layer" (the notebooks' absolute `Phi`). |

---

## 9. Multi-layer search algorithm

The driver builds an N-layer commensurate cell by a **hierarchical (tournament-tree)
reduction** over pairwise coincidence searches.

### Reduction

- **Round 1:** pair adjacent layers in stacking order — (1,2), (3,4), … — and run a
  bilayer search on each pair → a set of bilayer moiré-cell candidates per pair.
- **Round r > 1:** treat each block's moiré lattice vectors as an effective lattice,
  pair adjacent blocks, and run the same search between them.
- **Repeat** until one cell spans all N layers.
- **Odd leftover:** if a round has an odd number of blocks, the unpaired block is
  **carried forward** and combined in a later round (block sizes may be uneven, e.g.
  an 8-layer block combined with a 2-layer block).
- Pairing is **always adjacent**, not configurable (for now).

`n_layers` and `level` of a merged cell come from summing the parents
(`n_layers = A.n_layers + B.n_layers`, `level = n_layers - 1`). This makes the tree
yield intermediate 2-, 4-, …-layer structures, all saved as reusable candidates.

### Twist model — block-rigid

When two blocks combine, the upper block is rotated as a **single rigid unit** by the
swept twist, measured from the layer directly below it (the lower block's top layer). A tree with N leaves has exactly **N−1** internal combinations, matching
the **N−1** independent relative orientations of an N-layer stack — so no DOF is lost;
the twists are simply organised hierarchically (block-relative) rather than per-layer.

### Commensurability is transitive

A super-cell commensurate with block A's moiré cell is transitively commensurate with
every layer inside A. Hence the final cell is commensurate with all N layers.
**Residuals compound**, tracked via `cumulative_residual` (§5).

### Filtered beam search (no cap)

Each pairwise search returns many candidates. Going forward, **all candidates that
pass the round's filter** advance — the filter is the throttle, not an arbitrary pick:

- **good-angle match** — the moiré basis vectors must sit at a target
  `angle_between_vectors` (e.g. 60/90/120°); handled by `match.run_and_filter`'s
  `goodangles`.
- **residual tolerance** — drop candidates above a residual threshold.

No top-K cap for now (decision 10): the goal is to explore the full landscape and
find as many structures as possible. A top-K knob may be added later if the candidate
count explodes; the code leaves room for it.

### Atomic operation

The whole driver is orchestration over one function:

> `search_pair(A, B, params, filters) -> list[record]` — take two structure records,
> sweep the relative twist via `match.run_and_filter`, and return each surviving
> commensurate moiré cell as a new uniform-schema record
> (`parents=[A.id, B.id]`, summed `n_layers`/`material`/`source_poscar`, the found
> `twist`, moiré `lattice_vectors`, `residual`, `cumulative_residual`).

### Two kinds of YAML

| Kind | Direction | Holds |
|---|---|---|
| **Stack spec** | input | the job: ordered POSCAR list, per-round search params, filter thresholds |
| **Structure recipe** | output | one candidate's metadata + lineage (the uniform schema of §5) |

The engine is the Python API (takes a layer list + params); the stack-spec YAML is a
thin wrapper read into that API — the reproducible record of a job (decision 13).

---

## 10. Open questions / future work

- **Heterostrain engineering (future):** allow slight stretch/compress of a layer to
  drive residual toward zero (esp. for heterostructures). This modifies the *matching
  criterion*, orthogonal to the tree; the `residual` field is the natural hook.
- **Per-round search params:** whether deeper rounds use coarser/finer sweeps than
  round 1 is still to be tuned empirically (decision deferred until we see what works).
- **Top-K beam cap:** intentionally omitted now (decision 10); may be needed for large
  N if candidate counts explode.
- **CLI:** a `mlm` command-line wrapper over the Python API may follow the stack-spec
  loader.
- **`cumulative_residual` semantics:** the agreed max rule under-estimates compounded
  errors; `layer_residuals` / `max_layer_residual` give the direct number. Decided
  (2026-11-04): keep both; each stack spec chooses which to filter on
  (`max_cumulative_residual` and/or `max_layer_residual`).
- **`match.scan` skips coefficients with a zero component** (`n1 == 0 or n2 == 0`) and
  keeps only the 25 shortest matches per angle. Cells such as "the bilayer cell itself"
  (`n = (1,0),(0,1)`) can therefore never be found in round 2; the graphene trilayer
  reference shows the consequence (a 3× cell). Left unchanged because `match.py` must not
  be modified.
- **Schema evolution:** filename scheme and possibly additional metadata fields may
  change as the workflow matures; the authoritative `id` lives in the file to insulate
  against renames.
