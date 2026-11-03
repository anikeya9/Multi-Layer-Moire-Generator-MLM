# MLM — Software Design Document

**Project:** MLM (Multi-Layer Moiré) — building commensurate supercells of twisted
multilayer 2D materials.
**Status:** living document. Captures the design of the YAML-driven, multi-layer
search/storage workflow.
**Last updated:** 2026-06-21 (branch `dev`).

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

**Undecided / future** (tracked in §9): the multi-layer search model (joint
N-layer vs. chained pairwise) and the primary entry point (CLI vs. Python API).

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
(e.g. `..._22.62_36.86`), but each *file* stores only its own `twist` relative to its
parent. The full sequence is recovered from the lineage, not duplicated in each file.

---

## 5. The uniform YAML schema

Every file — level 0 through N — carries the **same keys**, using `null` where a
field does not apply at that level. A single shape means the loader never has to
branch on level to know which keys exist.

```yaml
id: bto_3_22.62_36.86          # unique; matches filename stem
level: 2                       # 0 = root/monolayer, 1 = bilayer, ...
n_layers: 3                    # = level + 1 (stored explicitly for readability)
material: [bto, bto, bto]      # list of material labels, one per layer

parents: [bto_2_22.62, bto_1]  # parent ids; [] at level 0
twist: 36.86                   # this layer's twist vs its parent (deg); null at level 0

lattice_vectors:               # uniform key at every level (a1, a2 as rows)
  - [1.2336, -2.1367]
  - [1.2336,  2.1367]
angle_between_vectors: 60.0    # angle between a1 and a2 (deg)

residual: 5.0e-4               # commensurability error (delvec) at THIS step; null at level 0
cumulative_residual: 5.0e-4    # max residual over the whole lineage; null at level 0
source_poscar:                 # always a list, one entry per primitive layer
  - moire_structures/bto/bto_1-layer.vasp
units: angstrom
```

### Field semantics by level

| Field | Level 0 (root) | Level ≥ 1 |
|---|---|---|
| `parents` | `[]` | list of parent ids |
| `twist` | `null` | twist of this layer vs its parent |
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
- **`twist`** — degrees, this layer relative to its parent. `null` at level 0.
- **`lattice_vectors`** — 2×2, rows are `a1`, `a2`, in `units`.
- **`angle_between_vectors`** — degrees, between `a1` and `a2`.
- **`residual`** — commensurability error (`delvec`) at this combination step. `null` at level 0.
- **`cumulative_residual`** — worst residual over the entire lineage,
  `max(residual, parentA.cumulative_residual, parentB.cumulative_residual)`. The
  field to filter on for final structure quality. `null` at level 0.
- **`source_poscar`** — list of paths to primitive POSCAR(s). Homobilayer lists the
  same file twice (uniformity over brevity).
- **`units`** — length unit for `lattice_vectors` (e.g. `angstrom`).

---

## 6. Filename convention

```
<material>_<n_layers>_<angles>.yaml
```

- `<material>` — material labels joined with `_` (e.g. `bto`, `pto_sto`).
- `<n_layers>` — integer layer count.
- `<angles>` — the twist sequence, each rounded to **2 decimal places**, joined with
  `_`. Rounding is for the *name only*; the file keeps full precision.

Examples: `bto_1.yaml` (root), `bto_2_22.62.yaml` (bilayer),
`bto_3_22.62_36.86.yaml` (trilayer).

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
swept twist. A tree with N leaves has exactly **N−1** internal combinations, matching
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
- **Schema evolution:** filename scheme and possibly additional metadata fields may
  change as the workflow matures; the authoritative `id` lives in the file to insulate
  against renames.
