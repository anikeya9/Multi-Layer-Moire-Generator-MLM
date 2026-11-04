"""Generate notebooks/moire_stack_explorer.ipynb (run from the repository root).

The notebook is the hands-on way to try MLM: the user lists the layers (POSCAR files)
and a twist range for every pair of neighbouring layers, then runs all cells.
Keeping the cells here as code makes the notebook easy to regenerate after API changes:

    python dev/make_moire_stack_explorer.py
    cd notebooks && jupyter nbconvert --to notebook --execute --inplace moire_stack_explorer.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md("""# MLM — moiré stack explorer

Build commensurate supercells of a twisted multilayer in four steps:

1. **Describe your stack** in the cell *"1. Your stack"*: the layers from bottom to top
   (POSCAR files) and a twist range for every pair of neighbouring layers.
2. **Run all cells** (*Run → Run All Cells*).
3. **Look at the results table** and pick a structure.
4. **Write it** as a VASP and/or LAMMPS file.

That is the only cell you need to edit. *Advanced settings* has the numerical knobs if
you need them.

**Angle convention.** A twist is the anticlockwise rotation of a layer relative to the
layer **directly below** it. For three layers with twists 22.62° and 36.87°, the absolute
orientations are 0°, 22.62° and 22.62 + 36.87 = 59.49°.

The **worked example** at the end (a PTO / STO / PTO heterostructure) runs on its own
and does not use your input.""")

code("""import os
os.environ["TQDM_DISABLE"] = "1"   # hide match.py's progress bars

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MLM.structure import MoireStructure, read_primitive
from MLM.search import SearchParams, SearchSchedule, Filters, build_stack, plan_rounds, job_to_spec_dict
from MLM.index import find, get, lineage, load_index
from MLM.spec import run_stack_spec
from MLM.build import build_atoms, write_structure


def find_repo_root(start=Path.cwd()):
    \"\"\"The folder that contains moire_structures/ (the notebook may run from notebooks/).\"\"\"
    for folder in [start, *start.parents]:
        if (folder / "moire_structures").is_dir():
            return folder
    raise FileNotFoundError("Run this notebook from inside the MLM repository.")


REPO = find_repo_root()
print("repository:", REPO.resolve())""")

md("""## Available layer structures

Primitive cells in `moire_structures/` that can be used as layers (files with at most 30
atoms; the larger files are finished moiré structures). The path column is what goes into
`LAYERS` below. You can also use your own POSCAR: in-plane vectors **a**, **b**,
**c** along z, Cartesian coordinates.""")

code("""rows = []
for path in sorted((REPO / "moire_structures").rglob("*.vasp")):
    try:
        primitive = read_primitive(path)
    except Exception:
        continue  # not a primitive layer POSCAR in the expected format
    if len(primitive["coords"]) > 30:
        continue
    a, b = primitive["in_plane"]
    symbols, counts = np.unique(primitive["atom_types"], return_counts=True)
    rows.append({
        "path": path.relative_to(REPO).as_posix(),
        "formula": "".join(f"{s}{c}" for s, c in zip(symbols, counts)),
        "|a| (Å)": round(float(np.linalg.norm(a)), 4),
        "|b| (Å)": round(float(np.linalg.norm(b)), 4),
        "γ (deg)": round(float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))), 2),
    })
pd.set_option("display.max_colwidth", 80)
pd.DataFrame(rows)""")

md("""## 1. Your stack  ✏️

Edit this cell, then run all cells.

- `LAYERS`: one entry per layer, **bottom to top**: (POSCAR path relative to the repository
  root, a short label such as `"bto"`). Repeat an entry for a homo-multilayer.
- `TWIST_RANGES`: one `(min, max)` range in degrees **per interface**, bottom to top.
  Entry *k* is the twist of layer *k+1* relative to layer *k*, so there is one entry fewer
  than layers. Narrow ranges are fast; see the time estimate in step 2.
- `CELL_ANGLES`: the angle between the two moiré cell vectors you want: `90` for
  square/rectangular cells, `60` or `120` for hexagonal ones. You can list several.
- `MAX_RESIDUAL`, `MAX_LAYER_RESIDUAL`: how far from perfectly commensurate a cell may be,
  in Å. The first checks each combination step; the second checks the final cell against
  every layer (safer for heterostructures). `None` switches a check off.""")

code("""# ========================== EDIT THIS CELL ==========================
RUN_NAME = "my_stack"            # results go to runs/<RUN_NAME>/

LAYERS = [                       # bottom -> top: (POSCAR, label)
    ("moire_structures/bto/bto_1-layer.vasp", "bto"),
    ("moire_structures/bto/bto_1-layer.vasp", "bto"),
    ("moire_structures/bto/bto_1-layer.vasp", "bto"),
]

TWIST_RANGES = [                 # degrees, one per interface, bottom -> top
    (22.0, 23.0),                #   layer 1 -> layer 2
    (36.0, 38.0),                #   layer 2 -> layer 3
]

CELL_ANGLES = [90.0]             # 90 = square/rectangular cell, 60/120 = hexagonal

MAX_RESIDUAL = 1e-3              # Å, error of each combination step (None = off)
MAX_LAYER_RESIDUAL = 1e-3        # Å, error of the final cell vs every layer (None = off)
# =====================================================================""")

md("""### Advanced settings

Usually fine as they are.

- `ANGLE_STEP`: grid spacing of the twist sweep.
- `MATCH_TOLERANCE`: tolerance on the fractional coordinates when matching lattice points
  (`dtol` in `match.py`). Raise it, e.g. to `9e-3`, for lattice-mismatched
  heterostructures.
- `N_RANGE_*`: how many lattice vectors are tried in each direction. Round 1 combines
  primitive cells; later rounds combine moiré cells, which needs a larger range.""")

code("""ANGLE_STEP = 0.01                # degrees
MATCH_TOLERANCE = 5e-4           # dimensionless (match.py's dtol)
N_RANGE_FIRST_ROUND = 10         # coefficients -10 ... 10 in round 1
N_RANGE_LATER_ROUNDS = 40        # coefficients -40 ... 40 in rounds 2, 3, ...
OVERWRITE_EXISTING_RUN = True    # delete runs/<RUN_NAME>/ before searching
TESTS_PER_SECOND = 4e5           # rough speed of match.py on this machine, for the time estimate""")

md("""## 2. Check the input and plan the search

Nothing is searched yet. This step reads the layers, checks the twist ranges, and shows
how the layers will be combined. MLM pairs neighbouring blocks in rounds (a tournament
tree): with three layers, round 1 joins layers 1 + 2, and round 2 adds layer 3 to that
bilayer. Each twist range is used in the round that joins its two layers.""")

code("""def check_inputs():
    problems = []
    if len(LAYERS) < 2:
        problems.append("LAYERS needs at least two layers.")
    if len(TWIST_RANGES) != len(LAYERS) - 1:
        problems.append(f"TWIST_RANGES needs {len(LAYERS) - 1} entries (one per interface), "
                        f"got {len(TWIST_RANGES)}.")
    for k, window in enumerate(TWIST_RANGES, start=1):
        if len(window) != 2 or not window[0] < window[1]:
            problems.append(f"TWIST_RANGES[{k - 1}] (layer {k} -> {k + 1}) must be (min, max) "
                            f"with min < max, got {window}.")
    for position, (poscar, label) in enumerate(LAYERS, start=1):
        if not (REPO / poscar).exists() and not Path(poscar).exists():
            problems.append(f"layer {position}: file not found: {poscar}")
    if problems:
        raise ValueError("Please fix the input cell:\\n  - " + "\\n  - ".join(problems))


check_inputs()
roots = [MoireStructure.from_poscar(poscar, label, search_dirs=[REPO]) for poscar, label in LAYERS]

layer_table = []
for position, ((poscar, label), root) in enumerate(zip(LAYERS, roots), start=1):
    primitive = read_primitive(poscar, [REPO])
    a, b = root.lattice_vectors
    layer_table.append({"layer": position, "label": label, "POSCAR": poscar,
                        "atoms/cell": len(primitive["coords"]),
                        "|a| (Å)": round(float(np.linalg.norm(a)), 4),
                        "|b| (Å)": round(float(np.linalg.norm(b)), 4),
                        "γ (deg)": round(root.angle_between_vectors, 2)})
pd.DataFrame(layer_table)""")

code("""# Search parameters: defaults, a larger n range from round 2 on, and your twist range per interface.
default = SearchParams(degmin=TWIST_RANGES[0][0], degmax=TWIST_RANGES[0][1], degstep=ANGLE_STEP,
                       dtol=MATCH_TOLERANCE, nmin=-N_RANGE_FIRST_ROUND, nmax=N_RANGE_FIRST_ROUND,
                       goodangles=tuple(CELL_ANGLES))
later_rounds = {r: {"nmin": -N_RANGE_LATER_ROUNDS, "nmax": N_RANGE_LATER_ROUNDS}
                for r in range(2, len(LAYERS) + 1)}
per_interface = {k: {"degmin": float(low), "degmax": float(high)}
                 for k, (low, high) in enumerate(TWIST_RANGES, start=1)}
schedule = SearchSchedule(default, later_rounds, per_interface)
filters = Filters(max_residual=MAX_RESIDUAL, max_layer_residual=MAX_LAYER_RESIDUAL)

plan_rows = []
for round_number, steps in enumerate(plan_rounds(len(LAYERS)), start=1):
    for step in steps:
        if step.is_carry:
            plan_rows.append({"round": round_number, "combination": str(step), "interface": "—",
                              "twist range (deg)": "—", "n range": "—", "≈ s per search": "—"})
            continue
        params = schedule.params_for_step(round_number, step)
        k = step.interface
        plan_rows.append({"round": round_number, "combination": str(step),
                          "interface": f"layer {k} → {k + 1}",
                          "twist range (deg)": f"{params.degmin:g} – {params.degmax:g}",
                          "n range": f"±{params.nmax}",
                          "≈ s per search": round(params.estimated_cost() / TESTS_PER_SECOND, 1)})
pd.DataFrame(plan_rows)""")

md("""**How long will it take?** Round 1 runs one search per pair of layers. Every later
combination runs one search **per candidate** of the blocks it joins: if round 1 finds 5
bilayers, round 2 runs 5 searches. The estimate above is per search; widen the ranges
step by step.""")

md("""## 3. Run the search

Every candidate, including rejected ones, is saved as a small YAML recipe in
`runs/<RUN_NAME>/yamls/`, with one summary table, `index.csv`. Only candidates that pass
the residual limits go on to the next round.""")

code("""run_dir = REPO / "runs" / RUN_NAME
if run_dir.exists() and OVERWRITE_EXISTING_RUN:
    shutil.rmtree(run_dir)

# The saved stack_spec.yaml uses repository-relative paths, so it can be re-run with
# `python -m MLM run runs/<RUN_NAME>/stack_spec.yaml` from the repository root.
spec = job_to_spec_dict(RUN_NAME, roots, schedule, filters, output_dir=f"runs/{RUN_NAME}")
result = build_stack(roots, schedule, filters, run_dir=run_dir, search_dirs=[REPO], spec=spec, log=print)
print()
print(result.summary())""")

md("""## 4. Results

The full-stack candidates that passed the limits, smallest cell (fewest atoms) first:

- the twist of every interface;
- `residual`: the error of the last combination step;
- `max layer residual`: the error of the cell measured against every layer;
- the cell size and the number of atoms it will contain.""")

code("""def candidate_table(records):
    rows = []
    for record in records:
        rotations = record.layer_rotations
        row = {"id": record.id}
        for k in range(1, record.n_layers):
            row[f"twist {k}→{k + 1} (deg)"] = round(rotations[k] - rotations[k - 1], 3)
        n_atoms = 0
        for poscar in record.source_poscar:
            primitive = read_primitive(poscar, [REPO])
            primitive_area = abs(np.linalg.det(primitive["in_plane"]))
            n_atoms += int(round(record.cell_area / primitive_area)) * len(primitive["coords"])
        a1, a2 = record.lattice_vectors
        row.update({"cell angle (deg)": round(record.angle_between_vectors, 2),
                    "|A1| (Å)": round(float(np.linalg.norm(a1)), 3),
                    "|A2| (Å)": round(float(np.linalg.norm(a2)), 3),
                    "atoms": n_atoms,
                    "residual (Å)": record.residual,
                    "max layer residual (Å)": record.worst_layer_residual})
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(["atoms", "max layer residual (Å)"]).reset_index(drop=True) if rows else table


results = candidate_table(result.final)
if results.empty:
    print("No full-stack cell passed the limits. Things to try:")
    print("  - widen TWIST_RANGES (one interface at a time),")
    print("  - raise MAX_RESIDUAL / MAX_LAYER_RESIDUAL, or MATCH_TOLERANCE for heterostructures,")
    print("  - check CELL_ANGLES (90 for square, 60/120 for hexagonal lattices),")
    print("  - look at the rejected candidates in the overview below.")
results if not results.empty else None""")

md("""Everything that was found, all levels (bilayers, …), passed or not:""")

code("""overview = load_index(run_dir)
overview.groupby(["n_layers", "passed_filter"]).size().rename("candidates").to_frame()""")

md("""## 5. Look at one structure

`CHOICE` is the row number in the results table (0 = smallest cell).""")

code("""CHOICE = 0

chosen = None
if results.empty:
    print("No structure to show (see the hints in step 4).")
else:
    chosen_id = results.loc[CHOICE, "id"]
    chosen = get(run_dir, chosen_id)
    print("id:", chosen_id)
    print("absolute layer orientations (deg):", np.round(chosen.layer_rotations, 3))
    print("moiré cell vectors (Å):\\n", chosen.lattice_vectors.round(4))
    print()
    print("lineage (how it was built):")
    print(lineage(run_dir, chosen_id))""")

code("""def plot_top_view(record, built):
    fig, ax = plt.subplots(figsize=(6, 6))
    for layer, group in built.atoms.groupby("layer"):
        ax.scatter(group["x"], group["y"], s=4, alpha=0.6,
                   label=f"layer {layer + 1}: {record.material[layer]} at {record.layer_rotations[layer]:.2f}°")
    a1, a2 = record.lattice_vectors
    outline = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
    ax.plot(outline[:, 0], outline[:, 1], "k-", lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title("top view of the moiré cell", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, markerscale=3)
    plt.show()


if chosen is not None:
    built = build_atoms(chosen, search_dirs=[REPO])
    print("atoms per layer:", built.layer_counts, "(expected", built.expected_counts, ")")
    print("composition:", built.composition)
    plot_top_view(chosen, built)""")

md("""## 6. Write the structure

`INTERLAYER_GAP` is the distance between the highest atom of one layer and the lowest
possible position of the next one; `VACUUM` is added on top of the stack along c.""")

code("""FORMATS = ["vasp", "lammps"]     # any of "vasp", "lammps"
INTERLAYER_GAP = 3.0             # Å
VACUUM = 50.0                    # Å

if chosen is not None:
    for fmt in FORMATS:
        path = write_structure(run_dir, chosen.id, fmt=fmt, interlayer_gap=INTERLAYER_GAP, vacuum=VACUUM)
        print(f"{fmt:7s} -> {path.relative_to(REPO)}")""")

md("""## Worked example: a heterostructure (PTO / STO / PTO)

This section is independent of your input above. It runs the stack spec
`examples/pto_sto_pto_demo.yaml` and writes to `runs/pto_sto_pto_demo/`.

PbTiO₃ (PTO, a = 3.880 Å) and SrTiO₃ (STO, a = 3.913 Å) are both square, with a 0.85 %
lattice mismatch, so they are never exactly commensurate. What is different from a
homostructure:

- **Two materials:** `material` and `source_poscar` follow the stacking order; the ids
  read `pto_sto_…` and `pto_sto_pto_…`.
- **A looser tolerance in round 1** (`dtol = 9e-3`, as in
  `notebooks/Bi-layer_structure_finder.ipynb`); with the homostructure default 5e-4 the
  PTO/STO pair has no match at all.
- **Residuals compound.** The trilayer cell is a large multiple of an imperfect bilayer cell,
  so its error against the STO layer can be much larger than the error of the last step.
  The spec therefore filters on `max_layer_residual`, the error measured against every layer.""")

code("""HETERO_SPEC = REPO / "examples" / "pto_sto_pto_demo.yaml"
print(HETERO_SPEC.read_text())""")

code("""hetero = run_stack_spec(HETERO_SPEC, base_dir=REPO, log=None)
hetero_dir = hetero.run_dir
print(hetero.summary())""")

md("""### Round 1: the PTO / STO bilayers

These are the same twists and residuals as the old bilayer notebook (`tblstopto.pkl`),
restricted to the 28–30° window used here:""")

code("""reference = pd.read_pickle(REPO / "moire_structures" / "PTO_STO" / "tblstopto.pkl")
in_window = reference[(reference["Theta"] >= 28.0) & (reference["Theta"] < 30.0)]
old = pd.DataFrame({"twist (deg)": in_window["Theta"].round(2), "old residual (Å)": in_window["delvec"]})
new_bilayers = find(hetero_dir, n_layers=2)
new = pd.DataFrame({"twist (deg)": new_bilayers["twist"].round(2), "new residual (Å)": new_bilayers["residual"],
                    "id": new_bilayers["id"]})
old.merge(new, on="twist (deg)", how="outer").sort_values("twist (deg)").reset_index(drop=True)""")

md("""### Round 2: the trilayers, and why `max_layer_residual` matters

Every trilayer found, passed or not. Compare two errors:
- `residual`: the error of the last step, bilayer cell + layer 3. It is small, because the
  bilayer cell is treated as perfect.
- `max layer residual`: the error of the final cell against every layer. It includes the
  STO mismatch inherited from round 1, multiplied up with the cell.

Even `cumulative_residual` (the max over all steps) underestimates it.""")

code("""all_trilayers = [get(hetero_dir, structure_id) for structure_id in find(hetero_dir, n_layers=3)["id"]]
trilayer_table = candidate_table(all_trilayers)
trilayer_table["passed"] = [get(hetero_dir, i).worst_layer_residual <= 5e-2 for i in trilayer_table["id"]]
worst = np.array([r.worst_layer_residual for r in all_trilayers])
step = np.array([r.residual for r in all_trilayers])
cumulative = np.array([r.cumulative_residual for r in all_trilayers])
print(f"{len(trilayer_table)} trilayer candidates, {int(trilayer_table['passed'].sum())} pass max_layer_residual <= 0.05 Å")
print(f"max layer residual / last-step residual:   median {np.median(worst / step):.0f}x")
print(f"max layer residual / cumulative_residual:   median {np.median(worst / cumulative):.1f}x, "
      f"max {np.max(worst / cumulative):.1f}x")
trilayer_table.head(10)""")

md("""### The smallest trilayer that passes

Rebuilt from its recipe and written as VASP and LAMMPS. The atom count of every layer
(PTO, STO, PTO) matches (cell area / primitive area) × atoms per cell.""")

code("""passing = candidate_table(hetero.final)
hetero_choice = get(hetero_dir, passing.loc[0, "id"])
print("id:", hetero_choice.id)
print("materials (bottom -> top):", hetero_choice.material)
print("absolute layer orientations (deg):", np.round(hetero_choice.layer_rotations, 3))
print("per-layer residuals (Å):", np.round(hetero_choice.layer_residuals, 4))
print()
print(lineage(hetero_dir, hetero_choice.id))
hetero_built = build_atoms(hetero_choice, search_dirs=[REPO])
print("atoms per layer:", hetero_built.layer_counts, "(expected", hetero_built.expected_counts, ")")
print("composition:", hetero_built.composition)
plot_top_view(hetero_choice, hetero_built)
for fmt in ["vasp", "lammps"]:
    print(f"{fmt:7s} -> {write_structure(hetero_dir, hetero_choice.id, fmt=fmt).relative_to(REPO)}")""")

md("""To try your own heterostructure, put two different POSCARs in `LAYERS` (section 1),
and raise `MATCH_TOLERANCE` (advanced settings) if nothing is found.""")

md("""## Coming back later

Everything is stored in `runs/<RUN_NAME>/`, so you can come back to any candidate without
searching again:

```python
from MLM.index import find, get, lineage
from MLM.build import write_structure

run_dir = "runs/my_stack"
find(run_dir, n_layers=3, max_layer_residual=1e-3)    # filter the index table
write_structure(run_dir, "<id>", fmt="vasp")         # rebuild any candidate
```

The job itself is saved as `runs/<RUN_NAME>/stack_spec.yaml`. From the repository root,
`python -m MLM run runs/<RUN_NAME>/stack_spec.yaml` repeats it from the command line.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/moire_stack_explorer.ipynb")
print("wrote notebooks/moire_stack_explorer.ipynb")
