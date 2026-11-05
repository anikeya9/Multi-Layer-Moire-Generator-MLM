import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))



md("""# YAML multilayer workflow — demo

This notebook walks through the new workflow end to end:

**stack spec → search (tournament tree) → `index.csv` → `find` → `lineage` → write a VASP file**

Every structure the search finds (monolayer roots, bilayers, trilayers, …) is saved as a
small YAML *recipe* — metadata only, no atoms. The atoms are rebuilt on demand from the
primitive POSCARs by following the recipe's `parents` back to the roots.

The notebook lives in `dev/`, while all paths in the spec are relative to the repository
root, so we pass `base_dir=".."` where needed.""")

code("""import os
os.environ["TQDM_DISABLE"] = "1"   # silence match.py's progress bars in this notebook

from pathlib import Path
import yaml
import numpy as np

from MLM.spec import load_stack_spec, run_stack_spec
from MLM.index import load_index, find, get, lineage
from MLM.build import write_structure, build_atoms

REPO = Path("..")""")

md("""## 1. The stack spec (the job)

A stack spec lists the layers bottom-up (POSCAR + material label), the search parameters
(a `default` block plus overrides per round) and the filters. This one is the BTO trilayer
of `notebooks/trilayer_candidate_finder.ipynb`, with narrow angle windows so it runs in seconds.""")

code("""spec_path = REPO / "examples" / "bto_trilayer_demo.yaml"
print(spec_path.read_text())""")

code("""spec = load_stack_spec(spec_path)
for number in (1, 2):
    print(f"round {number}:", spec.schedule.params_for_round(number))""")

md("""## 2. Dry run: what will be searched?

Three layers need two rounds: (L1 + L2) first while L3 is carried forward, then
(bilayer + L3). In round 2 the number of searches is (number of bilayer candidates) × 1,
which is only known after round 1 — the dry run says so instead of guessing.""")

code("""plan = run_stack_spec(spec_path, base_dir=REPO, dry_run=True)""")

md("""## 3. Run it

One call: builds the roots, runs both rounds, writes every candidate's YAML and the index.""")

code("""result = run_stack_spec(spec_path, base_dir=REPO, log=None)   # log=print shows progress live
run_dir = result.run_dir
print(result.summary())""")

code("""sorted(p.name for p in (run_dir / "yamls").glob("*.yaml"))""")

md("""## 4. The index table

`index.csv` has one row per recipe with all scalar fields, so we can filter without opening
any YAML file. `passed_filter` says whether the candidate went on to the next round
(rejected candidates are kept on disk too).""")

code("""table = load_index(run_dir)
table[["id", "n_layers", "twist", "residual", "cumulative_residual", "max_layer_residual",
       "cell_area", "round", "passed_filter"]]""")

md("""## 5. Jump to a structure with `find` and `get`

`find` filters the index (here: trilayers with a small cumulative residual), `get` loads the
full recipe of one id as a `MoireStructure`.""")

code("""trilayers = find(run_dir, n_layers=3, max_cumulative_residual=1e-3, material="bto")
trilayers[["id", "twist", "cumulative_residual", "cell_area"]]""")

code("""best_id = trilayers["id"].iloc[0]
trilayer = get(run_dir, best_id)
print(trilayer)
print("absolute layer rotations (deg):", np.round(trilayer.layer_rotations, 4))
print("moiré lattice vectors (Å):\\n", trilayer.lattice_vectors.round(4))""")

md("""This is the recipe file itself — note `parents` (ids, not copies). Every `twist` is
measured from the layer directly below: the trilayer's `twist` (36.87°) is layer 3 relative
to layer 2, and `layer_rotations` gives the absolute angles (0, 22.62, 22.62 + 36.87 = 59.49°).""")

code("""print((run_dir / "yamls" / f"{best_id}.yaml").read_text())""")

md("""## 6. Lineage

Following `parents` back to the level-0 roots. The same root appears three times because all
three layers come from the same POSCAR (identical recipe → identical id).""")

code("""tree = lineage(run_dir, best_id)
print(tree)""")

md("""## 7. Rebuild the atoms and write a VASP file

`write_structure` walks the lineage, rotates and stacks each primitive layer, folds the atoms
into the moiré cell and writes the file with ASE. The atom count per layer is checked against
(cell area / primitive area) × atoms per primitive cell.""")

code("""vasp_path = write_structure(run_dir, best_id, fmt="vasp")
lammps_path = write_structure(run_dir, best_id, fmt="lammps")
print(vasp_path)
print(lammps_path)
print("".join(vasp_path.read_text().splitlines(keepends=True)[:8]))""")

code("""built = build_atoms(trilayer, search_dirs=[REPO])
print("atoms per layer:", built.layer_counts, " expected:", built.expected_counts)
print("composition:", built.composition)""")

md("""A top view of the cell, one colour per layer (only Ti atoms, to keep it readable):""")

code("""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ti = built.atoms[built.atoms["type"] == "Ti"]
for layer, group in ti.groupby("layer"):
    angle = trilayer.layer_rotations[layer]
    ax.scatter(group["x"], group["y"], s=8, label=f"layer {layer + 1} ({angle:.2f}°)")
a1, a2 = trilayer.lattice_vectors
corners = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
ax.plot(corners[:, 0], corners[:, 1], "k-", lw=1)
ax.set_aspect("equal"); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(best_id, fontsize=9)
plt.show()""")

md("""## Where to go from here

- Widen the windows in the spec (round 1: `degmin: 1, degmax: 45`; round 2: `degmin: 1,
  degmax: 60.01`) to scan the whole landscape like the notebooks did. Because twists are
  measured from the layer below, round 2's 1–60.01° is the notebook's θ+1 … θ+60.01 window.
- `find(run_dir, ...)` accepts more filters: `angle_between_vectors`, `twist_range`,
  `max_cell_area`, `max_layer_residual`, `passed_filter`.
- See `docs/IMPLEMENTATION_NOTES.md` for the full API.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "dev/yaml_multilayer_demo.ipynb")
