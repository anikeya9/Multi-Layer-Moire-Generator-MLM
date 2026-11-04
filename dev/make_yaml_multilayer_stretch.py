import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))
md("""# More stacks: 4 and 5 layers, and a heterostructure

Companion to `yaml_multilayer_demo.ipynb`. Three stack specs from `examples/`:

| spec | what it exercises |
|---|---|
| `bto_4layer_demo.yaml` | two bilayer blocks combined as rigid units |
| `bto_5layer_demo.yaml` | carry-forward over two rounds (L5 joins in round 3) |
| `pto_sto_pto_demo.yaml` | a heterostructure, and why `max_layer_residual` matters |""")
code("""import os
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path
import numpy as np
from MLM.spec import run_stack_spec
from MLM.index import load_index, find, lineage
from MLM.build import write_structure, build_atoms
from MLM.index import get
REPO = Path("..")
COLUMNS = ["id", "n_layers", "twist", "residual", "cumulative_residual", "max_layer_residual", "cell_area", "passed_filter"]""")
md("""## 1. Four layers

Plan: round 1 = (L1 + L2), (L3 + L4); round 2 = (L1-2) + (L3-4). Both round-1 pairs give the
identical bilayer recipe, so it is stored once and appears twice as a parent.""")
code("""four = run_stack_spec(REPO / "examples/bto_4layer_demo.yaml", base_dir=REPO, log=None)
print(four.summary())""")
code("""best4 = find(four.run_dir, n_layers=4, passed_filter=True)
best4[COLUMNS]""")
code("""four_id = best4["id"].iloc[0]
print(lineage(four.run_dir, four_id))
print("layer rotations:", np.round(get(four.run_dir, four_id).layer_rotations, 3))""")
md("""The round-2 `twist` is measured from layer 2 (the top of the lower bilayer) to layer 3
(the bottom of the upper bilayer). Layers 3 and 4 keep their relative twist of 22.62° — the
upper bilayer was rotated as a block.""")
md("""## 2. Five layers (carry-forward)

L5 has no partner in rounds 1 and 2 and is carried forward unchanged.
The dry run shows the plan before anything is searched:""")
_ = None
code("""_ = run_stack_spec(REPO / "examples/bto_5layer_demo.yaml", base_dir=REPO, dry_run=True)""")
code("""five = run_stack_spec(REPO / "examples/bto_5layer_demo.yaml", base_dir=REPO, log=None)
print(five.summary())""")
code("""table5 = load_index(five.run_dir)
table5[COLUMNS]""")
md("""`passed_filter = False` rows are kept on disk. The spec filters on both the step
`residual` and `max_layer_residual` (the cell's error measured against every layer); this is
why each rejected candidate was dropped:""")
code("""from MLM.spec import load_stack_spec
filters5 = load_stack_spec(REPO / "examples/bto_5layer_demo.yaml").filters
for structure_id in table5.loc[~table5["passed_filter"], "id"]:
    print(structure_id, "->", "; ".join(filters5.reasons_rejected(get(five.run_dir, structure_id))))""")
code("""five_id = find(five.run_dir, n_layers=5, passed_filter=True)["id"].iloc[0]
print(lineage(five.run_dir, five_id))
path = write_structure(five.run_dir, five_id)
built = build_atoms(get(five.run_dir, five_id), search_dirs=[REPO])
print(path, "| atoms per layer:", built.layer_counts, "expected:", built.expected_counts)""")
md("""## 3. Heterostructure PTO / STO / PTO

PTO (a = 3.880 Å) and STO (a = 3.913 Å) are never exactly commensurate; the bilayer cells of
`Bi-layer_structure_finder.ipynb` (dtol 9e-3) have residuals of 0.008–0.03 Å.""")
code("""hetero = run_stack_spec(REPO / "examples/pto_sto_pto_demo.yaml", base_dir=REPO, log=None)
print(hetero.summary())""")
code("""bilayers = find(hetero.run_dir, n_layers=2)
bilayers[COLUMNS]""")
md("""The trilayer step residuals look harmless, but the cell's error with respect to the STO
layer (`max_layer_residual`) is several times larger (median ≈ 5× here), because the trilayer cell is a large
integer multiple of an already-imperfect bilayer cell. `cumulative_residual` (the agreed
"max over steps" rule) does not see this:""")
code("""tri = find(hetero.run_dir, n_layers=3)
print(f"{len(tri)} trilayer candidates, {tri['passed_filter'].sum()} pass max_layer_residual <= 0.05 Å")
ratio = tri["max_layer_residual"] / tri["cumulative_residual"]
print(f"max_layer_residual / cumulative_residual: median {ratio.median():.1f}, max {ratio.max():.0f}")
tri[tri["passed_filter"]].sort_values("cell_area")[COLUMNS].head(10)""")
code("""hetero_id = tri[tri["passed_filter"]].sort_values("cell_area")["id"].iloc[0]
print(lineage(hetero.run_dir, hetero_id))
record = get(hetero.run_dir, hetero_id)
print("materials:", record.material, " rotations:", np.round(record.layer_rotations, 2))
built = build_atoms(record, search_dirs=[REPO])
print("atoms per layer:", built.layer_counts, "expected:", built.expected_counts, "| composition:", built.composition)
print(write_structure(hetero.run_dir, hetero_id))""")
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "dev/yaml_multilayer_stretch.ipynb")
