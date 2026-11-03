# Agent plan — YAML metadata layer + N-layer moiré search

Branch: `claude-edits` (created from `origin/dev`). If the session restarts, read this
file and `docs/AGENT_LOG.md` first, then resume at the first unticked item.

## Core milestones (task §6)

- [x] **M0 Setup** — branch, install, fix `requirements.txt`/`setup.py`, read code +
      notebooks, this plan, baseline `run_and_filter` timings.
- [x] **M1** `MoireStructure` record (`structure.py`) + level-0 YAML + tests.
- [ ] **M2** `search_pair` + bilayer YAMLs + index table + tests + bilayer reproduction.
- [ ] **M3** pairing planner + `build_stack` (carry-forward, per-round params, filters,
      dry-run) + query helpers (`load_index`/`find`/`get`/`lineage`) + tests +
      trilayer reproduction.
- [ ] **M4** stack-spec loader (`spec.py`) + `write_structure` (`build.py`, VASP and
      LAMMPS) + structure reproduction checks.
- [ ] **M5** executed demo notebook `dev/yaml_multilayer_demo.ipynb`.
- [ ] **M6** docs: `docs/IMPLEMENTATION_NOTES.md`, `docs/DESIGN.md` update.

## Stretch goals (task §7, in order)

- [ ] **S1** 4-layer and 5-layer example (carry-forward) + PTO/STO heterostructure example.
- [ ] **S2** benchmark: ~2000 YAML files vs `index.csv` (time, bytes per file).
- [ ] **S3** CLI: `python -m MLM run|find|write`.
- [ ] **S4** polish pass (diff review vs `dev`, dead code, docstrings with examples,
      full test run, notes update).

## Hard rules to keep checking

- `git diff origin/dev -- src/MLM/match.py` must stay empty.
- Every commit leaves `pytest` green (only documented `xfail`s); push after each commit.
- Test suite < ~2 min (narrow angle windows).
