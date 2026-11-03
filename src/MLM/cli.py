"""Command-line interface: ``python -m MLM <command> ...``.

Commands
--------
``run``    run a stack spec              ``python -m MLM run examples/bto_trilayer_demo.yaml``
``find``   query a run's index table     ``python -m MLM find runs/bto_trilayer_demo --n-layers 3``
``show``   print a recipe and lineage    ``python -m MLM show runs/bto_trilayer_demo <id>``
``write``  rebuild atoms, write a file   ``python -m MLM write runs/bto_trilayer_demo <id> --fmt lammps``

Every command is a thin wrapper around the Python API (``MLM.spec``, ``MLM.index``,
``MLM.build``); see ``docs/IMPLEMENTATION_NOTES.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from MLM.build import FORMATS, write_structure
from MLM.index import find, get, lineage
from MLM.spec import run_stack_spec

#: Columns printed by ``find`` (the full table is in index.csv).
FIND_COLUMNS = ["id", "n_layers", "twist", "angle_between_vectors", "residual",
                "cumulative_residual", "max_layer_residual", "cell_area", "passed_filter"]


def _cmd_run(args: argparse.Namespace) -> int:
    result = run_stack_spec(args.spec, base_dir=args.base_dir, dry_run=args.dry_run,
                            log=None if args.quiet else print)
    if not args.dry_run:
        print()
        print(result.summary())
    return 0


def _cmd_find(args: argparse.Namespace) -> int:
    material = args.material
    if material is not None and "," in material:
        material = material.split(",")  # exact stacking order, e.g. pto,sto
    passed = {None: None, "yes": True, "no": False}[args.passed]
    table = find(args.run_dir, n_layers=args.n_layers, max_cumulative_residual=args.max_cumulative_residual,
                 angle_between_vectors=args.angle, material=material, passed_filter=passed,
                 max_cell_area=args.max_cell_area)
    with pd.option_context("display.max_rows", args.limit, "display.width", 200, "display.max_columns", 20):
        print(table[FIND_COLUMNS].head(args.limit).to_string(index=False))
    print(f"\n{len(table)} matching structure(s)")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    record = get(args.run_dir, args.id)
    import yaml

    print(yaml.safe_dump(record.to_dict(), sort_keys=False, default_flow_style=None))
    print("lineage:")
    print(lineage(args.run_dir, args.id).format(indent="  "))
    return 0


def _cmd_write(args: argparse.Namespace) -> int:
    path = write_structure(args.run_dir, args.id, fmt=args.fmt, out=args.out,
                           interlayer_gap=args.gap, vacuum=args.vacuum)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser (separate function so tests can inspect it)."""
    parser = argparse.ArgumentParser(prog="python -m MLM", description="Multi-layer moiré search and structure tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a stack spec (YAML job file)")
    run.add_argument("spec", type=Path)
    run.add_argument("--base-dir", type=Path, default=None,
                     help="directory relative paths in the spec start from (default: cwd)")
    run.add_argument("--dry-run", action="store_true", help="show the planned work without searching")
    run.add_argument("--quiet", action="store_true", help="no progress messages")
    run.set_defaults(handler=_cmd_run)

    query = commands.add_parser("find", help="filter a run's index.csv")
    query.add_argument("run_dir", type=Path)
    query.add_argument("--n-layers", type=int)
    query.add_argument("--max-cumulative-residual", type=float)
    query.add_argument("--angle", type=float, help="angle between the cell vectors (deg)")
    query.add_argument("--material", help="'bto' = only bto; 'pto,sto' = exact stacking order")
    query.add_argument("--passed", choices=["yes", "no"], help="only candidates that passed / failed the filters")
    query.add_argument("--max-cell-area", type=float)
    query.add_argument("--limit", type=int, default=50, help="max rows to print (default 50)")
    query.set_defaults(handler=_cmd_find)

    show = commands.add_parser("show", help="print one recipe and its lineage")
    show.add_argument("run_dir", type=Path)
    show.add_argument("id")
    show.set_defaults(handler=_cmd_show)

    write = commands.add_parser("write", help="rebuild atoms of one structure and write VASP/LAMMPS")
    write.add_argument("run_dir", type=Path)
    write.add_argument("id")
    write.add_argument("--fmt", choices=sorted(FORMATS), default="vasp")
    write.add_argument("--out", type=Path, default=None)
    write.add_argument("--gap", type=float, default=3.0, help="interlayer gap in Å (default 3)")
    write.add_argument("--vacuum", type=float, default=50.0, help="vacuum along c in Å (default 50)")
    write.set_defaults(handler=_cmd_write)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python -m MLM``; returns the exit code."""
    args = build_parser().parse_args(argv)
    return args.handler(args)
