"""Stack spec: the YAML *input* that describes one N-layer search job.

The Python API (``MLM.search.build_stack``) is the engine; this module is a thin
loader on top of it, and the spec file is the reproducible record of a job::

    name: bto_trilayer_demo
    output_dir: runs/bto_trilayer_demo
    layers:
      - {poscar: moire_structures/bto/bto_1-layer.vasp, material: bto}
      - {poscar: moire_structures/bto/bto_1-layer.vasp, material: bto}
      - {poscar: moire_structures/bto/bto_1-layer.vasp, material: bto}
    search:
      default: {degmin: 1.0, degmax: 45.0, degstep: 0.01, dtol: 5.0e-4,
                nmin: -10, nmax: 10, goodangles: [90.0]}
      rounds:
        2: {nmin: -40, nmax: 40}
    filters:
      max_residual: 1.0e-3
      max_cumulative_residual: null

Relative paths (``poscar`` entries and ``output_dir``) are resolved against
``base_dir`` (default: the current working directory), so the example above is
meant to be run from the repository root. The POSCAR paths are stored in the
structure recipes exactly as written in the spec.

Example
-------
>>> result = run_stack_spec("examples/bto_trilayer_demo.yaml")   # doctest: +SKIP
>>> print(result.summary())                                     # doctest: +SKIP
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from MLM.index import SPEC_FILENAME
from MLM.search import Filters, SearchParams, SearchSchedule, StackResult, build_stack
from MLM.structure import MoireStructure, PathLike

TOP_LEVEL_KEYS = {"name", "output_dir", "layers", "search", "filters"}
LAYER_KEYS = {"poscar", "material"}
SEARCH_KEYS = {"default", "rounds"}


@dataclass
class LayerSpec:
    """One layer of the stack: a primitive POSCAR and its material label."""

    poscar: str
    material: str


@dataclass
class StackSpec:
    """A parsed stack spec. Build it with :func:`load_stack_spec` or :func:`parse_stack_spec`."""

    name: str
    output_dir: str
    layers: list[LayerSpec]
    schedule: SearchSchedule
    filters: Filters
    source_path: Optional[Path] = None

    def roots(self, base_dir: PathLike = ".") -> list[MoireStructure]:
        """Level-0 records for the layers (paths kept as written in the spec)."""
        return [
            MoireStructure.from_poscar(layer.poscar, layer.material, search_dirs=[base_dir])
            for layer in self.layers
        ]

    def run_dir(self, base_dir: PathLike = ".") -> Path:
        """``output_dir`` resolved against ``base_dir``."""
        output = Path(self.output_dir)
        return output if output.is_absolute() else Path(base_dir) / output


def _check_keys(section: str, data: dict[str, Any], allowed: set[str], required: set[str] = frozenset()) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"stack spec: unknown key(s) in {section}: {sorted(unknown)} (allowed: {sorted(allowed)})")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"stack spec: missing key(s) in {section}: {sorted(missing)}")


def parse_search(data: Optional[dict[str, Any]]) -> SearchSchedule:
    """``search:`` section → :class:`~MLM.search.SearchSchedule`.

    Missing ``default`` keys fall back to :class:`~MLM.search.SearchParams`'
    defaults; ``rounds`` maps a round number to the keys that differ in that round.

    >>> parse_search({"default": {"nmin": -10, "nmax": 10}, "rounds": {2: {"nmax": 40, "nmin": -40}}}).params_for_round(2).nmax
    40
    """
    data = data or {}
    _check_keys("search", data, SEARCH_KEYS)
    default_data = dict(data.get("default") or {})
    allowed = {f.name for f in fields(SearchParams)}
    _check_keys("search.default", default_data, allowed)
    rounds = {int(number): dict(overrides or {}) for number, overrides in (data.get("rounds") or {}).items()}
    for number, overrides in rounds.items():
        _check_keys(f"search.rounds.{number}", overrides, allowed)
    return SearchSchedule(SearchParams(**default_data), rounds)


def parse_filters(data: Optional[dict[str, Any]]) -> Filters:
    """``filters:`` section → :class:`~MLM.search.Filters` (``null`` = not applied)."""
    data = data or {}
    _check_keys("filters", data, {f.name for f in fields(Filters)})
    return Filters(**data)


def parse_stack_spec(data: dict[str, Any], source_path: Optional[Path] = None) -> StackSpec:
    """Validate a spec dict (as read by ``yaml.safe_load``) and build a :class:`StackSpec`."""
    if not isinstance(data, dict):
        raise ValueError("stack spec: top level must be a mapping")
    _check_keys("the top level", data, TOP_LEVEL_KEYS, required={"name", "layers"})
    layers = []
    for position, layer in enumerate(data["layers"], start=1):
        _check_keys(f"layers[{position}]", layer, LAYER_KEYS, required=LAYER_KEYS)
        layers.append(LayerSpec(poscar=str(layer["poscar"]), material=str(layer["material"])))
    if not layers:
        raise ValueError("stack spec: 'layers' must list at least one layer")
    name = str(data["name"])
    return StackSpec(
        name=name,
        output_dir=str(data.get("output_dir") or f"runs/{name}"),
        layers=layers,
        schedule=parse_search(data.get("search")),
        filters=parse_filters(data.get("filters")),
        source_path=source_path,
    )


def load_stack_spec(path: PathLike) -> StackSpec:
    """Read and validate a stack-spec YAML file (``yaml.safe_load``).

    >>> spec = load_stack_spec("examples/bto_trilayer_demo.yaml")   # doctest: +SKIP
    >>> spec.schedule.params_for_round(2).nmax                      # doctest: +SKIP
    40
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_stack_spec(data, source_path=path)


def run_stack_spec(
    path: PathLike,
    base_dir: Optional[PathLike] = None,
    dry_run: bool = False,
    log: Optional[Callable[[str], None]] = print,
) -> StackResult:
    """Load a stack spec and run it with :func:`MLM.search.build_stack`.

    The spec file is copied unchanged to ``<run_dir>/stack_spec.yaml``.
    ``base_dir`` is where relative paths in the spec start from (default: cwd);
    e.g. from a notebook in ``dev/`` use ``base_dir=".."``.

    >>> result = run_stack_spec("examples/bto_trilayer_demo.yaml", dry_run=True)   # doctest: +SKIP
    """
    base_dir = Path(base_dir) if base_dir is not None else Path(".")
    spec = load_stack_spec(path)
    run_dir = spec.run_dir(base_dir)
    result = build_stack(
        spec.roots(base_dir),
        spec.schedule,
        spec.filters,
        run_dir=run_dir,
        dry_run=dry_run,
        search_dirs=[base_dir],
        spec={},  # we copy the original file instead (keeps the user's comments)
        log=log,
    )
    if not dry_run:
        shutil.copyfile(path, run_dir / SPEC_FILENAME)
    return result
