"""Run directory storage, the index table, and the "jump to a structure" helpers.

A run directory looks like::

    runs/<run_name>/
      stack_spec.yaml     # the job that produced this run
      yamls/*.yaml        # one recipe per structure, all levels, roots included
      index.csv           # one row per recipe: scalar fields + round/passed_filter/path

``index.csv`` exists so that filtering thousands of candidates never requires
parsing thousands of YAML files. The YAML files remain the source of truth; the
index can always be regenerated from them with :func:`rebuild_index`.

Query helpers (all take the run directory):

- :func:`load_index` → ``pandas.DataFrame``
- :func:`find`       → filtered ``DataFrame``
- :func:`get`        → one :class:`~MLM.structure.MoireStructure`
- :func:`lineage`    → the tree of ancestors back to the roots
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from MLM.structure import MoireStructure, PathLike

INDEX_FILENAME = "index.csv"
YAML_DIRNAME = "yamls"
SPEC_FILENAME = "stack_spec.yaml"

#: Columns of ``index.csv``, in order. Lists are stored as joined strings
#: (``material`` with "," and ``parents`` with ";") so that every cell is a scalar.
INDEX_COLUMNS: tuple[str, ...] = (
    "id",
    "level",
    "n_layers",
    "material",
    "parents",
    "twist",
    "angle_between_vectors",
    "residual",
    "cumulative_residual",
    "max_layer_residual",
    "cell_area",
    "a1_length",
    "a2_length",
    "units",
    "round",
    "passed_filter",
    "path",
)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def index_row(record: MoireStructure, round_number: int, passed_filter: bool, path: str) -> dict:
    """One ``index.csv`` row for ``record`` (``path`` relative to the run directory)."""
    a1, a2 = record.lattice_vectors
    return {
        "id": record.id,
        "level": record.level,
        "n_layers": record.n_layers,
        "material": ",".join(record.material),
        "parents": ";".join(record.parents),
        "twist": record.twist,
        "angle_between_vectors": record.angle_between_vectors,
        "residual": record.residual,
        "cumulative_residual": record.cumulative_residual,
        "max_layer_residual": record.worst_layer_residual,
        "cell_area": record.cell_area,
        "a1_length": float((a1 @ a1) ** 0.5),
        "a2_length": float((a2 @ a2) ** 0.5),
        "units": record.units,
        "round": round_number,
        "passed_filter": bool(passed_filter),
        "path": path,
    }


@dataclass
class RunStore:
    """Writes recipes into a run directory and keeps the index rows in memory.

    The tree driver uses one ``RunStore`` per run::

        store = RunStore("runs/demo")
        store.add(record, round_number=1, passed_filter=True)
        store.write_index()
    """

    run_dir: Path
    rows: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        (self.run_dir / YAML_DIRNAME).mkdir(parents=True, exist_ok=True)

    def add(self, record: MoireStructure, round_number: int, passed_filter: bool) -> Path:
        """Write ``record``'s YAML (if new) and remember its index row.

        The same recipe can be produced more than once (e.g. the bilayers (1,2)
        and (3,4) of a homo-4-layer are identical): it is stored once, keeping
        the first round it appeared in.
        """
        path = record.to_yaml(self.run_dir / YAML_DIRNAME)
        if record.id not in self.rows:
            relative = path.relative_to(self.run_dir).as_posix()
            self.rows[record.id] = index_row(record, round_number, passed_filter, relative)
        return path

    def write_index(self) -> Path:
        """(Re)write ``index.csv`` from the rows collected so far."""
        return write_index(self.run_dir, list(self.rows.values()))


def write_index(run_dir: PathLike, rows: Sequence[dict]) -> Path:
    """Write ``rows`` to ``<run_dir>/index.csv`` (sorted by round, level, twist, id)."""
    table = pd.DataFrame(list(rows), columns=list(INDEX_COLUMNS))
    table = table.sort_values(["round", "level", "twist", "id"], na_position="first", kind="stable")
    path = Path(run_dir) / INDEX_FILENAME
    table.to_csv(path, index=False)
    return path


def rebuild_index(run_dir: PathLike, passed_filter: bool = True) -> Path:
    """Regenerate ``index.csv`` from the YAML files alone.

    Useful if the index was deleted or YAMLs were copied in by hand. The YAML files
    do not know which round produced them or whether they passed the run's filters,
    so ``round`` is set to the record's tree depth and ``passed_filter`` to the given value.

    >>> rebuild_index("runs/bto_trilayer_demo")   # doctest: +SKIP
    """
    run_dir = Path(run_dir)
    records = {}
    for path in sorted((run_dir / YAML_DIRNAME).glob("*.yaml")):
        record = MoireStructure.from_yaml(path)
        records[record.id] = (record, path.relative_to(run_dir).as_posix())
    depth_cache: dict[str, int] = {}

    def depth(record_id: str) -> int:
        if record_id not in depth_cache:
            record = records[record_id][0]
            depth_cache[record_id] = 0 if record.is_root else 1 + max(depth(p) for p in record.parents)
        return depth_cache[record_id]

    rows = [index_row(record, depth(rid), passed_filter, rel) for rid, (record, rel) in records.items()]
    return write_index(run_dir, rows)


# ---------------------------------------------------------------------------
# Reading / querying
# ---------------------------------------------------------------------------
def load_index(run_dir: PathLike) -> pd.DataFrame:
    """Read ``<run_dir>/index.csv`` as a pandas DataFrame.

    >>> table = load_index("runs/bto_trilayer_demo")      # doctest: +SKIP
    >>> table[["id", "n_layers", "twist", "cumulative_residual"]].head()   # doctest: +SKIP
    """
    path = Path(run_dir) / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no {INDEX_FILENAME} in {run_dir}; did the run finish a round?")
    table = pd.read_csv(path, keep_default_na=True)
    table["passed_filter"] = table["passed_filter"].astype(bool)
    return table


def _material_matches(stored: str, wanted: Union[str, Sequence[str]]) -> bool:
    labels = stored.split(",")
    if isinstance(wanted, str):
        # A single label means "made only of this material" (any number of layers).
        return set(labels) == {wanted}
    return labels == list(wanted)


def find(
    run_dir: PathLike,
    n_layers: Optional[int] = None,
    max_cumulative_residual: Optional[float] = None,
    angle_between_vectors: Optional[float] = None,
    material: Union[str, Sequence[str], None] = None,
    passed_filter: Optional[bool] = None,
    max_residual: Optional[float] = None,
    max_layer_residual: Optional[float] = None,
    max_cell_area: Optional[float] = None,
    twist_range: Optional[tuple[float, float]] = None,
    angle_tol: float = 1e-3,
) -> pd.DataFrame:
    """Filter the index table; every argument left as ``None`` is ignored.

    - ``material="bto"`` keeps structures made only of bto; a list such as
      ``["pto", "sto"]`` must match the stacking order exactly.
    - ``angle_between_vectors`` matches within ``angle_tol`` degrees.
    - ``twist_range=(lo, hi)`` keeps rows with ``lo <= twist <= hi``.

    Results are sorted by ``cumulative_residual`` (best first).

    >>> find("runs/bto_trilayer_demo", n_layers=3, max_cumulative_residual=1e-3)   # doctest: +SKIP
    """
    table = load_index(run_dir)
    keep = pd.Series(True, index=table.index)
    if n_layers is not None:
        keep &= table["n_layers"] == n_layers
    if max_cumulative_residual is not None:
        keep &= table["cumulative_residual"] <= max_cumulative_residual
    if max_residual is not None:
        keep &= table["residual"] <= max_residual
    if max_layer_residual is not None:
        keep &= table["max_layer_residual"] <= max_layer_residual
    if max_cell_area is not None:
        keep &= table["cell_area"] <= max_cell_area
    if angle_between_vectors is not None:
        keep &= (table["angle_between_vectors"] - angle_between_vectors).abs() <= angle_tol
    if twist_range is not None:
        low, high = twist_range
        keep &= table["twist"].between(low, high)
    if passed_filter is not None:
        keep &= table["passed_filter"] == passed_filter
    if material is not None:
        keep &= table["material"].apply(lambda stored: _material_matches(stored, material))
    result = table[keep]
    return result.sort_values("cumulative_residual", na_position="first", kind="stable")


def get(run_dir: PathLike, structure_id: str) -> MoireStructure:
    """Load one structure by id (the id stored in the file, looked up via the index).

    >>> trilayer = get("runs/bto_trilayer_demo", "bto_3_22.62_36.87_1a2b3c")   # doctest: +SKIP
    """
    table = load_index(run_dir)
    rows = table[table["id"] == structure_id]
    if rows.empty:
        raise KeyError(f"{structure_id!r} is not in {Path(run_dir) / INDEX_FILENAME}")
    record = MoireStructure.from_yaml(Path(run_dir) / rows.iloc[0]["path"])
    if record.id != structure_id:
        raise ValueError(f"index row points to a file whose id is {record.id!r}, not {structure_id!r}")
    return record


@dataclass
class LineageNode:
    """A record together with the lineage trees of its parents."""

    record: MoireStructure
    parents: list["LineageNode"] = field(default_factory=list)

    def roots(self) -> list[MoireStructure]:
        """Level-0 records at the leaves, in stacking order (one per layer)."""
        if not self.parents:
            return [self.record]
        return [root for parent in self.parents for root in parent.roots()]

    def walk(self) -> list[MoireStructure]:
        """All records of the tree, the structure itself first (depth-first, pre-order)."""
        return [self.record] + [record for parent in self.parents for record in parent.walk()]

    def format(self, indent: str = "") -> str:
        """Readable multi-line tree, e.g. for ``print``."""
        record = self.record
        twist = "" if record.twist is None else f"  twist={record.twist:.4f}"
        residual = "" if record.residual is None else f"  residual={record.residual:.2e}"
        lines = [f"{indent}{record.id}  ({record.n_layers} layer{'s' * (record.n_layers > 1)}){twist}{residual}"]
        for parent in self.parents:
            lines.append(parent.format(indent + "    "))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def lineage(run_dir: PathLike, structure_id: str) -> LineageNode:
    """The tree of ancestors of ``structure_id`` back to the level-0 roots.

    >>> print(lineage("runs/bto_trilayer_demo", trilayer_id))   # doctest: +SKIP
    bto_3_22.62_36.87_...  (3 layers)  twist=36.8700  residual=5.75e-05
        bto_2_22.62_...  (2 layers)  twist=22.6200  residual=3.39e-05
            bto_1_65687f  (1 layer)
            bto_1_65687f  (1 layer)
        bto_1_65687f  (1 layer)
    """
    cache: dict[str, MoireStructure] = {}

    def load(record_id: str) -> MoireStructure:
        if record_id not in cache:
            cache[record_id] = get(run_dir, record_id)
        return cache[record_id]

    def build(record_id: str) -> LineageNode:
        record = load(record_id)
        return LineageNode(record, [build(parent) for parent in record.parents])

    return build(structure_id)
