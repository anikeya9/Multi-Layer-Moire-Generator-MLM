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

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from MLM.structure import MoireStructure, PathLike

INDEX_FILENAME = "index.csv"
#: Alternative index for very large runs (``index_format="sqlite"``): one table, "structures".
SQLITE_FILENAME = "index.sqlite"
SQLITE_TABLE = "structures"
INDEX_FORMATS = ("csv", "sqlite")
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
    "max_strain",
    "strain_energy",
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
        "max_strain": record.max_strain,
        "strain_energy": record.strain_energy,
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
    index_format: str = "csv"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.index_format not in INDEX_FORMATS:
            raise ValueError(f"index_format must be one of {INDEX_FORMATS}, got {self.index_format!r}")
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

    def load_existing(self) -> None:
        """Start from the rows of an existing index (used when resuming a run)."""
        if index_path(self.run_dir) is None:
            return
        table = load_index(self.run_dir)
        for row in table.to_dict(orient="records"):
            # NaN (pandas' "missing") back to None, as written by index_row
            clean = {key: (None if isinstance(value, float) and value != value else value)
                     for key, value in row.items()}
            clean["parents"] = clean["parents"] or ""
            self.rows.setdefault(clean["id"], clean)

    def write_index(self) -> Path:
        """(Re)write the index (``index.csv`` or ``index.sqlite``) from the rows collected so far."""
        return write_index(self.run_dir, list(self.rows.values()), self.index_format)


def write_index(run_dir: PathLike, rows: Sequence[dict], index_format: str = "csv") -> Path:
    """Write ``rows`` to ``index.csv`` (default) or ``index.sqlite``, sorted by round, level, twist, id.

    SQLite is meant for very large runs: :func:`find` then filters with SQL instead of
    loading the whole table, and the most-used columns are indexed.
    """
    table = pd.DataFrame(list(rows), columns=list(INDEX_COLUMNS))
    table = table.sort_values(["round", "level", "twist", "id"], na_position="first", kind="stable")
    if index_format == "csv":
        path = Path(run_dir) / INDEX_FILENAME
        table.to_csv(path, index=False)
        return path
    if index_format != "sqlite":
        raise ValueError(f"index_format must be one of {INDEX_FORMATS}, got {index_format!r}")
    path = Path(run_dir) / SQLITE_FILENAME
    with sqlite3.connect(path) as connection:
        table.to_sql(SQLITE_TABLE, connection, if_exists="replace", index=False)
        for column in ("id", "n_layers", "cumulative_residual", "max_layer_residual", "cell_area"):
            connection.execute(f"CREATE INDEX IF NOT EXISTS idx_{column} ON {SQLITE_TABLE} ({column})")
    connection.close()
    return path


def index_path(run_dir: PathLike) -> Optional[Path]:
    """The run's index file: ``index.csv`` or ``index.sqlite`` (the newer one if both exist)."""
    candidates = [Path(run_dir) / name for name in (INDEX_FILENAME, SQLITE_FILENAME)]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _normalise(table: pd.DataFrame) -> pd.DataFrame:
    """Same column types for both formats (SQLite stores booleans as 0/1)."""
    table["passed_filter"] = table["passed_filter"].astype(bool)
    if "max_strain" not in table.columns:  # index written before strain support: all rigid
        table.insert(INDEX_COLUMNS.index("max_strain"), "max_strain", 0.0)
    if "strain_energy" not in table.columns:  # index written before elastic constants
        table.insert(INDEX_COLUMNS.index("strain_energy"), "strain_energy", float("nan"))
    return table


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
    path = index_path(run_dir)
    if path is None:
        raise FileNotFoundError(f"no {INDEX_FILENAME} or {SQLITE_FILENAME} in {run_dir}; did the run finish a round?")
    if path.name == SQLITE_FILENAME:
        with sqlite3.connect(path) as connection:
            table = pd.read_sql(f"SELECT * FROM {SQLITE_TABLE}", connection)
        connection.close()
    else:
        table = pd.read_csv(path, keep_default_na=True)
    return _normalise(table)


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
    max_strain: Optional[float] = None,
    max_strain_energy: Optional[float] = None,
    angle_tol: float = 1e-3,
) -> pd.DataFrame:
    """Filter the index table; every argument left as ``None`` is ignored.

    - ``material="bto"`` keeps structures made only of bto; a list such as
      ``["pto", "sto"]`` must match the stacking order exactly.
    - ``angle_between_vectors`` matches within ``angle_tol`` degrees.
    - ``twist_range=(lo, hi)`` keeps rows with ``lo <= twist <= hi``.
    - ``max_strain`` (percent): largest absolute principal strain of any layer.
    - ``max_strain_energy`` (meV/Å²): rows without a known energy are dropped.

    Results are sorted by ``cumulative_residual`` (best first).

    >>> find("runs/bto_trilayer_demo", n_layers=3, max_cumulative_residual=1e-3)   # doctest: +SKIP
    """
    # Every numeric filter as (SQL condition, parameters, pandas mask function).
    conditions: list[tuple[str, tuple, object]] = []

    def at_most(column: str, value: Optional[float]) -> None:
        if value is not None:
            conditions.append((f"{column} <= ?", (value,), lambda t, c=column, v=value: t[c] <= v))

    if n_layers is not None:
        conditions.append(("n_layers = ?", (n_layers,), lambda t: t["n_layers"] == n_layers))
    at_most("cumulative_residual", max_cumulative_residual)
    at_most("residual", max_residual)
    at_most("max_layer_residual", max_layer_residual)
    at_most("cell_area", max_cell_area)
    at_most("max_strain", max_strain)
    at_most("strain_energy", max_strain_energy)
    if angle_between_vectors is not None:
        conditions.append(("ABS(angle_between_vectors - ?) <= ?", (angle_between_vectors, angle_tol),
                           lambda t: (t["angle_between_vectors"] - angle_between_vectors).abs() <= angle_tol))
    if twist_range is not None:
        low, high = twist_range
        conditions.append(("twist BETWEEN ? AND ?", (low, high), lambda t: t["twist"].between(low, high)))
    if passed_filter is not None:
        conditions.append(("passed_filter = ?", (int(passed_filter),),
                           lambda t: t["passed_filter"] == passed_filter))

    path = index_path(run_dir)
    if path is not None and path.name == SQLITE_FILENAME:
        # Large runs: let SQLite filter, so the whole table never has to be loaded.
        where = " AND ".join(sql for sql, _, _ in conditions) or "1"
        parameters = tuple(value for _, values, _ in conditions for value in values)
        with sqlite3.connect(path) as connection:
            result = pd.read_sql(f"SELECT * FROM {SQLITE_TABLE} WHERE {where}", connection, params=parameters)
        connection.close()
        result = _normalise(result)
    else:
        table = load_index(run_dir)
        keep = pd.Series(True, index=table.index)
        for _, _, mask in conditions:
            keep &= mask(table)
        result = table[keep]
    if material is not None:
        result = result[result["material"].apply(lambda stored: _material_matches(stored, material))]
    return result.sort_values("cumulative_residual", na_position="first", kind="stable")


def get(run_dir: PathLike, structure_id: str) -> MoireStructure:
    """Load one structure by id (the id stored in the file, looked up via the index).

    >>> trilayer = get("runs/bto_trilayer_demo", "bto_3_22.62_36.87_015688")   # doctest: +SKIP
    """
    path = index_path(run_dir)
    if path is not None and path.name == SQLITE_FILENAME:
        with sqlite3.connect(path) as connection:
            found = connection.execute(f"SELECT path FROM {SQLITE_TABLE} WHERE id = ?", (structure_id,)).fetchall()
        connection.close()
        paths = [row[0] for row in found]
    else:
        table = load_index(run_dir)
        paths = list(table.loc[table["id"] == structure_id, "path"])
    if not paths:
        raise KeyError(f"{structure_id!r} is not in the index of {run_dir}")
    record = MoireStructure.from_yaml(Path(run_dir) / paths[0])
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
    bto_3_22.62_36.87_015688  (3 layers)  twist=36.8700  residual=1.33e-04
        bto_2_22.62_1fdb6f  (2 layers)  twist=22.6200  residual=3.39e-05
            bto_1_406932  (1 layer)
            bto_1_406932  (1 layer)
        bto_1_406932  (1 layer)
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
