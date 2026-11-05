"""Graphical interface for MLM (Jupyter widgets; also served as a web page by Voilà).

In a notebook::

    from MLM.gui import MLMApp
    MLMApp().display()

As a stand-alone page in the browser: ``mlm gui`` (runs Voilà on ``notebooks/mlm_gui.ipynb``).

Tabs, in the order of a job:

1. **Stack**   — layers (file, label, strain limit, elastic constants), a twist range per
                 interface, cell angle, residual limits, strain mode, advanced settings.
2. **Plan**    — the dry run: which pair of layers is joined in which round, and a time estimate.
3. **Run**     — run in the background with a progress bar; Stop, and Resume later.
4. **Results** — open any run directory, filter the candidates, scatter plot, pick one.
5. **Viewer**  — 3D view (py3Dmol): colour by layer or element, show/hide layers, cell,
                 2x2 / 3x3 repeat, top / side view; lineage, layer table and sanity checks.
6. **Export**  — VASP / LAMMPS with interlayer gap, vacuum, lateral shifts, Poisson ratios;
                 download links, and the stack spec of the job.

The heavy lifting is the normal API (``MLM.search``, ``MLM.index``, ``MLM.build``,
``MLM.checks``); this module only wires widgets to it. Everything a button does is also
a method (``plan()``, ``run()``, ``select()``, ``export()``), which the tests call.
"""

from __future__ import annotations

import base64
import html
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from MLM.build import build_atoms, default_search_dirs, write_structure
from MLM.checks import check_built, to_ase
from MLM.index import find, get, lineage, load_index
from MLM.search import (
    Filters,
    SearchParams,
    SearchSchedule,
    build_stack,
    job_to_spec_dict,
    plan_rounds,
)
from MLM.strain import Elastic
from MLM.structure import MoireStructure, read_primitive

#: Rough speed of the fast scanner (lattice-point tests per second), for time estimates.
TESTS_PER_SECOND = 2e7

#: Simple table style for the HTML tables shown in the GUI.
TABLE_CSS = ("<style>.mlm-table{border-collapse:collapse;font-size:13px}.mlm-table th,.mlm-table td"
             "{border:1px solid #ccc;padding:3px 8px;text-align:left}.mlm-table th{background:#f2f2f2}</style>")

#: Colours of layers 1, 2, 3, ... in the viewer and the plots.
LAYER_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _widgets():
    """Import ipywidgets lazily, with a clear message if it is missing."""
    try:
        import ipywidgets
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("the GUI needs ipywidgets: pip install 'MLM[gui]'") from error
    return ipywidgets


# ---------------------------------------------------------------------------
# Pure helpers (no widgets; unit-tested)
# ---------------------------------------------------------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    """The folder containing ``moire_structures/`` (the GUI may run from ``notebooks/``)."""
    start = Path.cwd() if start is None else Path(start)
    for folder in [start, *start.resolve().parents]:
        if (folder / "moire_structures").is_dir():
            return folder
    return start


def list_layer_files(root: Path, max_atoms: int = 30) -> list[str]:
    """Primitive layer files (POSCAR/CIF) under ``root/moire_structures`` with ≤ ``max_atoms`` atoms."""
    files = []
    base = Path(root) / "moire_structures"
    for path in sorted([*base.rglob("*.vasp"), *base.rglob("*.cif")]):
        try:
            if len(read_primitive(path)["coords"]) <= max_atoms:
                files.append(path.relative_to(root).as_posix())
        except Exception:
            continue
    return files


def list_runs(runs_dir: Path) -> list[str]:
    """Run directories (those with an index) under ``runs_dir``, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    found = [d for d in runs_dir.iterdir() if d.is_dir() and ((d / "index.csv").exists() or (d / "index.sqlite").exists())]
    return [d.name for d in sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)]


def build_job(
    layers: Sequence[dict[str, Any]],
    twist_ranges: Sequence[tuple[float, float]],
    cell_angles: Sequence[float],
    max_residual: Optional[float],
    max_layer_residual: Optional[float],
    strain_mode: str = "isotropic",
    angle_step: float = 0.01,
    match_tolerance: float = 5e-4,
    n_first: int = 10,
    n_later: int = 40,
    scanner: str = "fast",
    include_axes: bool = False,
    merge_twists: str = "none",
    root: Path = Path("."),
) -> dict[str, Any]:
    """Turn the form values into the arguments of :func:`MLM.search.build_stack`.

    ``layers``: dicts with ``poscar``, ``label`` and optional ``strain`` (%), ``young``
    (N/m), ``poisson``. Returns ``{"roots", "schedule", "filters", "strain_limits", "elastic"}``.
    """
    if len(layers) < 2:
        raise ValueError("add at least two layers")
    if len(twist_ranges) != len(layers) - 1:
        raise ValueError(f"need {len(layers) - 1} twist ranges (one per interface), got {len(twist_ranges)}")
    for k, (low, high) in enumerate(twist_ranges, start=1):
        if not low < high:
            raise ValueError(f"interface {k}: the twist range needs min < max, got ({low}, {high})")
    if not cell_angles:
        raise ValueError("choose at least one cell angle")
    roots = [MoireStructure.from_poscar(layer["poscar"], layer["label"] or Path(layer["poscar"]).stem,
                                        search_dirs=[root]) for layer in layers]
    default = SearchParams(degmin=float(twist_ranges[0][0]), degmax=float(twist_ranges[0][1]), degstep=angle_step,
                           dtol=match_tolerance, nmin=-n_first, nmax=n_first, goodangles=tuple(cell_angles),
                           strain_mode=strain_mode, scanner=scanner, include_axes=include_axes,
                           merge_equivalent_twists=merge_twists)
    rounds = {r: {"nmin": -n_later, "nmax": n_later} for r in range(2, len(layers) + 1)}
    interfaces = {k: {"degmin": float(low), "degmax": float(high)} for k, (low, high) in enumerate(twist_ranges, start=1)}
    elastic = None
    if any(layer.get("young") for layer in layers):
        elastic = [Elastic(young=float(layer["young"]), poisson=float(layer.get("poisson") or 0.0))
                   if layer.get("young") else None for layer in layers]
    return {
        "roots": roots,
        "schedule": SearchSchedule(default, rounds, interfaces),
        "filters": Filters(max_residual=max_residual, max_layer_residual=max_layer_residual),
        "strain_limits": [float(layer.get("strain") or 0.0) for layer in layers],
        "elastic": elastic,
    }


def plan_table(job: dict[str, Any]) -> pd.DataFrame:
    """One row per step of the plan, with the twist window, strain limits and a time estimate."""
    rows = []
    limits = job["strain_limits"]
    for round_number, steps in enumerate(plan_rounds(len(job["roots"])), start=1):
        for step in steps:
            if step.is_carry:
                rows.append({"round": round_number, "combination": str(step), "interface": "—", "twist range (deg)": "—",
                             "n range": "—", "strain limits (%)": "—", "≈ s per search": "—"})
                continue
            params = job["schedule"].params_for_step(round_number, step)
            lower = limits[step.lower[0]] if step.lower[1] - step.lower[0] == 1 else 0.0
            upper = limits[step.upper[0]] if step.upper[1] - step.upper[0] == 1 else 0.0
            rows.append({"round": round_number, "combination": str(step),
                         "interface": f"layer {step.interface} → {step.interface + 1}",
                         "twist range (deg)": f"{params.degmin:g} – {params.degmax:g}",
                         "n range": f"±{params.nmax}",
                         "strain limits (%)": f"{lower:g} / {upper:g}" if lower or upper else "rigid",
                         "≈ s per search": round(params.estimated_cost() / TESTS_PER_SECOND, 2)})
    return pd.DataFrame(rows)


def results_table(run_dir: Path, n_layers: Optional[int] = None, passed_only: bool = True,
                  max_residual: Optional[float] = None, max_layer_residual: Optional[float] = None,
                  max_strain: Optional[float] = None) -> pd.DataFrame:
    """Candidates of a run (default: the full stack), smallest cell first."""
    table = load_index(run_dir)
    if n_layers is None:
        n_layers = int(table["n_layers"].max())
    hits = find(run_dir, n_layers=n_layers, passed_filter=True if passed_only else None,
                max_residual=max_residual, max_layer_residual=max_layer_residual, max_strain=max_strain)
    columns = ["id", "twist", "residual", "max_layer_residual", "max_strain", "strain_energy", "cell_area", "passed_filter"]
    return hits.sort_values(["cell_area", "max_layer_residual"])[columns].reset_index(drop=True)


def viewer_atoms(built: Any, visible_layers: Optional[Sequence[int]] = None, repeat: int = 1):
    """ASE atoms of the selected layers, repeated ``repeat`` x ``repeat`` in-plane, with a
    ``layer`` array (1-based) for colouring."""
    atoms = to_ase(built)
    layers = built.atoms["layer"].to_numpy() + 1
    atoms.set_array("layer", layers)
    if visible_layers is not None:
        atoms = atoms[np.isin(layers, list(visible_layers))]
    if repeat > 1:
        atoms = atoms.repeat((repeat, repeat, 1))
    return atoms


def py3dmol_view(atoms, colour_by: str = "layer", view: str = "top", show_cell: bool = True,
                 cell: Optional[np.ndarray] = None, width: int = 640, height: int = 480):
    """A ``py3Dmol.view`` of ``atoms`` (one model per layer, so layers get their own colour)."""
    import py3Dmol

    viewer = py3Dmol.view(width=width, height=height)
    layers = atoms.get_array("layer") if "layer" in atoms.arrays else np.ones(len(atoms), dtype=int)
    for index, layer in enumerate(sorted(set(layers.tolist()))):
        subset = atoms[layers == layer]
        xyz = f"{len(subset)}\n\n" + "\n".join(f"{s} {x:.5f} {y:.5f} {z:.5f}"
                                              for s, (x, y, z) in zip(subset.get_chemical_symbols(), subset.positions))
        viewer.addModel(xyz, "xyz")
        if colour_by == "layer":
            style = {"sphere": {"radius": 0.45, "color": LAYER_COLOURS[(layer - 1) % len(LAYER_COLOURS)]}}
        else:
            style = {"sphere": {"radius": 0.45, "colorscheme": "Jmol"}}
        viewer.setStyle({"model": index}, style)
    if show_cell and cell is not None:
        a1, a2, a3 = np.asarray(cell, dtype=float)
        z0 = float(atoms.positions[:, 2].min()) if len(atoms) else 0.0
        corners = [np.zeros(3), a1, a1 + a2, a2, np.zeros(3)]
        for start, end in zip(corners[:-1], corners[1:]):
            viewer.addLine({"start": dict(zip("xyz", (start + [0, 0, z0]).tolist())),
                            "end": dict(zip("xyz", (end + [0, 0, z0]).tolist())), "color": "black"})
    viewer.zoomTo()
    if view == "side":
        viewer.rotate(-90, "x")
    elif view == "perspective":
        viewer.rotate(-60, "x")
    return viewer


def download_link(path: Path, label: Optional[str] = None) -> str:
    """HTML link that downloads ``path`` (embedded as a data URI; works in Voilà too)."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    name = html.escape(Path(path).name)
    return (f'<a download="{name}" href="data:application/octet-stream;base64,{data}" '
            f'style="margin-right:1em">⬇ {html.escape(label or name)}</a>')


def parse_shift(text: str) -> Optional[tuple[float, float]]:
    """``"1/3, 2/3"`` or ``"0.5 0.5"`` → (s1, s2); empty → ``None``."""
    text = text.strip()
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"a shift needs two numbers 's1, s2', got {text!r}")

    def number(part: str) -> float:
        if "/" in part:
            numerator, denominator = part.split("/")
            return float(numerator) / float(denominator)
        return float(part)

    return number(parts[0]), number(parts[1])


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------
class MLMApp:
    """The MLM graphical interface. ``MLMApp().display()`` shows it in a notebook."""

    def __init__(self, root: Optional[Path] = None, runs_dir: Optional[Path] = None) -> None:
        self.w = _widgets()
        self.root = find_repo_root(root)
        self.runs_dir = Path(runs_dir) if runs_dir is not None else self.root / "runs"
        self.layer_files = list_layer_files(self.root)
        self.layer_rows: list[dict[str, Any]] = []
        self.run_dir: Optional[Path] = None
        self.selected: Optional[MoireStructure] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._build_stack_tab()
        self._build_plan_tab()
        self._build_run_tab()
        self._build_results_tab()
        self._build_viewer_tab()
        self._build_export_tab()
        tabs = [self.stack_tab, self.plan_tab, self.run_tab, self.results_tab, self.viewer_tab, self.export_tab]
        self.tabs = self.w.Tab(children=tabs)
        for index, title in enumerate(["1 Stack", "2 Plan", "3 Run", "4 Results", "5 Viewer", "6 Export"]):
            self.tabs.set_title(index, title)
        self.widget = self.w.VBox([self.w.HTML("<h2 style='margin:0'>MLM — multi-layer moiré builder</h2>"), self.tabs])
        self.set_layers([{"poscar": "moire_structures/bto/bto_1-layer.vasp", "label": "bto"}] * 3,
                        [(22.0, 23.0), (36.0, 38.0)])
        self.refresh_runs()

    def display(self) -> None:
        """Show the app (in a notebook or Voilà)."""
        from IPython.display import display

        display(self.widget)

    # --- 1. Stack ---------------------------------------------------------------
    def _build_stack_tab(self) -> None:
        w = self.w
        self.layers_box = w.VBox()
        add = w.Button(description="Add layer", icon="plus")
        add.on_click(lambda _: self._add_layer_row({"poscar": self.layer_files[0] if self.layer_files else "",
                                                     "label": ""}, refresh=True))
        self.interfaces_box = w.VBox()
        self.cell_angles = w.SelectMultiple(options=[60.0, 90.0, 120.0], value=(90.0,), rows=3,
                                            description="cell angle")
        wide = {"description_width": "130px"}
        self.max_residual = w.FloatText(value=1e-3, description="max residual (Å)", step=1e-4, style=wide)
        self.max_layer_residual = w.FloatText(value=1e-3, description="max layer residual (Å)", step=1e-4, style=wide)
        self.use_layer_residual = w.Checkbox(value=True, description="use it", indent=False)
        self.strain_mode = w.Dropdown(options=["isotropic", "axial", "full"], value="isotropic",
                                      description="strain mode", style=wide)
        self.run_name = w.Text(value="gui_run", description="run name", style=wide)
        self.angle_step = w.FloatText(value=0.01, description="angle step°")
        self.match_tolerance = w.FloatText(value=5e-4, description="match tol.")
        self.n_first = w.IntText(value=10, description="n round 1")
        self.n_later = w.IntText(value=40, description="n later")
        self.scanner = w.Dropdown(options=["fast", "match"], value="fast", description="scanner")
        self.include_axes = w.Checkbox(value=False, description="include axis points")
        self.merge_twists = w.Dropdown(options=["none", "rotation", "rotation+mirror"], value="none",
                                       description="merge twists")
        self.workers = w.IntText(value=1, description="workers (0=all)")
        self.index_format = w.Dropdown(options=["csv", "sqlite"], value="csv", description="index")
        advanced = w.Accordion(children=[w.VBox([
            w.HBox([self.angle_step, self.match_tolerance]), w.HBox([self.n_first, self.n_later]),
            w.HBox([self.scanner, self.include_axes]), w.HBox([self.merge_twists, self.workers]), self.index_format,
            w.HTML("<small>Strained searches: an angle step of 0.1° is usually enough; raise the match "
                   "tolerance (e.g. 9e-3) for rigid lattice-mismatched heterostructures.</small>")])])
        advanced.set_title(0, "Advanced settings")
        advanced.selected_index = None
        self.stack_message = w.HTML()
        self.stack_tab = w.VBox([
            w.HTML("<b>Layers</b>, bottom → top. Strain in % (0 = rigid); Young's modulus (N/m) and "
                   "Poisson ratio are optional (energy-weighted strain sharing, strain energy)."),
            self.layers_box, add,
            w.HTML("<b>Twist ranges</b> (degrees): layer k+1 relative to layer k (the layer directly below)."),
            self.interfaces_box,
            w.HBox([self.cell_angles, w.VBox([self.max_residual, w.HBox([self.max_layer_residual, self.use_layer_residual]),
                                               self.strain_mode, self.run_name])]),
            advanced, self.stack_message])

    def _add_layer_row(self, values: dict[str, Any], refresh: bool = False) -> None:
        w = self.w
        options = list(self.layer_files)
        if values.get("poscar") and values["poscar"] not in options:
            options.append(values["poscar"])
        row = {
            "poscar": w.Combobox(options=options, value=values.get("poscar", ""), placeholder="path to POSCAR/CIF",
                                 ensure_option=False, layout=w.Layout(width="420px")),
            "label": w.Text(value=values.get("label", ""), placeholder="label", layout=w.Layout(width="80px")),
            "strain": w.FloatText(value=float(values.get("strain") or 0.0), layout=w.Layout(width="70px")),
            "young": w.FloatText(value=float(values.get("young") or 0.0), layout=w.Layout(width="80px")),
            "poisson": w.FloatText(value=float(values.get("poisson") or 0.0), layout=w.Layout(width="70px")),
        }
        up, down, remove = (w.Button(icon=icon, layout=w.Layout(width="36px")) for icon in ("arrow-up", "arrow-down", "trash"))
        up.on_click(lambda _, r=row: self._move_layer(r, -1))
        down.on_click(lambda _, r=row: self._move_layer(r, +1))
        remove.on_click(lambda _, r=row: self._remove_layer(r))
        row["box"] = w.HBox([row["poscar"], row["label"], w.Label("strain %"), row["strain"], w.Label("Y"), row["young"],
                             w.Label("ν"), row["poisson"], up, down, remove])
        self.layer_rows.append(row)
        if refresh:
            self._refresh_layers()

    def _move_layer(self, row: dict[str, Any], direction: int) -> None:
        index = self.layer_rows.index(row)
        target = index + direction
        if 0 <= target < len(self.layer_rows):
            self.layer_rows[index], self.layer_rows[target] = self.layer_rows[target], self.layer_rows[index]
            self._refresh_layers()

    def _remove_layer(self, row: dict[str, Any]) -> None:
        self.layer_rows.remove(row)
        self._refresh_layers()

    def _refresh_layers(self) -> None:
        """Redraw the layer rows and keep one twist slider per interface."""
        w = self.w
        for number, row in enumerate(self.layer_rows, start=1):
            row["box"].children[0].description = f"L{number}"
        self.layers_box.children = [row["box"] for row in self.layer_rows]
        old = [slider.value for slider in getattr(self, "twist_sliders", [])]
        self.twist_sliders = []
        for k in range(1, len(self.layer_rows)):
            value = old[k - 1] if k - 1 < len(old) else (1.0, 45.0)
            self.twist_sliders.append(w.FloatRangeSlider(value=value, min=-180.0, max=180.0, step=0.01,
                                                         description=f"L{k} → L{k + 1}", readout_format=".2f",
                                                         continuous_update=False, layout=w.Layout(width="620px")))
        self.interfaces_box.children = self.twist_sliders or [w.HTML("<i>add a second layer</i>")]

    def set_layers(self, layers: Sequence[dict[str, Any]], twist_ranges: Optional[Sequence[tuple[float, float]]] = None) -> None:
        """Fill the Stack tab from code (also used for the default example)."""
        self.layer_rows = []
        for values in layers:
            self._add_layer_row(dict(values))
        self.twist_sliders = []
        self._refresh_layers()
        for slider, value in zip(self.twist_sliders, twist_ranges or []):
            slider.value = tuple(value)

    def layer_values(self) -> list[dict[str, Any]]:
        return [{"poscar": row["poscar"].value, "label": row["label"].value, "strain": row["strain"].value,
                 "young": row["young"].value or None, "poisson": row["poisson"].value or None}
                for row in self.layer_rows]

    def job(self) -> dict[str, Any]:
        """The current form as ``build_stack`` arguments (raises ``ValueError`` on bad input)."""
        return build_job(
            self.layer_values(), [slider.value for slider in self.twist_sliders], list(self.cell_angles.value),
            self.max_residual.value or None, self.max_layer_residual.value if self.use_layer_residual.value else None,
            self.strain_mode.value, self.angle_step.value, self.match_tolerance.value, self.n_first.value,
            self.n_later.value, self.scanner.value, self.include_axes.value, self.merge_twists.value, self.root)

    # --- 2. Plan ------------------------------------------------------------------
    def _build_plan_tab(self) -> None:
        w = self.w
        button = w.Button(description="Plan (dry run)", icon="list", button_style="info")
        button.on_click(lambda _: self.plan())
        self.plan_out = w.HTML()
        self.plan_tab = w.VBox([w.HTML("Shows the work before anything is searched. Later rounds run one search "
                                       "per candidate of the blocks they join, known only after the earlier round."),
                                button, self.plan_out])

    def plan(self) -> Optional[pd.DataFrame]:
        try:
            table = plan_table(self.job())
        except Exception as error:
            self.plan_out.value = f"<p style='color:#c00'>⚠ {html.escape(str(error))}</p>"
            return None
        self.plan_out.value = TABLE_CSS + table.to_html(index=False, classes="mlm-table", border=0)
        return table

    # --- 3. Run -------------------------------------------------------------------
    def _build_run_tab(self) -> None:
        w = self.w
        self.run_button = w.Button(description="Run", icon="play", button_style="success")
        self.stop_button = w.Button(description="Stop", icon="stop", button_style="danger", disabled=True)
        self.resume = w.Checkbox(value=False, description="resume an interrupted run of the same job",
                                 indent=False, layout=w.Layout(width="360px"))
        self.progress = w.IntProgress(value=0, min=0, max=1, description="searches")
        self.progress_label = w.Label()
        self.log = w.Output(layout=w.Layout(max_height="300px", overflow="auto", border="1px solid #ddd"))
        self.run_button.on_click(lambda _: self.run(background=True))
        self.stop_button.on_click(lambda _: self._stop.set())
        self.run_tab = w.VBox([w.HBox([self.run_button, self.stop_button, self.resume]),
                               w.HBox([self.progress, self.progress_label]), self.log])

    def run(self, background: bool = True) -> None:
        """Run the job of the Stack tab (in a thread, so the Stop button keeps working)."""
        try:
            job = self.job()
        except Exception as error:
            self.log.append_stdout(f"⚠ {error}\n")
            return
        run_dir = self.runs_dir / (self.run_name.value.strip() or "gui_run")
        self._stop.clear()
        self.run_button.disabled, self.stop_button.disabled = True, False
        self.log.clear_output()

        def say(message: str) -> None:
            # append_stdout, not `with self.log: print(...)`: the run happens in a background
            # thread, where an Output widget's context manager does not capture anything.
            self.log.append_stdout(message + "\n")

        def on_progress(event: dict[str, Any]) -> None:
            self.progress.max = max(event["total"], 1)
            self.progress.value = event["done"]
            self.progress_label.value = f"round {event['round']}: {event['step']} ({event['done']}/{event['total']})"

        def work() -> None:
            try:
                spec = job_to_spec_dict(run_dir.name, job["roots"], job["schedule"], job["filters"],
                                        f"runs/{run_dir.name}", job["strain_limits"], job["elastic"])
                result = build_stack(job["roots"], job["schedule"], job["filters"], run_dir=run_dir,
                                     search_dirs=[self.root], spec=spec, log=say, strain_limits=job["strain_limits"],
                                     elastic=job["elastic"], workers=self.workers.value, resume=self.resume.value,
                                     should_stop=self._stop.is_set, progress_callback=on_progress,
                                     index_format=self.index_format.value)
                say("" if result.status == "finished" else "stopped — tick 'resume' and Run to continue")
                say(f"{len(result.final)} full-stack candidates passed. See the Results tab.")
                self.refresh_runs(select=run_dir.name)
            except Exception as error:  # report, do not kill the GUI
                say(f"⚠ {type(error).__name__}: {error}")
            finally:
                self.run_button.disabled, self.stop_button.disabled = False, True

        if background:
            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
        else:
            work()

    # --- 4. Results -----------------------------------------------------------------
    def _build_results_tab(self) -> None:
        w = self.w
        self.run_select = w.Dropdown(description="run")
        refresh = w.Button(icon="refresh", layout=w.Layout(width="40px"))
        refresh.on_click(lambda _: self.refresh_runs())
        self.n_layers_select = w.Dropdown(description="layers")
        self.passed_only = w.Checkbox(value=True, description="passed filters only")
        wide = {"description_width": "150px"}
        self.filter_residual = w.FloatText(value=0.0, description="max layer residual (Å)", style=wide)
        self.filter_strain = w.FloatText(value=0.0, description="max strain (%)", style=wide)
        apply = w.Button(description="Apply", icon="filter")
        apply.on_click(lambda _: self.show_results())
        self.run_select.observe(lambda change: self._run_changed(), names="value")
        self.candidates = w.Select(rows=10, layout=w.Layout(width="100%"))
        self.candidates.add_class("mlm-candidates")  # stable hook for scripted browser tests
        open_button = w.Button(description="View selected", icon="eye", button_style="primary")
        open_button.on_click(lambda _: self.select(self.candidates.value))
        self.results_summary = w.HTML()
        self.results_plot = w.Image(format="png", layout=w.Layout(max_width="720px"))
        self.results_tab = w.VBox([
            w.HBox([self.run_select, refresh, self.n_layers_select, self.passed_only]),
            w.HBox([self.filter_residual, self.filter_strain, apply]),
            self.results_summary, self.candidates, open_button, self.results_plot])

    def refresh_runs(self, select: Optional[str] = None) -> None:
        runs = list_runs(self.runs_dir)
        self.run_select.options = runs
        if select in runs:
            self.run_select.value = select
        elif runs and self.run_select.value not in runs:
            self.run_select.value = runs[0]

    def _run_changed(self) -> None:
        if not self.run_select.value:
            return
        self.run_dir = self.runs_dir / self.run_select.value
        layers = sorted(int(n) for n in load_index(self.run_dir)["n_layers"].unique())
        self.n_layers_select.options = layers
        self.n_layers_select.value = layers[-1]
        self.show_results()

    def show_results(self) -> pd.DataFrame:
        """Fill the candidate list and the scatter plot for the chosen run and filters."""
        if self.run_dir is None:
            return pd.DataFrame()
        table = results_table(self.run_dir, self.n_layers_select.value, self.passed_only.value,
                              max_layer_residual=self.filter_residual.value or None,
                              max_strain=self.filter_strain.value or None)
        self.results = table
        self.candidates.options = [
            (f"{row.id}   twist {row.twist:8.3f}°   area {row.cell_area:9.1f} Å²   "
             f"layer res {row.max_layer_residual:.1e} Å   strain {row.max_strain:.3f} %", row.id)
            for row in table.itertuples()]
        self.results_summary.value = f"<b>{len(table)}</b> candidate(s) in <code>{html.escape(str(self.run_dir))}</code>"
        self._plot_results(table)
        return table

    def _plot_results(self, table: pd.DataFrame) -> None:
        """Scatter plot as a PNG in an Image widget (works from the background run thread too)."""
        import io

        from matplotlib.figure import Figure

        if table.empty:
            self.results_plot.value = b""
            return
        fig = Figure(figsize=(6.5, 3.2), dpi=110)
        ax = fig.subplots()
        colour = np.log10(table["max_layer_residual"].clip(lower=1e-16))
        points = ax.scatter(table["twist"], table["cell_area"], c=colour, cmap="viridis", s=18)
        if self.selected is not None and self.selected.id in set(table["id"]):
            row = table[table["id"] == self.selected.id].iloc[0]
            ax.scatter([row["twist"]], [row["cell_area"]], s=120, facecolors="none", edgecolors="red")
        ax.set_yscale("log")
        ax.set_xlabel("twist of the last interface (deg)")
        ax.set_ylabel("cell area (Å²)")
        fig.colorbar(points, ax=ax, label="log10 max layer residual (Å)")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        self.results_plot.value = buffer.getvalue()

    def select(self, structure_id: Optional[str]) -> Optional[MoireStructure]:
        """Open a candidate in the Viewer tab."""
        if not structure_id or self.run_dir is None:
            return None
        self.selected = get(self.run_dir, structure_id)
        self.show_structure()
        self.tabs.selected_index = 4
        return self.selected

    # --- 5. Viewer -------------------------------------------------------------------
    def _build_viewer_tab(self) -> None:
        w = self.w
        self.colour_by = w.ToggleButtons(options=["layer", "element"], value="layer", description="colour")
        self.view_direction = w.ToggleButtons(options=["top", "side", "perspective"], value="top", description="view")
        self.repeat = w.Dropdown(options=[("1×1", 1), ("2×2", 2), ("3×3", 3)], value=1, description="repeat")
        self.show_cell = w.Checkbox(value=True, description="cell")
        self.visible_layers = w.SelectMultiple(description="layers", rows=4)
        redraw = w.Button(description="Redraw", icon="refresh")
        redraw.on_click(lambda _: self.show_structure())
        self.viewer_out = w.Output()
        self.viewer_info = w.HTML()
        self.viewer_tab = w.VBox([
            w.HBox([self.colour_by, self.view_direction]),
            w.HBox([self.repeat, self.show_cell, self.visible_layers, redraw]),
            self.viewer_out, self.viewer_info])

    def show_structure(self) -> Optional[str]:
        """Draw the selected structure; returns the viewer's HTML (used by the tests)."""
        record = self.selected
        if record is None:
            return None
        built = build_atoms(record, search_dirs=default_search_dirs(self.run_dir))
        self.built = built
        labels = [(f"{k}: {record.material[k - 1]} ({record.layer_rotations[k - 1]:.2f}°)", k)
                  for k in range(1, record.n_layers + 1)]
        if [value for _, value in self.visible_layers.options] != [value for _, value in labels]:
            self.visible_layers.options = labels
            self.visible_layers.value = tuple(value for _, value in labels)
        atoms = viewer_atoms(built, self.visible_layers.value or None, self.repeat.value)
        cell = built.cell * np.array([[self.repeat.value], [self.repeat.value], [1]])
        view = py3dmol_view(atoms, self.colour_by.value, self.view_direction.value, self.show_cell.value, cell)
        markup = view._make_html()
        self.viewer_out.clear_output(wait=True)
        with self.viewer_out:
            from IPython.display import HTML, display

            display(HTML(markup))
        report = check_built(built, search_dirs=default_search_dirs(self.run_dir))
        layer_rows = "".join(
            f"<tr><td>{k + 1}</td><td>{html.escape(record.material[k])}</td><td>{record.layer_rotations[k]:.4f}</td>"
            f"<td>{np.round(record.layer_strains[k], 4).tolist()}</td>"
            f"<td>{'' if record.layer_residuals is None else f'{record.layer_residuals[k]:.2e}'}</td>"
            f"<td>{built.layer_counts[k]}</td></tr>" for k in range(record.n_layers))
        energy = "—" if record.strain_energy is None else f"{record.strain_energy:.4f} meV/Å²"
        self.viewer_info.value = (
            f"<h4>{html.escape(record.id)}</h4>"
            f"<p>cell |A1| = {np.linalg.norm(record.lattice_vectors[0]):.3f} Å, |A2| = "
            f"{np.linalg.norm(record.lattice_vectors[1]):.3f} Å, angle {record.angle_between_vectors:.2f}°, "
            f"area {record.cell_area:.1f} Å²; strain energy {energy}</p>"
            f"{TABLE_CSS}<table class='mlm-table'><tr><th>layer</th><th>material</th><th>rotation (°)</th><th>strain tensor (%)</th>"
            f"<th>residual (Å)</th><th>atoms</th></tr>{layer_rows}</table>"
            f"<p><b>Checks:</b></p><pre>{html.escape(str(report))}</pre>"
            f"<p><b>Lineage:</b></p><pre>{html.escape(str(lineage(self.run_dir, record.id)))}</pre>")
        return markup

    # --- 6. Export -----------------------------------------------------------------------
    def _build_export_tab(self) -> None:
        w = self.w
        self.formats = w.SelectMultiple(options=["vasp", "lammps"], value=("vasp",), description="formats", rows=2)
        self.gap = w.FloatText(value=3.0, description="gap Å")
        self.vacuum = w.FloatText(value=50.0, description="vacuum Å")
        self.shifts_text = w.Text(value="", description="shifts",
                                  placeholder="per layer 's1 s2' separated by ';', e.g. ; 1/3 2/3 ;",
                                  layout=w.Layout(width="620px"))
        self.use_poisson = w.Checkbox(value=False, description="Poisson thickness change (ν from the Stack tab)",
                                      indent=False, layout=w.Layout(width="420px"))
        button = w.Button(description="Write files", icon="download", button_style="primary")
        button.on_click(lambda _: self.export())
        self.export_out = w.HTML()
        self.export_tab = w.VBox([
            w.HTML("Writes the structure selected in the Viewer. Shifts are fractions of each layer's own "
                   "primitive cell (AB stacking of hexagonal layers: 1/3 2/3)."),
            w.HBox([self.formats, w.VBox([self.gap, self.vacuum])]), self.shifts_text, self.use_poisson,
            button, self.export_out])

    def export(self) -> list[Path]:
        """Write the selected structure (and the job's stack spec) and show download links."""
        record = self.selected
        if record is None or self.run_dir is None:
            self.export_out.value = "<p style='color:#c00'>⚠ select a structure first (Results tab)</p>"
            return []
        try:
            shifts = None
            if self.shifts_text.value.strip():
                parts = self.shifts_text.value.split(";")
                if len(parts) != record.n_layers:
                    raise ValueError(f"give {record.n_layers} shifts separated by ';' (empty = no shift)")
                shifts = [parse_shift(part) for part in parts]
            poisson = None
            if self.use_poisson.value:
                values = [row["poisson"].value for row in self.layer_rows]
                poisson = values if len(values) == record.n_layers else None
            paths = [write_structure(self.run_dir, record.id, fmt=fmt, interlayer_gap=self.gap.value,
                                     vacuum=self.vacuum.value, shifts=shifts, poisson_ratios=poisson,
                                     search_dirs=default_search_dirs(self.run_dir), check=False)
                     for fmt in self.formats.value]
        except Exception as error:
            self.export_out.value = f"<p style='color:#c00'>⚠ {html.escape(str(error))}</p>"
            return []
        spec = self.run_dir / "stack_spec.yaml"
        links = [download_link(path) for path in paths] + ([download_link(spec)] if spec.exists() else [])
        self.export_out.value = ("<p>" + "".join(links) + "</p><p><small>" +
                                 "<br>".join(html.escape(str(path)) for path in paths) + "</small></p>")
        return paths


def launch(port: Optional[int] = None, open_browser: bool = True) -> int:
    """``mlm gui``: serve ``notebooks/mlm_gui.ipynb`` with Voilà (a stand-alone web page)."""
    import shutil
    import subprocess

    root = find_repo_root()
    notebook = root / "notebooks" / "mlm_gui.ipynb"
    if not notebook.exists():
        raise FileNotFoundError(f"{notebook} not found; run `mlm gui` inside the MLM repository")
    voila = shutil.which("voila")
    if voila is None:
        raise RuntimeError("Voilà is not installed: pip install 'MLM[gui]'")
    command = [voila, str(notebook), f"--Voila.root_dir={root}"]
    if port is not None:
        command.append(f"--port={port}")
    if not open_browser:
        command.append("--no-browser")
    return subprocess.call(command, cwd=root)


__all__ = ["MLMApp", "launch", "build_job", "plan_table", "results_table", "viewer_atoms", "py3dmol_view",
           "download_link", "parse_shift", "list_layer_files", "list_runs", "find_repo_root"]
