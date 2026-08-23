#!/usr/bin/env python3
"""Generate the split public-release figure notebooks and inventory."""

from __future__ import annotations

import ast
import csv
import re
import textwrap
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEGACY = ROOT / "JAMES_Manuscript_and_Supplement_Reproducibility.ipynb"


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def py(text):
    return nbformat.v4.new_code_cell(text.strip() + "\n")


def write_nb(name, cells):
    nb = nbformat.v4.new_notebook(cells=cells)
    for index, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{index:03d}"
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    nbformat.write(nb, HERE / name)
    print((HERE / name).relative_to(ROOT))


def node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def select(source, names, rename=None):
    rename = rename or {}
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        name = node_name(node)
        if name in names:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node.name = rename.get(name, name)
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def embed_script(relative_path, exclude_names=()):
    exclude_names = set(exclude_names)
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__"
    ]
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test))
    ]
    tree.body = [node for node in tree.body if node_name(node) not in exclude_names]
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    source = re.sub(
        r"Path\((['\"])/data/[^'\"]+/data/original\1\)",
        "REPO_ROOT",
        source,
    )
    source = source.replace(
        "import netCDF4",
        "try:\n    import netCDF4\nexcept ImportError:\n    netCDF4 = None",
    )
    return f"# Consolidated from {relative_path}\nREPO_ROOT = ROOT\n__file__ = str(ROOT / {relative_path!r})\n\n{source}"


def clean_notebook_cell(source, rename=None):
    rename = rename or {}
    tree = ast.parse(source)
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test))
    ]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in rename:
            node.name = rename[node.name]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def wrap_cell(source, function_name):
    return f"def {function_name}():\n{textwrap.indent(clean_notebook_cell(source), '    ')}"


SETUP = r'''
from __future__ import annotations
import hashlib, importlib.metadata, io, json, math, os, re, shutil, sys, zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/james-mpl-cache")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Image as NotebookImage, SVG, display
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d

try:
    import cartopy.crs as ccrs
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False
try:
    import wavebreaking as wb
    HAVE_WAVEBREAKING = True
except ImportError:
    HAVE_WAVEBREAKING = False
try:
    from contrack import contrack as ConTrack
    HAVE_CONTRACK = True
except ImportError:
    HAVE_CONTRACK = False

def find_repo_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "ready_version").is_dir() and (candidate / "derived_data").is_dir():
            return candidate
    raise FileNotFoundError("Run from the repository or one of its subdirectories")

ROOT = find_repo_root()
RAW_DIR = ROOT
DERIVED_DIR = ROOT / "derived_data"
PUBLICATION_DIR = ROOT / "publication_notebooks"
FIGURE_DIR = PUBLICATION_DIR / "outputs"
TABLE_DIR = PUBLICATION_DIR / "outputs"
SI_DIR = ROOT / "paper_revision" / "supplemental_analysis"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RUN_HEAVY_PROCESSING = False
USE_PRECOMPUTED_DERIVED_DATA = True
SAVE_FIGURES = True
SHOW_FIGURES = False
REBUILD_FIGURES = os.environ.get(
    "PUBLICATION_REBUILD_FIGURES",
    os.environ.get("PUBLICATION_FULL_REBUILD", "1"),
) == "1"
FULL_REBUILD = REBUILD_FIGURES
EARTH_RADIUS_M = 6_371_229.0
EARTH_OMEGA = 7.2921e-5
GRAVITY = 9.80616
RD = 287.05
CP = 1004.0
KAPPA = RD / CP
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9})

def show_docx_media(docx_relative_path, members):
    members = [members] if isinstance(members, str) else list(members)
    with zipfile.ZipFile(ROOT / docx_relative_path) as archive:
        for member in members:
            payload = archive.read(member)
            display(SVG(data=payload.decode("utf-8")) if member.lower().endswith(".svg") else NotebookImage(data=payload))

def show_files(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    for relative_path in paths:
        path = ROOT / relative_path
        if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            display(NotebookImage(filename=str(path)))

# Optional environment diagnostics are intentionally disabled in the public notebooks.
# print(f"Repository root: {ROOT.relative_to(ROOT)}")
# print(f"Rebuild figures: {REBUILD_FIGURES}")
# print(f"Cartopy={HAVE_CARTOPY}, WaveBreaking={HAVE_WAVEBREAKING}, ConTrack={HAVE_CONTRACK}")
'''


FIGURE10_SOURCE = r'''
FIGURE10_COUNTS = {
    "b15n1": 3, "b15n1s-10": 2, "b15n1s-5": 3, "b15n1s10": 10, "b15n1s5": 8,
    "b15n3": 2, "b15n3s-10": 0, "b15n3s-5": 2, "b15n3s10": 4, "b15n3s5": 3,
    "b15n6": 2, "b15n6s-10": 0, "b15n6s-5": 1, "b15n6s10": 2, "b15n6s5": 2,
    "b1n1": 2, "b1n1s-10": 2, "b1n1s-5": 2, "b1n1s10": 7, "b1n1s5": 5,
    "b1n3": 2, "b1n3s-10": 0, "b1n3s-5": 0, "b1n3s10": 3, "b1n3s5": 2,
    "b1n6": 1, "b1n6s-10": 0, "b1n6s-5": 0, "b1n6s10": 2, "b1n6s5": 2,
    "b2n1": 4, "b2n1s-10": 2, "b2n1s-5": 3, "b2n1s10": 19, "b2n1s5": 8,
    "b2n3": 2, "b2n3s-10": 0, "b2n3s-5": 2, "b2n3s10": 5, "b2n3s5": 4,
    "b2n6": 2, "b2n6s-10": 0, "b2n6s-5": 1, "b2n6s10": 4, "b2n6s5": 2,
}

def figure10_table():
    rows = []
    pattern = re.compile(r"b(15|1|2)n(1|3|6)(?:s(-?\d+))?")
    for case, count in FIGURE10_COUNTS.items():
        match = pattern.fullmatch(case)
        rows.append({
            "case": case,
            "b": 1.5 if match.group(1) == "15" else float(match.group(1)),
            "n": int(match.group(2)),
            "s": int(match.group(3) or 0),
            "mergers": count,
        })
    return pd.DataFrame(rows)

def plot_figure10():
    from matplotlib.colors import LinearSegmentedColormap
    data = figure10_table()
    b_levels = [1.0, 1.5, 2.0]
    n_levels = [6, 3, 1]
    s_levels = [-10, -5, 0, 5, 10]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes = axes.ravel()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    color_map = LinearSegmentedColormap.from_list("red_only", ["white", "red"])
    for index, shift in enumerate(s_levels):
        axis = axes[index]
        matrix = data[data.s == shift].pivot(index="b", columns="n", values="mergers").reindex(index=b_levels, columns=n_levels)
        axis.imshow(matrix.values, origin="lower", cmap=color_map, vmin=0, vmax=20, aspect="equal")
        axis.set_xticks(np.arange(3), [str(value) for value in n_levels])
        axis.set_yticks(np.arange(3), [str(value) for value in b_levels])
        axis.tick_params(labelsize=20)
        axis.set_xlabel("n (wider jet →)", fontsize=22)
        axis.set_ylabel("b (larger b →)", fontsize=22)
        axis.set_title(f"{labels[index]}  s = {shift}°", fontsize=24)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{int(matrix.values[row, column])}", ha="center", va="center", fontsize=22)
    axes[-1].axis("off")
    data.to_csv(TABLE_DIR / "Figure10_coalescence_event_counts.csv", index=False)
    figure.tight_layout()
    return save_figure(figure, "Figure10_coalescence_event_counts.png")
'''


def make_inventory():
    rows = []
    def add(doc, fig, media, artifact, source, notebook, confidence, data):
        rows.append((doc, fig, media, artifact, source, notebook, confidence, data))

    main_doc = "ready_version/AGU_Manuscript_FV3_Mingfei.docx"
    main_specs = [
        (2, "image3.png", "current_version/updated_figures/Figure2_panel_titles_and_shading_corrected.png", "paper_revision/figure_updates/rebuild_figure2.py", "01_main_figures_02_06.ipynb", "high", "analytic initial-condition generator"),
        (3, "image4.png", f"{main_doc}::word/media/image4.png", "analysis_notebooks/case_studies/case.ipynb; analysis_scripts/case_visualization/visu.py", "01_main_figures_02_06.ipynb", "high", "raw BCwave NetCDF"),
        (4, "image6.svg", f"{main_doc}::word/media/image6.svg", "analysis_notebooks/case_studies/case.ipynb; analysis_notebooks/case_studies/casecheck.ipynb", "01_main_figures_02_06.ipynb", "source_layout", "raw BCwave NetCDF"),
        (5, "image7.png", f"{main_doc}::word/media/image7.png", "analysis_notebooks/rwb/RWB.ipynb; analysis_scripts/overturning/overturning.py", "01_main_figures_02_06.ipynb", "high", "derived_data/pv_isentropic"),
        (6, "image9.svg", f"{main_doc}::word/media/image9.svg", "analysis_notebooks/energetics/regional_eke.ipynb", "01_main_figures_02_06.ipynb", "high", "raw BCwave NetCDF"),
        (7, "image11.svg", "figures/case_studies/figure6_omg500_polar_days10-14.png", "analysis_notebooks/case_studies/casecheck.ipynb", "02_main_figures_07_10.ipynb", "high", "raw BCwave NetCDF"),
        (8, "image13.svg", "current_version/updated_figures/Figure8_corrected.png", "paper_revision/figure_updates/rebuild_figure8_right_panels.py; analysis_notebooks/tracking/contrack.ipynb", "02_main_figures_07_10.ipynb", "high", "raw NetCDF; merger catalog; wind diagnostics"),
        (9, "image14.png", "paper_revision/manuscript/revised/media_from_docx/media/image13.png", "analysis_notebooks/exploratory/check.ipynb; analysis_notebooks/merger/pri_cyclone_merger.ipynb", "02_main_figures_07_10.ipynb", "high", "raw BCwave NetCDF"),
        (10, "image15.png", f"{main_doc}::word/media/image15.png", "analysis_notebooks/merger/merger.ipynb; analysis_notebooks/merger/pri_cyclone_merger.ipynb", "02_main_figures_07_10.ipynb", "high", "merging_times.txt catalogs"),
        (11, "image17.svg", f"{main_doc}::word/media/image17.svg", "analysis_notebooks/case_studies/casecheck.ipynb", "03_main_figures_11_13.ipynb", "high", "raw BCwave NetCDF"),
        (12, "image19.svg", "current_version/updated_figures/Figure12_corrected.png", "paper_revision/figure_updates/rebuild_figures_8_12.py; analysis_notebooks/exploratory/check.ipynb", "03_main_figures_11_13.ipynb", "high", "raw BCwave NetCDF"),
        (13, "image21.svg", "paper_revision/manuscript/revised/media_from_docx/media/image19.svg", "analysis_notebooks/case_studies/casecheck.ipynb; revision_analysis/scripts/r2_2_stationarity_reinterpretation.py", "03_main_figures_11_13.ipynb", "high", "anticyclone overlap masks/tracks"),
    ]
    for number, media, artifact, source, notebook, confidence, data in main_specs:
        add("main", f"Figure {number}", f"word/media/{media}", artifact, source, notebook, confidence, data)

    a = "eddy/controlled_umax30_bns_analysis_20260722"
    add("supplement", "Figure A1", "word/media/image1.png;word/media/image2.png", f"{a}/figures/F1a_rwb_theta_lat_environment_by_b.png;{a}/figures/F1b_original_vs_controlled_ns_group_means.png", f"{a}/scripts/build_figure_a1_rwb_cross_sections.py", "04_supplement_A_to_C.ipynb", "exact_layout", "archive and fixed-Umax RWB products")
    for label, image, artifact in [("A2", 3, "F2_controlled_all45_heatmaps.png"), ("A3", 4, "F3_archive_controlled_paired.png"), ("A4", 5, "F4_parameter_spearman_comparison.png")]:
        add("supplement", f"Figure {label}", f"word/media/image{image}.png", f"{a}/figures/{artifact}", f"{a}/scripts/build_umax30_all45_section_f.py", "04_supplement_A_to_C.ipynb", "exact_layout", "controlled response tables")
    add("supplement", "Figure A5", "word/media/image6.png", f"{a}/original_style_comparison/figures/Figure5_original_vs_controlled_same_style.png", f"{a}/original_style_comparison/scripts/make_original_style_comparison_figures.py", "04_supplement_A_to_C.ipynb", "high", "archive/control RWB counts")

    b = "eddy/eddy_full_coalescence_review_analysis_20260723"
    for label, image, artifact, script in [
        ("B1", 7, "figure06_case_level_full_eddy_scatter.png", "aggregate_and_plot.py"),
        ("B2", 8, "figure03_full_eddy_group_trends.png", "aggregate_and_plot.py"),
        ("B3", 9, "figure02_eddy_threshold_sensitivity_heatmaps.png", "aggregate_and_plot.py"),
        ("B4", 10, "figure07_eddy_vorticity_core_outcomes.png", "aggregate_and_plot.py"),
        ("B5", 11, "figure11_archive_controlled_full_eddy_four_row_comparison.png", "aggregate_controlled_full_eddy.py"),
        ("B6", 12, "figure12_controlled_eddy_threshold_sensitivity.png", "aggregate_controlled_full_eddy.py"),
    ]:
        add("supplement", f"Figure {label}", f"word/media/image{image}.png", f"{b}/figures/{artifact}", f"{b}/scripts/{script}", "04_supplement_A_to_C.ipynb", "exact", "full/eddy coalescence tables")

    c = "eddy/rwb_post_onset_analysis_20260722"
    add("supplement", "Figure C1", "word/media/image13.png", f"{c}/figures/rwb_full_post_onset_and_rate_by_parameters.png", f"{c}/scripts/build_rwb_post_onset_analysis.py", "04_supplement_A_to_C.ipynb", "exact", "post-onset RWB summary")
    add("supplement", "Figure C2", "word/media/image14.png", f"{c}/figures/rwb_post_onset_rate_heatmaps.png", f"{c}/scripts/build_rwb_post_onset_analysis.py", "04_supplement_A_to_C.ipynb", "exact", "post-onset RWB summary")

    d = "eddy/initial_state_analysis_20260722"
    add("supplement", "Figure D1", "word/media/image15.png", "paper_revision/supplemental_analysis/figures/r1_12_initial_pv_cross_sections.png", "paper_revision/supplemental_analysis/compute_r1_12_initial_pv_cross_sections.py", "05_supplement_D_to_G.ipynb", "exact", "raw initial states")
    for label, image, artifact in [("D2", 16, "initial_cross_sections_b_variation.png"), ("D3", 17, "initial_cross_sections_n_s_variations.png"), ("D4", 18, "initial_predictor_response_correlation_heatmap.png")]:
        add("supplement", f"Figure {label}", f"word/media/image{image}.png", f"{d}/figures/{artifact}", f"{d}/scripts/build_initial_state_report_analysis.py", "05_supplement_D_to_G.ipynb", "exact_layout", "initial-state diagnostics")
    add("supplement", "Figure D5", "word/media/image19.png", "paper_revision/supplemental_analysis/figures/r1_6_zonal_mean_jet_evolution.png", "paper_revision/supplemental_analysis/compute_r1_6_zonal_mean_jet_evolution.py", "05_supplement_D_to_G.ipynb", "exact", "raw representative cases")
    add("supplement", "Figure E1", ";".join(f"word/media/image{i}.png" for i in range(20, 30)), "eddy/primary_secondary_first_binary_coalescence_A1_20260725/pages/Figure_A1_primary_secondary_page_01_of_10.png;...page_10_of_10.png", "eddy/primary_secondary_first_binary_coalescence_A1_20260725/scripts/build_primary_secondary_A1_atlas.py; eddy/expanded_A1_cyclone_interaction_atlas_20260723/scripts/build_expanded_A1_atlas.py", "05_supplement_D_to_G.ipynb", "high", "atlas manifest; raw/derived PV")
    add("supplement", "Figure F1", "word/media/image30.png", "revision_analysis/figures/r2_2_supplementary_lifetime_and_centroid_motion.png", "revision_analysis/scripts/r2_2_stationarity_reinterpretation.py", "05_supplement_D_to_G.ipynb", "exact_layout", "anticyclone stationarity metrics/tracks")
    add("supplement", "Figure G1", "word/media/image31.png", "paper_revision/manuscript/revised/media_from_docx/media/image20.png", "analytic initialization code consolidated from legacy reproducibility notebook", "05_supplement_D_to_G.ipynb", "high", "analytic initial-condition parameters")

    topics = {
        "Figure 2": "Initial jet and thermal-wind structure",
        "Figure 3": "Surface-pressure and 850-hPa-temperature life cycle", "Figure 4": "Upper-level flow evolution",
        "Figure 5": "RWB object-time statistics", "Figure 6": "Domain-mean EKE evolution",
        "Figure 7": "Cyclone pressure-contour interaction", "Figure 8": "Surface-wind response around pressure connection",
        "Figure 9": "Cyclone Hovmöller diagrams", "Figure 10": "Cyclone-interaction event counts",
        "Figure 11": "Persistent-anticyclone evolution", "Figure 12": "High-pressure Hovmöller diagrams",
        "Figure 13": "Anticyclone overlap duration", "Figure A1": "RWB environment and controlled-speed group means",
        "Figure A2": "Controlled-speed response matrices", "Figure A3": "Paired archive/control responses",
        "Figure A4": "Parameter Spearman associations", "Figure A5": "RWB orientation fractions",
        "Figure B1": "Full-field versus eddy-SLP case counts", "Figure B2": "Full/eddy grouped trends",
        "Figure B3": "Eddy-SLP threshold sensitivity", "Figure B4": "Vorticity-core outcomes",
        "Figure B5": "Archive/control full/eddy comparison", "Figure B6": "Controlled eddy-threshold sensitivity",
        "Figure C1": "Post-onset RWB counts and rates", "Figure C2": "Post-onset RWB rate heatmaps",
        "Figure D1": "Initial pressure-coordinate PV", "Figure D2": "Initial and growth-phase b cross sections",
        "Figure D3": "Initial n and s cross sections", "Figure D4": "Predictor-response correlations",
        "Figure D5": "Eulerian zonal-mean jet evolution", "Figure E1": "Primary-secondary binary-coalescence atlas",
        "Figure F1": "Anticyclone lifetime and centroid motion", "Figure G1": "Initial zonal-wind profiles",
    }
    columns = ["document", "figure", "docx_media", "reference_artifact", "source_code", "public_notebook", "match_confidence", "primary_data_dependency", "figure_topic"]
    with (HERE / "figure_code_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows([(*row, topics[row[1]]) for row in rows])
    return rows


def shared(legacy):
    manifest_source = legacy.cells[5].source.replace(
        "'raw_file':str(RAW_DIR/f'{stem}.nc')",
        "'raw_file':f'{stem}.nc'",
    ).replace(
        "'coarse_file':str(DERIVED_DIR/'coarse5avg'/f'{stem}.nc')",
        "'coarse_file':f'derived_data/coarse5avg/{stem}.nc'",
    ).replace(
        "'pv_file':str(DERIVED_DIR/'pv_isentropic'/f'{stem}.nc')",
        "'pv_file':f'derived_data/pv_isentropic/{stem}.nc'",
    )
    manifest_source = manifest_source.replace(
        "TABLE1.to_csv(TABLE_DIR/'Table1_experiment_manifest.csv',index=False)\nTABLE1",
        "TABLE1.to_csv(TABLE_DIR/'Table1_experiment_manifest.csv',index=False)\n"
        "# Optional manifest preview is disabled to keep notebook output figure-only.\n"
        "# display(TABLE1)",
    )
    return [py(SETUP), py(manifest_source), py(legacy.cells[6].source)]


def run_and_show(call, previews):
    return py(
        "# Diagnostic stdout/stderr is hidden so saved notebook output contains only publication figures.\n"
        "_diagnostic_output = io.StringIO()\n"
        "_original_argv = sys.argv[:]\n"
        "sys.argv = ['publication_notebook']\n"
        "try:\n"
        "    if REBUILD_FIGURES:\n"
        "        with redirect_stdout(_diagnostic_output), redirect_stderr(_diagnostic_output):\n"
        f"            {call}\n"
        "finally:\n"
        "    sys.argv = _original_argv\n"
        f"show_files({previews!r})"
    )


def source_section(title, source, previews, call="main()", embedded_source=None):
    code = embedded_source if embedded_source is not None else embed_script(source)
    return [
        md(f"## {title}\n\nOriginal source: `{source}`"),
        py(code),
        run_and_show(call, previews),
    ]


def build_notebooks(legacy):
    casecheck = nbformat.read(ROOT / "analysis_notebooks/case_studies/casecheck.ipynb", as_version=4)
    case = nbformat.read(ROOT / "analysis_notebooks/case_studies/case.ipynb", as_version=4)
    rwb = nbformat.read(ROOT / "analysis_notebooks/rwb/RWB.ipynb", as_version=4)
    regional_eke = nbformat.read(ROOT / "analysis_notebooks/energetics/regional_eke.ipynb", as_version=4)
    exploratory = nbformat.read(ROOT / "analysis_notebooks/exploratory/check.ipynb", as_version=4)
    merger = nbformat.read(ROOT / "analysis_notebooks/merger/pri_cyclone_merger.ipynb", as_version=4)
    revision_utils = (ROOT / "revision_analysis/scripts/revision_utils.py").read_text(encoding="utf-8")

    figure2_source = embed_script("paper_revision/figure_updates/rebuild_figure2.py").replace(
        "def main() -> Path:", "def plot_figure2_original() -> Path:"
    )
    figure2_source += "\n\ndef plot_figure2():\n    source = plot_figure2_original()\n    output = FIGURE_DIR / 'Figure2_initial_jet_and_temperature.png'\n    shutil.copy2(source, output)\n    return output"

    figure3_left = clean_notebook_cell(case.cells[2].source, {"plot_days_6_to_14_from_hourly": "_plot_figure3_left"}).replace("plt.show()", "return fig")
    figure3_right = clean_notebook_cell(case.cells[3].source, {"plot_days_6_to_14_from_hourly": "_plot_figure3_right"}).replace("plt.show()", "return fig")
    figure3_source = figure3_left + "\n\n" + figure3_right + r'''
def plot_figure3():
    from PIL import Image as PILImage
    figures = [_plot_figure3_left(str(RAW_DIR), 'BCwave_b2n3'), _plot_figure3_right(str(RAW_DIR), 'BCwave_b2n3s10')]
    images = []
    for figure in figures:
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        images.append(PILImage.open(buffer).convert('RGB').copy())
        plt.close(figure)
    height = max(image.height for image in images)
    resized = [image.resize((round(image.width * height / image.height), height), PILImage.Resampling.LANCZOS) for image in images]
    canvas = PILImage.new('RGB', (sum(image.width for image in resized), height), 'white')
    offset = 0
    for image in resized:
        canvas.paste(image, (offset, 0)); offset += image.width
    output = FIGURE_DIR / 'Figure3_surface_pressure_temperature_lifecycle.png'
    canvas.save(output, dpi=(300, 300))
    return output
'''

    figure4_source = clean_notebook_cell(case.cells[6].source, {"plot_gh300_vort500": "_plot_figure4_original"})
    figure4_source = figure4_source.replace("plot_gh300_vort500('./', 'BCwave_b2n3s10')", "")
    figure4_source = figure4_source.replace("plt.show()", "return fig")
    figure4_source += r'''

def plot_figure4():
    from PIL import Image as PILImage
    images = []
    for filename, title in [
        ('BCwave_b2n3', 'b=2, n=3, s=0'),
        ('BCwave_b2n3s10', 'b=2, n=3, s=10'),
    ]:
        figure = _plot_figure4_original(str(RAW_DIR), filename)
        for axis in figure.axes[:5]:
            axis.set_title('', loc='center')
        figure.suptitle(title, fontsize=20, y=0.995)
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        images.append(PILImage.open(buffer).convert('RGB').copy())
        plt.close(figure)
    height = max(image.height for image in images)
    resized = [
        image.resize((round(image.width * height / image.height), height), PILImage.Resampling.LANCZOS)
        for image in images
    ]
    panel_height = round(height * 0.86)
    colorbar = resized[-1].crop((
        round(resized[-1].width * 0.10),
        round(height * 0.87),
        round(resized[-1].width * 0.90),
        height,
    ))
    canvas_width = sum(image.width for image in resized)
    canvas = PILImage.new('RGB', (canvas_width, panel_height + colorbar.height), 'white')
    offset = 0
    for image in resized:
        canvas.paste(image.crop((0, 0, image.width, panel_height)), (offset, 0))
        offset += image.width
    canvas.paste(colorbar, ((canvas_width - colorbar.width) // 2, panel_height))
    output = FIGURE_DIR / 'Figure4_upper_level_lifecycle.png'
    canvas.save(output, dpi=(300, 300))
    return output
'''

    figure5_source = clean_notebook_cell(rwb.cells[0].source, {"main": "plot_figure5"})
    figure5_source = re.sub(
        r"data_dir\s*=\s*(['\"])/data/[^'\"]+/data/original/derived_data/pv_isentropic/?\1",
        "data_dir = str(DERIVED_DIR / 'pv_isentropic')",
        figure5_source,
    ).replace("higher jet →", "larger b →").replace("plt.show()", "return save_figure(fig, 'Figure5_RWB_object_time_statistics.png')")

    figure6_source = wrap_cell(regional_eke.cells[0].source, "plot_figure6")
    figure6_source = figure6_source.replace("path = '.'", "path = str(RAW_DIR)")
    figure6_source = figure6_source.replace("plt.show()", "return save_figure(fig, 'Figure6_EKE_evolution.png')")

    figure7_source = wrap_cell(casecheck.cells[55].source, "plot_figure7")
    for filename in ("BCwave_b2n3.nc", "BCwave_b2n3s10.nc", "BCwave_b2n1s10.nc"):
        figure7_source = figure7_source.replace(f"'{filename}'", f"str(RAW_DIR / '{filename}')")
    figure7_source = figure7_source.replace(
        "plt.savefig('figure6_omg500_polar_days10-14.png', dpi=600, bbox_inches='tight')",
        "output = save_figure(fig, 'Figure7_pressure_contour_coalescence.png')",
    ).replace("plt.show()", "return output")

    figure8_analysis = clean_notebook_cell(merger.cells[10].source)
    figure8_source = embed_script(
        "paper_revision/figure_updates/rebuild_figure8_right_panels.py",
        exclude_names={"load_original_analysis_namespace"},
    ).replace(
        "def main() -> None:", "def plot_figure8_original() -> None:"
    )
    figure8_source = figure8_source.replace("    namespace = load_original_analysis_namespace()\n", "")
    figure8_source = figure8_source.replace(
        "        buckets_n = namespace['collect_all_series_rel_by_n'](",
        "        buckets_n = collect_all_series_rel_by_n(",
    )
    figure8_source = figure8_analysis + "\n\n" + figure8_source
    figure8_source += "\n\ndef plot_figure8():\n    plot_figure8_original()\n    source = ROOT / 'current_version/updated_figures/Figure8_corrected.png'\n    output = FIGURE_DIR / 'Figure8_wind_and_pressure_coalescence.png'\n    shutil.copy2(source, output)\n    return output"

    figure9_source = wrap_cell(exploratory.cells[2].source, "plot_figure9")
    figure9_source = figure9_source.replace("data_folder = './'", "data_folder = str(RAW_DIR)")
    figure9_source = figure9_source.replace("plt.show()", "return save_figure(fig, 'Figure9_cyclone_hovmoller.png')")

    figure11_source = wrap_cell(casecheck.cells[51].source, "plot_figure11")
    figure11_source = figure11_source.replace("xr.open_dataset('BCwave_b2n1s10.nc'", "xr.open_dataset(RAW_DIR / 'BCwave_b2n1s10.nc'")
    figure11_source = figure11_source.replace(
        "plt.savefig('polar_omg500_days7-12_custom_labels.png', dpi=600, bbox_inches='tight')",
        "output = save_figure(fig, 'Figure11_persistent_anticyclone.png')",
    ).replace("plt.show()", "return output")

    figure12_source = embed_script("paper_revision/figure_updates/rebuild_figures_8_12.py")
    figure12_source += "\n\ndef plot_figure12():\n    source = rebuild_figure12()\n    output = FIGURE_DIR / 'Figure12_high_pressure_hovmoller.png'\n    shutil.copy2(source, output)\n    return output"
    figure13_source = next(
        cell.source
        for cell in casecheck.cells
        if cell.cell_type == "code" and "def plot_panels():" in cell.source and "track_id_to_count = 1" in cell.source
    )
    figure13_source = figure13_source.replace('folder = "./"', "folder = str(RAW_DIR)")
    figure13_source = figure13_source.replace("def plot_panels():", "def plot_figure13():")
    figure13_source = figure13_source.replace(
        "b (higher jet →)", "b (larger b →)").replace(
        "    plt.tight_layout()\n    plt.show()\n\nif __name__ == \"__main__\":\n    plot_panels()",
        "    plt.tight_layout()\n    return save_figure(fig, 'Figure13_anticyclone_overlap_duration.png')",
    )

    write_nb("00_inventory_and_environment.ipynb", [
        md("# Manuscript Figure and Code Inventory\n\nFigure 1 is intentionally excluded from the public plotting package. The main-text notebooks cover Figures 2–13, and the supporting-information notebooks cover Figures A1–A5, B1–B6, C1–C2, D1–D5, E1, F1, and G1. Figure 14 in the legacy combined notebook corresponds to Figure G1 in the submitted supporting information."),
        py(SETUP),
        py("inventory = pd.read_csv(PUBLICATION_DIR / 'figure_code_inventory.csv')\n# Optional inventory preview is disabled to keep notebook output figure-only.\n# display(inventory)\nassert len(inventory) == 33\nassert 'Figure 1' not in set(inventory.figure)\nassert inventory.public_notebook.map(lambda name: (PUBLICATION_DIR / name).exists()).all()"),
        py("# Optional manuscript-reference preview is disabled to avoid duplicate figure output.\n# show_docx_media('ready_version/AGU_Manuscript_FV3_Mingfei.docx', 'word/media/image3.png')"),
    ])

    common = shared(legacy)
    write_nb("01_main_figures_02_06.ipynb", [
        md("# Main Figures 2–6\n\nInitialization, life cycles, RWB, and EKE. Each section contains the plotting code, runs it directly by default, and displays the generated file."), *common,
        md("## Figure 2"), py(figure2_source), run_and_show("plot_figure2()", ["publication_notebooks/outputs/Figure2_initial_jet_and_temperature.png"]),
        md("## Figure 3"), py(figure3_source), run_and_show("plot_figure3()", ["publication_notebooks/outputs/Figure3_surface_pressure_temperature_lifecycle.png"]),
        md("## Figure 4"), py(figure4_source), run_and_show("plot_figure4()", ["publication_notebooks/outputs/Figure4_upper_level_lifecycle.png"]),
        md("## Figure 5"), py(figure5_source), run_and_show("plot_figure5()", ["publication_notebooks/outputs/Figure5_RWB_object_time_statistics.png"]),
        md("## Figure 6"), py(figure6_source), run_and_show("plot_figure6()", ["publication_notebooks/outputs/Figure6_EKE_evolution.png"]),
    ])

    write_nb("02_main_figures_07_10.ipynb", [
        md("# Main Figures 7–10\n\nCyclone interaction, wind response, Hovmöller diagnostics, and coalescence counts. Each generated figure is displayed immediately after its plotting cell."), *common,
        md("## Figure 7"), py(figure7_source), run_and_show("plot_figure7()", ["publication_notebooks/outputs/Figure7_pressure_contour_coalescence.png"]),
        md("## Figure 8"), py(figure8_source), run_and_show("plot_figure8()", ["publication_notebooks/outputs/Figure8_wind_and_pressure_coalescence.png"]),
        md("## Figure 9"), py(figure9_source), run_and_show("plot_figure9()", ["publication_notebooks/outputs/Figure9_cyclone_hovmoller.png"]),
        md("## Figure 10"), py(FIGURE10_SOURCE), run_and_show("plot_figure10()", ["publication_notebooks/outputs/Figure10_coalescence_event_counts.png"]),
    ])

    write_nb("03_main_figures_11_13.ipynb", [
        md("# Main Figures 11–13\n\nPersistent-anticyclone evolution, propagation, and overlap duration. Each generated figure is displayed immediately after its plotting cell."), *common,
        md("## Figure 11"), py(figure11_source), run_and_show("plot_figure11()", ["publication_notebooks/outputs/Figure11_persistent_anticyclone.png"]),
        md("## Figure 12"), py(figure12_source), run_and_show("plot_figure12()", ["publication_notebooks/outputs/Figure12_high_pressure_hovmoller.png"]),
        md("## Figure 13"), py(figure13_source), run_and_show("plot_figure13()", ["publication_notebooks/outputs/Figure13_anticyclone_overlap_duration.png"]),
    ])

    a2_a4_source = embed_script(
        "eddy/controlled_umax30_bns_analysis_20260722/scripts/build_umax30_all45_section_f.py",
        exclude_names={"make_grouped_figure", "table", "finding_lines", "write_reports", "main"},
    ).replace("import subprocess\n", "")
    a2_a4_source += r'''

def build_figures_a2_a4():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(TABLES / 'archive_vs_umax30_paired_responses.csv').sort_values(['b', 'n', 's'])
    controlled = pd.read_csv(TABLES / 'umax30_all45_headline_responses.csv').sort_values(['b', 'n', 's'])
    if len(paired) != 45 or len(controlled) != 45 or paired[['b', 'n', 's']].duplicated().any():
        raise RuntimeError('The complete 45-case matrix is not available')
    correlations, grouped = summarize(paired)
    correlations.to_csv(TABLES / 'F1_archive_controlled_parameter_correlations.csv', index=False)
    grouped.to_csv(TABLES / 'F2_archive_controlled_group_means.csv', index=False)
    make_heatmaps(controlled)
    make_paired_figure(paired)
    make_correlation_figure(correlations)
    return [
        FIGURES / 'F2_controlled_all45_heatmaps.png',
        FIGURES / 'F3_archive_controlled_paired.png',
        FIGURES / 'F4_parameter_spearman_comparison.png',
    ]
'''

    cells = [md("# Supporting Figures A1–C2\n\nControlled-speed, eddy-SLP, and post-onset RWB robustness. Plotting functions are included in the notebook and executed directly."), py(SETUP)]
    cells += source_section("Figure A1", "eddy/controlled_umax30_bns_analysis_20260722/scripts/build_figure_a1_rwb_cross_sections.py", ["eddy/controlled_umax30_bns_analysis_20260722/figures/F1a_rwb_theta_lat_environment_by_b.png", "eddy/controlled_umax30_bns_analysis_20260722/figures/F1b_original_vs_controlled_ns_group_means.png"])
    cells += source_section("Figures A2–A4", "eddy/controlled_umax30_bns_analysis_20260722/scripts/build_umax30_all45_section_f.py", ["eddy/controlled_umax30_bns_analysis_20260722/figures/F2_controlled_all45_heatmaps.png", "eddy/controlled_umax30_bns_analysis_20260722/figures/F3_archive_controlled_paired.png", "eddy/controlled_umax30_bns_analysis_20260722/figures/F4_parameter_spearman_comparison.png"], call="build_figures_a2_a4()", embedded_source=a2_a4_source)
    cells += source_section("Figure A5", "eddy/controlled_umax30_bns_analysis_20260722/original_style_comparison/scripts/make_original_style_comparison_figures.py", ["eddy/controlled_umax30_bns_analysis_20260722/original_style_comparison/figures/Figure5_original_vs_controlled_same_style.png"])
    cells += source_section("Figures B1–B4", "eddy/eddy_full_coalescence_review_analysis_20260723/scripts/aggregate_and_plot.py", ["eddy/eddy_full_coalescence_review_analysis_20260723/figures/figure06_case_level_full_eddy_scatter.png", "eddy/eddy_full_coalescence_review_analysis_20260723/figures/figure03_full_eddy_group_trends.png", "eddy/eddy_full_coalescence_review_analysis_20260723/figures/figure02_eddy_threshold_sensitivity_heatmaps.png", "eddy/eddy_full_coalescence_review_analysis_20260723/figures/figure07_eddy_vorticity_core_outcomes.png"])
    cells += source_section("Figures B5–B6", "eddy/eddy_full_coalescence_review_analysis_20260723/scripts/aggregate_controlled_full_eddy.py", ["eddy/eddy_full_coalescence_review_analysis_20260723/figures/figure11_archive_controlled_full_eddy_four_row_comparison.png", "eddy/eddy_full_coalescence_review_analysis_20260723/figures/figure12_controlled_eddy_threshold_sensitivity.png"])
    cells += source_section("Figures C1–C2", "eddy/rwb_post_onset_analysis_20260722/scripts/build_rwb_post_onset_analysis.py", ["eddy/rwb_post_onset_analysis_20260722/figures/rwb_full_post_onset_and_rate_by_parameters.png", "eddy/rwb_post_onset_analysis_20260722/figures/rwb_post_onset_rate_heatmaps.png"])
    write_nb("04_supplement_A_to_C.ipynb", cells)

    inline_relative_vorticity = select(revision_utils, {"A_EARTH", "relative_vorticity"})
    expanded_atlas_source = embed_script(
        "eddy/expanded_A1_cyclone_interaction_atlas_20260723/scripts/build_expanded_A1_atlas.py"
    )
    expanded_atlas_source = expanded_atlas_source.replace("sys.path.insert(0, str(SCRIPT_DIR))\n", "")
    expanded_atlas_source = expanded_atlas_source.replace("from revision_utils import relative_vorticity\n", "")
    expanded_atlas_source = inline_relative_vorticity + "\n\n" + expanded_atlas_source + r'''

_expanded_needed_times = needed_times
_expanded_load_case_fields = load_case_fields
_expanded_longitude_delta = longitude_delta
_expanded_make_page = make_page
'''

    primary_atlas_source = embed_script(
        "eddy/primary_secondary_first_binary_coalescence_A1_20260725/scripts/build_primary_secondary_A1_atlas.py",
        exclude_names={"load_source_module", "main"},
    )
    primary_atlas_source = primary_atlas_source.replace("import importlib.util\n", "")
    primary_atlas_source = primary_atlas_source.replace("import subprocess\n", "")
    primary_atlas_source = primary_atlas_source.replace("atlas = load_source_module()\natlas.BASE = BASE\natlas.ROWS_PER_PAGE = ROWS_PER_PAGE\n", "")
    primary_atlas_source = primary_atlas_source.replace("atlas.longitude_delta", "_expanded_longitude_delta")
    primary_atlas_source = primary_atlas_source.replace("atlas.precursor_labels = precursor_labels\n", "")
    primary_atlas_source = primary_atlas_source.replace("    atlas.ROWS_PER_PAGE = ROWS_PER_PAGE\n", "")
    primary_atlas_source = primary_atlas_source.replace("atlas.make_page", "_expanded_make_page")
    primary_atlas_source = primary_atlas_source.replace("atlas.load_case_fields", "_expanded_load_case_fields")
    primary_atlas_source = primary_atlas_source.replace("atlas.needed_times", "_expanded_needed_times")
    primary_atlas_source = primary_atlas_source.replace(
        "    temporary_pdf = combined_pdf.with_suffix('.tmp.pdf')\n"
        "    subprocess.run(['gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4', f'-sOutputFile={temporary_pdf}', *[str(path) for path in page_pdfs]], check=True)\n"
        "    temporary_pdf.replace(combined_pdf)\n",
        "    from PIL import Image as PILImage\n"
        "    page_images = [PILImage.open(path.with_suffix('.png')).convert('RGB') for path in page_pdfs]\n"
        "    if page_images:\n"
        "        page_images[0].save(combined_pdf, save_all=True, append_images=page_images[1:])\n"
        "    for page_image in page_images:\n"
        "        page_image.close()\n",
    )
    primary_atlas_source += r'''

def build_figure_e1():
    events = load_event_catalog()
    pdf, total_pages, rendered_pages = build_atlas(events)
    return pdf, total_pages, rendered_pages
'''

    inline_stationarity_helpers = select(
        revision_utils,
        {"A_EARTH", "lon_diff_deg", "great_circle_km", "save_png_pdf"},
    )
    stationarity_source = embed_script("revision_analysis/scripts/r2_2_stationarity_reinterpretation.py")
    stationarity_source = stationarity_source.replace("sys.path.insert(0, str(ROOT / 'revision_analysis' / 'scripts'))\n", "")
    stationarity_source = stationarity_source.replace("from revision_utils import great_circle_km, save_png_pdf\n", "")
    stationarity_source = inline_stationarity_helpers + "\n\n" + stationarity_source

    cells = [md("# Supporting Figures D1–G1\n\nDynamics, ten-page atlas, stationarity, and initialization. Plotting functions and project helpers are included directly in the notebook."), py(SETUP)]
    cells += source_section("Figure D1", "paper_revision/supplemental_analysis/compute_r1_12_initial_pv_cross_sections.py", ["paper_revision/supplemental_analysis/figures/r1_12_initial_pv_cross_sections.png"])
    cells += source_section("Figures D2–D4", "eddy/initial_state_analysis_20260722/scripts/build_initial_state_report_analysis.py", ["eddy/initial_state_analysis_20260722/figures/initial_cross_sections_b_variation.png", "eddy/initial_state_analysis_20260722/figures/initial_cross_sections_n_s_variations.png", "eddy/initial_state_analysis_20260722/figures/initial_predictor_response_correlation_heatmap.png"])
    cells += source_section("Figure D5", "paper_revision/supplemental_analysis/compute_r1_6_zonal_mean_jet_evolution.py", ["paper_revision/supplemental_analysis/figures/r1_6_zonal_mean_jet_evolution.png"])
    cells += source_section("Figure E1 helper", "eddy/expanded_A1_cyclone_interaction_atlas_20260723/scripts/build_expanded_A1_atlas.py", [], "pass", embedded_source=expanded_atlas_source)
    cells += source_section("Figure E1 atlas", "eddy/primary_secondary_first_binary_coalescence_A1_20260725/scripts/build_primary_secondary_A1_atlas.py", [f"eddy/primary_secondary_first_binary_coalescence_A1_20260725/pages/Figure_A1_primary_secondary_page_{n:02d}_of_10.png" for n in range(1, 11)], call="build_figure_e1()", embedded_source=primary_atlas_source)
    cells += source_section("Figure F1", "revision_analysis/scripts/r2_2_stationarity_reinterpretation.py", ["revision_analysis/figures/r2_2_supplementary_lifetime_and_centroid_motion.png"], embedded_source=stationarity_source)
    g1 = select(legacy.cells[10].source, {"ETA_INTERFACES", "ETA_MID", "PRESSURE_MID_HPA", "AnalyticJet"}) + r'''

def plot_figure_g1():
    latitude = np.linspace(-89.95, 89.95, 1800)
    levels = np.arange(0, 41, 5)
    figure, axes = plt.subplots(3, 3, figsize=(14, 9.25), sharex=True, sharey=True)
    filled = None
    panel_index = 0
    for row, b_value in enumerate((2.0, 1.5, 1.0)):
        for column, n_value in enumerate((6, 3, 1)):
            axis = axes[row, column]
            wind = AnalyticJet(b=b_value, n=n_value).zonal_wind(latitude_deg=latitude)
            filled = axis.contourf(
                latitude,
                PRESSURE_MID_HPA,
                wind,
                levels=levels,
                cmap='Reds',
                extend='max',
            )
            contours = axis.contour(
                latitude,
                PRESSURE_MID_HPA,
                wind,
                levels=levels[1:],
                colors='black',
                linewidths=0.8,
            )
            axis.clabel(contours, fmt='%d', fontsize=8)
            axis.invert_yaxis()
            axis.set_xlim(-90, 90)
            axis.set_xticks([-90, -45, 0, 45, 90])
            axis.set_yticks([100, 400, 700, 1000])
            axis.set_title(
                f'({chr(97 + panel_index)})  b={b_value:g}, n={n_value}',
                fontweight='bold',
            )
            if column == 0:
                axis.set_ylabel('Pressure (hPa)')
            if row == 2:
                axis.set_xlabel('Latitude (°)')
            panel_index += 1
    figure.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.18, wspace=0.25, hspace=0.25)
    colorbar_axis = figure.add_axes([0.27, 0.055, 0.46, 0.035])
    colorbar = figure.colorbar(filled, cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label('ua (m/s)')
    return save_figure(figure, 'FigureG1_initial_zonal_wind_profiles.png')
'''
    cells += [md("## Figure G1\n\nFormerly labeled Figure 14 in the legacy notebook."), py(select(legacy.cells[6].source, {"save_figure"})), py(g1), run_and_show("plot_figure_g1()", ["publication_notebooks/outputs/FigureG1_initial_zonal_wind_profiles.png"])]
    write_nb("05_supplement_D_to_G.ipynb", cells)


def main():
    HERE.mkdir(exist_ok=True)
    (HERE / "outputs").mkdir(exist_ok=True)
    rows = make_inventory()
    assert len(rows) == 33
    legacy = nbformat.read(LEGACY, as_version=4)
    build_notebooks(legacy)


if __name__ == "__main__":
    main()
