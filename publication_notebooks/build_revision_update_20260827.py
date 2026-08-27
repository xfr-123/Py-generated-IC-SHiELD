#!/usr/bin/env python3
"""Build the August 27, 2026 EKE-domain revision notebook and manifests."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "publication_notebooks" / "revision_20260827"
ARTIFACT = REVISION / "artifacts" / "main_eke_25_90_update"
NOTEBOOK = ROOT / "publication_notebooks" / "07_revision_updates_20260827.ipynb"
MANIFEST = REVISION / "revision_manifest.csv"
CHECKSUMS = REVISION / "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> None:
    figure_paths = sorted(path.relative_to(REVISION).as_posix() for path in (ARTIFACT / "figures").glob("*") if path.is_file())
    script_paths = sorted(path.relative_to(REVISION).as_posix() for path in (ARTIFACT / "scripts").glob("*.py") if path.is_file())
    table_paths = sorted(path.relative_to(REVISION).as_posix() for path in (ARTIFACT / "tables").glob("*") if path.is_file())
    rows = [{
        "id": "main_eke_25_90_update",
        "label": "Figure 5 and Figure 7 EKE-domain update",
        "title": "25–90°N EKE, initial Eady context, and revised eddy-heat-flux windows",
        "scientific_change": "Recomputes area–mass-weighted EKE over 25–90°N, recomputes Figure 5 initial Eady means over 25–90°N, and rediagnoses Figure 7 50–80% EKE-growth windows and [v′T′] profiles for all 90 simulations.",
        "figures": ";".join(figure_paths),
        "scripts": ";".join(script_paths),
        "tables": ";".join(table_paths),
    }]
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_notebook() -> None:
    figure5 = "revision_20260827/artifacts/main_eke_25_90_update/figures/Figure5_EKE_evolution_25_90N_with_Eady.png"
    figure7 = "revision_20260827/artifacts/main_eke_25_90_update/figures/Figure7_vertical_structure_eddy_activity_25_90N_vT.png"
    table_windows = "revision_20260827/artifacts/main_eke_25_90_update/tables/eke_25_90N_windows_all90.csv"
    table_checks = "revision_20260827/artifacts/main_eke_25_90_update/tables/Figure7_vT_group_mean_checks.csv"
    cells = [
        nbf.v4.new_markdown_cell(
            "# Publication Revision Update — August 27, 2026\n\n"
            "This notebook supersedes the EKE-related portions of the August 23 revision snapshot. "
            "It does not modify the manuscript, Supporting Information, or response letter. "
            "The update uses a 25–90°N domain for area–mass-weighted EKE and initial Eady context, "
            "then recomputes the Figure 7 rising-phase windows and eddy heat-flux profiles."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import hashlib\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'publication_notebooks').exists():\n"
            "    ROOT = ROOT.parent\n"
            "ASSET_ROOT = ROOT / 'publication_notebooks' / 'revision_20260827'\n"
            "manifest = pd.read_csv(ASSET_ROOT / 'revision_manifest.csv')\n"
            "manifest[['label', 'title', 'scientific_change']]"
        ),
        nbf.v4.new_markdown_cell(
            "## Updated Figure 5\n\n"
            "The EKE curves use 25–90°N cosine-latitude area weighting and pressure-trapezoid mass weighting. "
            "The inset values are initial 25–90°N area–mass-weighted mean Eady growth rates.\n\n"
            f"![Updated Figure 5]({figure5})"
        ),
        nbf.v4.new_code_cell(
            f"table_path = ROOT / 'publication_notebooks' / '{table_windows}'\n"
            "windows = pd.read_csv(table_path)\n"
            "windows.groupby('ensemble')[['eke_50pct_window_start_h', 'eke_80pct_window_end_h', 'eke_50_80pct_window_count']].describe()"
        ),
        nbf.v4.new_markdown_cell(
            "## Updated Figure 7\n\n"
            "Panels (a–d) retain the existing jet-core-local zonal-wind and Eady profiles. "
            "Panels (e–f) use 25–90°N EKE, and panels (g–h) use eddy heat flux averaged over each case’s newly diagnosed first 50–80% rising-phase interval.\n\n"
            f"![Updated Figure 7]({figure7})"
        ),
        nbf.v4.new_code_cell(
            f"checks = pd.read_csv(ROOT / 'publication_notebooks' / '{table_checks}')\n"
            "checks"
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibility\n\n"
            "The exact scripts and compact data tables are stored under `revision_20260827/artifacts/main_eke_25_90_update/`. "
            "The scripts accept `PYGEN_DATA_ROOT` for the Keeling archive, `PYGEN_EKE_OUTPUT` for the output directory, "
            "and `PYGEN_OLD_FIGURE7_PACKAGE` for the previous 0–90°N window table.\n\n"
            "```bash\n"
            "PYGEN_DATA_ROOT=/data/keeling/a/mingfei5/a/data/original \\\n"
            "PYGEN_EKE_OUTPUT=/path/to/output \\\n"
            "$HOME/anaconda3/envs/pygen_clean/bin/python \\\n"
            "revision_20260827/artifacts/main_eke_25_90_update/scripts/recompute_25_90_diagnostics.py --workers 2\n"
            "PYGEN_DATA_ROOT=/data/keeling/a/mingfei5/a/data/original \\\n"
            "PYGEN_EKE_OUTPUT=/path/to/output \\\n"
            "$HOME/anaconda3/envs/pygen_clean/bin/python \\\n"
            "revision_20260827/artifacts/main_eke_25_90_update/scripts/replot_figures5_7_25_90.py\n"
            "```"
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.write(notebook, NOTEBOOK)


def build_checksums() -> None:
    files = sorted(path for path in REVISION.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    CHECKSUMS.write_text("\n".join(f"{sha256(path)}  {path.relative_to(REVISION)}" for path in files) + "\n", encoding="utf-8")


def main() -> None:
    if not ARTIFACT.exists():
        raise FileNotFoundError(ARTIFACT)
    build_manifest()
    build_notebook()
    (REVISION / "VALIDATION.md").write_text(
        "# August 27, 2026 validation\n\n"
        "- The update contains the 25–90°N EKE/Eady recalculation and the corresponding Figure 7 window/heat-flux update.\n"
        "- 90 cases and 32,400 hourly EKE records are included in the compact tables.\n"
        "- The large source model files remain external Keeling inputs and are not included.\n"
        "- Figure 5 and Figure 7 PNG/PDF files were visually inspected before this snapshot was built.\n"
        "- Run `sha256sum -c SHA256SUMS` from this directory to validate the snapshot.\n",
        encoding="utf-8",
    )
    build_checksums()
    print(NOTEBOOK)
    print(MANIFEST)
    print(CHECKSUMS)


if __name__ == "__main__":
    main()
