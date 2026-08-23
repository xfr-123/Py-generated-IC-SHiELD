#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication_notebooks" / "revision_20260823"
NOTEBOOK = ROOT / "publication_notebooks" / "06_revision_updates_20260823.ipynb"
MANIFEST = OUT / "revision_manifest.csv"

ARTIFACTS = [
    {
        "id": "main_rwb_constant_u0",
        "label": "Current Figure 6",
        "title": "RWB statistics for the constant-u0 ensemble",
        "change": "Uses only the 45 constant-u0 simulations and removes the in-figure overall title.",
        "figures": [
            "publication_notebooks/outputs/Figure6_RWB_fraction_constant_u0_no_suptitle.png",
            "publication_notebooks/outputs/Figure6_RWB_fraction_constant_u0_no_suptitle.pdf",
        ],
        "scripts": ["publication_notebooks/replot_figure5_rwb_constant_u0_no_suptitle.py"],
        "tables": [],
    },
    {
        "id": "main_eke_midlatitude_eady",
        "label": "EKE evolution revision",
        "title": "NH EKE evolution with initial 30–70°N Eady-growth context",
        "change": "Adds experiment legends and initial midlatitude Eady-growth annotations using the NH-consistent diagnostic.",
        "figures": [
            "paper_revision/eke_evolution_midlatitude_eady_20260820/Figure6_EKE_evolution_with_midlatitude_Eady_mean.png",
            "paper_revision/eke_evolution_midlatitude_eady_20260820/Figure6_EKE_evolution_with_midlatitude_Eady_mean.pdf",
        ],
        "scripts": ["paper_revision/eke_evolution_midlatitude_eady_20260820/plot_figure6_eke_with_midlatitude_eady.py"],
        "tables": ["paper_revision/eke_evolution_midlatitude_eady_20260820/Figure6_initial_midlatitude_Eady_annotations.csv"],
    },
    {
        "id": "main_cyclone_overview",
        "label": "Current Figure 9",
        "title": "Cyclone interaction from eddy surface pressure",
        "change": "Combines the hemispheric evolution and the b2n3s10 local zoom using eddy-SLP contours.",
        "figures": [
            "paper_revision/cyclone_interaction_reorganization_20260820/figures/Figure9_eddy_cyclone_overview_plus_zoom.png",
            "paper_revision/cyclone_interaction_reorganization_20260820/figures/Figure9_eddy_cyclone_overview_plus_zoom.pdf",
        ],
        "scripts": ["paper_revision/cyclone_interaction_reorganization_20260820/scripts/reorganize_eddy_cyclone_figures.py"],
        "tables": [],
    },
    {
        "id": "main_cyclone_wind_evolution",
        "label": "Current Figure 10",
        "title": "1000-hPa wind evolution around eddy-contour connection",
        "change": "Separates the time-series figure and adds panel-specific legends for s, n, and b.",
        "figures": [
            "paper_revision/cyclone_interaction_reorganization_20260820/figures/Figure10_eddy_cyclone_wind_evolution_timeseries.png",
            "paper_revision/cyclone_interaction_reorganization_20260820/figures/Figure10_eddy_cyclone_wind_evolution_timeseries.pdf",
        ],
        "scripts": ["paper_revision/cyclone_interaction_reorganization_20260820/scripts/reorganize_eddy_cyclone_figures.py"],
        "tables": ["paper_revision/eddy_field_replot_20260818/tables/Figure8_eddy_wind_speed_series_standard.csv"],
    },
    {
        "id": "main_persistent_anticyclone",
        "label": "Current Figure 11",
        "title": "Persistent-anticyclone evolution diagnosed from eddy SLP",
        "change": "Uses eddy-SLP contours, improved spacing, corrected sectors, and two high-center labels in the local panels.",
        "figures": [
            "paper_revision/anticyclone_figure11_eddyfield_replot_20260819_v7/figures/Figure11_eddy_persistent_anticyclone.png",
            "paper_revision/anticyclone_figure11_eddyfield_replot_20260819_v7/figures/Figure11_eddy_persistent_anticyclone.pdf",
        ],
        "scripts": ["paper_revision/anticyclone_figure11_eddyfield_replot_20260819_v7/scripts/plot_figure11_eddy_qc.py"],
        "tables": ["paper_revision/anticyclone_figure11_eddyfield_replot_20260819_v7/tables/Figure11_eddy_panel_metadata.csv"],
    },
    {
        "id": "main_high_pressure_hovmoller",
        "label": "Current Figure 12",
        "title": "High-pressure eddy-SLP Hovmöller diagrams",
        "change": "Uses 2-hPa intervals, a ±2-hPa white center, ±10-hPa limits, and corrected b-axis labels.",
        "figures": [
            "paper_revision/hovmoller_2hpa_pm10_blabels_replot_20260819/figures/Figure12_eddy_high_pressure_hovmoller_2hpa_pm10_blabels.png",
            "paper_revision/hovmoller_2hpa_pm10_blabels_replot_20260819/figures/Figure12_eddy_high_pressure_hovmoller_2hpa_pm10_blabels.pdf",
        ],
        "scripts": ["paper_revision/hovmoller_2hpa_pm10_blabels_replot_20260819/scripts/replot_figure12_eddy_hovmoller_2hpa_pm10_blabels.py"],
        "tables": [],
    },
    {
        "id": "main_heatmaps_james_qc",
        "label": "Heatmap presentation revision",
        "title": "Eddy-SLP cyclone counts and anticyclone duration",
        "change": "Applies the final JAMES-style panel labels, removes oversized suptitles, and retains the eddy-field diagnostics.",
        "figures": [
            "paper_revision/count_duration_heatmaps_james_qc_20260819/figures/Figure6_eddy_slp_coalescence_heatmaps.png",
            "paper_revision/count_duration_heatmaps_james_qc_20260819/figures/Figure6_eddy_slp_coalescence_heatmaps.pdf",
            "paper_revision/count_duration_heatmaps_james_qc_20260819/figures/Figure13_eddy_anticyclone_overlap_duration.png",
            "paper_revision/count_duration_heatmaps_james_qc_20260819/figures/Figure13_eddy_anticyclone_overlap_duration.pdf",
        ],
        "scripts": ["paper_revision/count_duration_heatmaps_james_qc_20260819/scripts/replot_heatmaps_james_qc.py"],
        "tables": [],
    },
    {
        "id": "si_s9_s10_upper_lower",
        "label": "SI Figures S9–S10",
        "title": "Initial and growth-phase upper/lower baroclinicity profiles",
        "change": "Uses 1000–850 hPa for the lower-layer shading and labels the ensembles constant u0 and constant Umax = 30 m s−1.",
        "figures": [
            "analysis/upper_lower_baroclinicity/figures/initial_vertical_structure_standard_vs_u30_1000_850.png",
            "analysis/upper_lower_baroclinicity/figures/initial_vertical_structure_standard_vs_u30_1000_850.pdf",
            "analysis/upper_lower_baroclinicity/figures/eddy_flux_vertical_structure_standard_vs_u30_1000_850.png",
            "analysis/upper_lower_baroclinicity/figures/eddy_flux_vertical_structure_standard_vs_u30_1000_850.pdf",
        ],
        "scripts": ["analysis/upper_lower_baroclinicity/scripts/make_upper_lower_baroclinicity_figures.py"],
        "tables": [
            "analysis/upper_lower_baroclinicity/initial_vertical_profiles.nc",
            "analysis/upper_lower_baroclinicity/growth_stage_flux_profiles.nc",
            "analysis/upper_lower_baroclinicity/upper_lower_baroclinicity_profile_statistics.csv",
        ],
    },
    {
        "id": "si_s11_rwb_two_ensembles",
        "label": "SI Figure S11",
        "title": "RWB orientation fractions in the two ensembles",
        "change": "Updates visible terminology to constant u0 and constant Umax = 30 m s−1.",
        "figures": [
            "eddy/controlled_umax30_bns_analysis_20260722/original_style_comparison/figures/Figure5_original_vs_controlled_same_style.png",
            "eddy/controlled_umax30_bns_analysis_20260722/original_style_comparison/figures/Figure5_original_vs_controlled_same_style.pdf",
        ],
        "scripts": ["eddy/controlled_umax30_bns_analysis_20260722/original_style_comparison/scripts/make_original_style_comparison_figures.py"],
        "tables": ["eddy/controlled_umax30_bns_analysis_20260722/original_style_comparison/tables/controlled_rwb_direct_counts_all45.csv"],
    },
    {
        "id": "si_s12_s13_colorbars",
        "label": "SI Figures S12–S13",
        "title": "Initial cross sections with corrected colorbars",
        "change": "Repositions colorbars without changing the underlying initial-state or conversion diagnostics.",
        "figures": [
            "paper_revision/figure_s12_colorbar_fix_20260820/figures/Figure_S12_initial_cross_sections_b_variation_colorbar_fixed.png",
            "paper_revision/figure_s12_colorbar_fix_20260820/figures/Figure_S12_initial_cross_sections_b_variation_colorbar_fixed.pdf",
            "paper_revision/figure_s13_colorbar_fix_20260820/figures/Figure_S13_initial_cross_sections_n_s_variations_colorbar_fixed.png",
            "paper_revision/figure_s13_colorbar_fix_20260820/figures/Figure_S13_initial_cross_sections_n_s_variations_colorbar_fixed.pdf",
        ],
        "scripts": [
            "paper_revision/figure_s12_colorbar_fix_20260820/scripts/fix_figure_s12_colorbars.py",
            "paper_revision/figure_s13_colorbar_fix_20260820/scripts/fix_figure_s13_colorbar.py",
        ],
        "tables": [],
    },
    {
        "id": "si_s14_two_ensemble_correlations",
        "label": "SI Figure S14",
        "title": "Constant-u0 and constant-Umax relationship comparison",
        "change": "Provides the two-ensemble predictor/response comparison with consistent terminology.",
        "figures": [
            "paper_revision/figure_S14_two_ensembles_20260818/Figure_S14_constant_u0_vs_constant_Umax.png",
            "paper_revision/figure_S14_two_ensembles_20260818/Figure_S14_constant_u0_vs_constant_Umax.pdf",
        ],
        "scripts": ["paper_revision/figure_S14_two_ensembles_20260818/build_figure_s14_two_ensembles.py"],
        "tables": [
            "paper_revision/figure_S14_two_ensembles_20260818/Figure_S14_two_ensemble_spearman_long.csv",
            "paper_revision/figure_S14_two_ensembles_20260818/Figure_S14_constant_u0_correlation_matrix.csv",
            "paper_revision/figure_S14_two_ensembles_20260818/Figure_S14_constant_Umax_30_ms_correlation_matrix.csv",
        ],
    },
    {
        "id": "si_s15_jet_adjustment",
        "label": "SI Figure S15",
        "title": "Standard-ensemble jet and low-level-baroclinicity adjustment",
        "change": "Combines representative cross sections with independently verified Day-8/Day-15 statistics for all 45 standard cases.",
        "figures": [
            "paper_revision/supplemental_analysis/figure_S15_standard_ensemble_update_20260813/Figure_S15_standard_ensemble_jet_adjustment.png",
            "paper_revision/supplemental_analysis/figure_S15_standard_ensemble_update_20260813/Figure_S15_standard_ensemble_jet_adjustment.pdf",
        ],
        "scripts": ["paper_revision/supplemental_analysis/figure_S15_standard_ensemble_update_20260813/make_figure_S15_standard_ensemble_update.py"],
        "tables": ["paper_revision/supplemental_analysis/figure_S15_standard_ensemble_update_20260813/Figure_S15_plotted_values_all45.csv"],
    },
    {
        "id": "si_s19_primary_secondary_atlas",
        "label": "SI Figure S19",
        "title": "Primary/secondary cyclone-interaction atlas",
        "change": "Reformats the 44-case atlas from four to eight rows per page and separates the colorbar from the lowest row.",
        "figures": [
            f"paper_revision/primary_secondary_atlas_8rows_20260820/pages/Figure_A1_primary_secondary_8rows_page_{index:02d}_of_05.png"
            for index in range(1, 6)
        ],
        "scripts": ["paper_revision/primary_secondary_atlas_8rows_20260820/scripts/build_primary_secondary_atlas_8rows.py"],
        "tables": ["paper_revision/primary_secondary_atlas_8rows_20260820/tables/Figure_A1_primary_secondary_atlas_8rows_crosswalk.csv"],
    },
    {
        "id": "si_s20_s21_eddy_slp",
        "label": "SI Figures S20–S21",
        "title": "Eddy-SLP anticyclone tracking and definition sensitivity",
        "change": "Recomputes objects, tracks, lifetimes, mobility, and sensitivity statistics from eddy surface pressure rather than full SLP.",
        "figures": [
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/Figure_S20_eddy_slp_lifetime_centroid_speed.png",
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/Figure_S20_eddy_slp_lifetime_centroid_speed.pdf",
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/Figure_S21_eddy_slp_definition_sensitivity.png",
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/Figure_S21_eddy_slp_definition_sensitivity.pdf",
        ],
        "scripts": ["paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/recompute_figures_s20_s21_eddy_slp.py"],
        "tables": [
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/S20_case_level_results.csv",
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/S21_grouped_results.csv",
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/S21_definition_audit.csv",
            "paper_revision/Figures_S20_S21_eddy_slp_recomputed_20260821/S21_baseline_validation.csv",
        ],
    },
    {
        "id": "table_s3_eddy_slp",
        "label": "Table S3",
        "title": "Eddy-SLP persistence, size, intensity, and centroid mobility",
        "change": "Recomputes the fixed n=1, s=+10° table consistently with the eddy-SLP tracker used for Figures S20–S21.",
        "figures": [],
        "scripts": ["paper_revision/Table_S3_eddy_slp_recomputed_20260821/recompute_table_s3_eddy_slp.py"],
        "tables": [
            "paper_revision/Table_S3_eddy_slp_recomputed_20260821/Table_S3_eddy_slp_recomputed.csv",
            "paper_revision/Table_S3_eddy_slp_recomputed_20260821/Table_S3_old_full_field_vs_new_eddy_slp.csv",
            "paper_revision/Table_S3_eddy_slp_recomputed_20260821/Table_S3_validation_against_Figure_S20.csv",
        ],
    },
    {
        "id": "ep_flux_evolution",
        "label": "EP-flux supplementary diagnostic",
        "title": "EP-flux evolution during the baroclinic-wave life cycle",
        "change": "Uses pressure-coordinate QG EP flux, corrected log-pressure presentation, less-dense lower-level vectors, and common scales.",
        "figures": [
            "paper_revision/supplemental_analysis/figures/r1_6_ep_flux_evolution.png",
            "paper_revision/supplemental_analysis/figures/r1_6_ep_flux_evolution.pdf",
        ],
        "scripts": ["paper_revision/supplemental_analysis/compute_r1_6_ep_flux_evolution.py"],
        "tables": [],
    },
    {
        "id": "initial_pv_bns",
        "label": "Initial-PV supplementary diagnostic",
        "title": "Initial PV and dynamical-tropopause structure across b, n, and s",
        "change": "Expands the original three-panel latitude-shift comparison to a controlled 3×3 b/n/s comparison with shared levels and labeled theta contours.",
        "figures": [
            "paper_revision/supplemental_analysis/figures/r1_11_initial_pv_cross_sections_bns.png",
            "paper_revision/supplemental_analysis/figures/r1_11_initial_pv_cross_sections_bns.pdf",
        ],
        "scripts": ["paper_revision/supplemental_analysis/compute_r1_11_initial_pv_cross_sections_bns.py"],
        "tables": [],
    },
    {
        "id": "section32_combined",
        "label": "Section 3.2 consolidated diagnostic",
        "title": "Vertical profiles, Eady growth, NH EKE, and mature-stage eddy fluxes",
        "change": "Consolidates four two-column comparisons with consistent panel numbering and constant-u0/constant-Umax terminology.",
        "figures": [
            "paper_revision/section32_vertical_profiles_eke_fluxes_20260820/figures/section32_four_part_combined_8panels.png",
            "paper_revision/section32_vertical_profiles_eke_fluxes_20260820/figures/section32_four_part_combined_8panels.pdf",
        ],
        "scripts": ["paper_revision/section32_vertical_profiles_eke_fluxes_20260820/scripts/make_section32_figures.py"],
        "tables": [
            "paper_revision/section32_vertical_profiles_eke_fluxes_20260820/data/initial_profiles_at_jet_core.csv",
            "paper_revision/section32_vertical_profiles_eke_fluxes_20260820/data/eke_nh_area_mass_weighted_timeseries_all90.csv",
            "paper_revision/section32_vertical_profiles_eke_fluxes_20260820/data/eddy_flux_profiles_50_80pct_peak_eke_all90.csv",
        ],
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(relative: str, artifact_id: str, kind: str) -> str:
    source = ROOT / relative
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir = OUT / "artifacts" / artifact_id / kind
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if not destination.exists() or sha256(destination) != sha256(source):
        shutil.copy2(source, destination)
    return destination.relative_to(ROOT / "publication_notebooks").as_posix()


def build_assets() -> list[dict[str, str]]:
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for artifact in ARTIFACTS:
        copied = {"figures": [], "scripts": [], "tables": []}
        for kind in copied:
            for relative in artifact[kind]:
                copied[kind].append(copy_file(relative, artifact["id"], kind))
        rows.append(
            {
                "id": artifact["id"],
                "label": artifact["label"],
                "title": artifact["title"],
                "scientific_or_presentation_change": artifact["change"],
                "figures": ";".join(copied["figures"]),
                "scripts": ";".join(copied["scripts"]),
                "tables": ";".join(copied["tables"]),
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def image_markdown(path: str, title: str) -> str:
    return f"![{title}]({path})"


def build_notebook(rows: list[dict[str, str]]) -> None:
    notebook = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Publication Revision Update — August 23, 2026\n\n"
            "This notebook consolidates the figure and table updates produced during the current revision round. "
            "The six earlier publication notebooks are retained as a historical reproducibility record; this notebook identifies the revised artifacts that supersede their corresponding presentation or diagnostic definitions.\n\n"
            "Key terminology is standardized to **constant $u_0$** and **constant $U_{\\max}=30$ m s$^{-1}$**. "
            "Cyclone and anticyclone feature diagnostics use the eddy surface-pressure field where specified. "
            "No manuscript or Word document is generated or edited by this notebook."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'publication_notebooks').exists():\n"
            "    ROOT = ROOT.parent\n"
            "ASSET_ROOT = ROOT / 'publication_notebooks' / 'revision_20260823'\n"
            "manifest = pd.read_csv(ASSET_ROOT / 'revision_manifest.csv')\n"
            "manifest[['label', 'title', 'scientific_or_presentation_change']]"
        ),
    ]
    for row in rows:
        scripts = [item for item in row["scripts"].split(";") if item]
        tables = [item for item in row["tables"].split(";") if item]
        figures = [item for item in row["figures"].split(";") if item]
        source_lines = [f"- Script: `{path}`" for path in scripts]
        source_lines += [f"- Supporting data: `{path}`" for path in tables]
        text = f"## {row['label']}: {row['title']}\n\n{row['scientific_or_presentation_change']}\n\n" + "\n".join(source_lines)
        pngs = [path for path in figures if path.lower().endswith(".png")]
        if pngs:
            text += "\n\n" + "\n\n".join(image_markdown(path, row["title"]) for path in pngs)
        cells.append(nbf.v4.new_markdown_cell(text))
        if tables:
            csv_tables = [path for path in tables if path.lower().endswith(".csv")]
            if csv_tables:
                cells.append(
                    nbf.v4.new_code_cell(
                        "table_path = ROOT / 'publication_notebooks' / " + repr(csv_tables[0]) + "\n"
                        "pd.read_csv(table_path).head(10)"
                    )
                )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Reproduction and provenance\n\n"
            "The exact scripts and compact supporting tables are stored under `publication_notebooks/revision_20260823/artifacts/`. "
            "Large raw SHiELD NetCDF files are intentionally not duplicated in the Git repository. "
            "Scripts retain the original Keeling data dependencies documented in the project README and individual analysis reports. "
            "`SHA256SUMS` records every file included in this revision bundle."
        )
    )
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    nbf.write(notebook, NOTEBOOK)


def write_readme(rows: list[dict[str, str]]) -> None:
    lines = [
        "# August 2026 Revision Update",
        "",
        "This directory accompanies `../06_revision_updates_20260823.ipynb` and contains the final revised figures, exact plotting/analysis scripts, and compact supporting tables assembled on August 23, 2026.",
        "",
        "The original multi-GB SHiELD simulation files are not included. Paths and required source datasets remain documented in the copied scripts and analysis reports.",
        "",
        "## Included artifacts",
        "",
    ]
    for row in rows:
        lines.append(f"- **{row['label']} — {row['title']}**: {row['scientific_or_presentation_change']}")
    lines.extend(["", "See `revision_manifest.csv` and `SHA256SUMS` for the complete machine-readable inventory.", ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_checksums() -> None:
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256(path)}  {path.relative_to(OUT).as_posix()}" for path in files]
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_assets()
    build_notebook(rows)
    write_readme(rows)
    write_checksums()
    print(NOTEBOOK)
    print(OUT)
    print(f"artifacts={len(rows)}")


if __name__ == "__main__":
    main()
