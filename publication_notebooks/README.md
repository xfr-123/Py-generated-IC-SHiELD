# Publication Figure Notebooks

This directory contains the public-release notebooks and the August 2026 reviewer-stage figure update for the dry idealized baroclinic-wave experiments.

## Historical publication notebooks

The six notebooks below preserve the August 2, 2026 publication package:

- `00_inventory_and_environment.ipynb`
- `01_main_figures_02_06.ipynb`
- `02_main_figures_07_10.ipynb`
- `03_main_figures_11_13.ipynb`
- `04_supplement_A_to_C.ipynb`
- `05_supplement_D_to_G.ipynb`

The main-figure notebooks contain executed outputs. Complete rebuilding requires the original SHiELD simulation and derived-analysis files, which are too large for this Git repository.

## Current revision notebook

`06_revision_updates_20260823.ipynb` is the current reviewer-stage update. It consolidates 18 revised figure/table products, including:

- constant-$u_0$ RWB statistics without an in-figure overall title;
- NH EKE evolution with initial 30–70°N Eady-growth context;
- eddy-SLP cyclone and anticyclone diagnostics;
- panel-specific legends for the cyclone-interaction wind-evolution figure;
- 1000–850-hPa lower-layer shading in SI Figures S9–S10;
- constant-$u_0$/constant-$U_{\max}$ terminology updates;
- corrected SI colorbars and the expanded initial-PV comparison;
- recomputed eddy-SLP Figures S20–S21 and Table S3;
- EP-flux, jet-adjustment, event-atlas, and Section 3.2 diagnostics.

The associated self-contained release material is stored in `revision_20260823/`:

- final PNG and PDF figures;
- exact plotting/analysis scripts;
- compact supporting CSV/NetCDF tables;
- `revision_manifest.csv`;
- `VALIDATION.md`;
- `SHA256SUMS`.

Large raw model outputs are intentionally excluded. The copied scripts retain the original Keeling data dependencies and document the required source files.

## Validation

The revision notebook was executed on August 23, 2026 with Python 3.10 in the project `rwb` environment:

- code cells completed: 12/12;
- manifest entries: 18;
- missing linked artifacts: 0;
- SHA-256 mismatches: 0.

## Local archive rebuild

Within the complete Keeling analysis archive, rebuild the revision bundle with:

```bash
$HOME/anaconda3/envs/rwb/bin/python publication_notebooks/build_revision_update_notebook.py
```

Execute and validate the generated notebook with:

```bash
PUBLICATION_REBUILD_FIGURES=0 \
$HOME/anaconda3/envs/rwb/bin/python \
publication_notebooks/run_notebook_cells.py \
publication_notebooks/06_revision_updates_20260823.ipynb \
--status-dir publication_notebooks/validation/status_revision_20260823
```

## Latest EKE-domain update — August 27, 2026

`07_revision_updates_20260827.ipynb` and `revision_20260827/` supersede the EKE-related portions of the August 23 reviewer-stage snapshot. The update uses a 25–90°N area- and mass-weighted EKE domain, recomputes Figure 5 initial Eady annotations over the same domain, rediagnoses all Figure 7 50–80% EKE-growth windows, and reaverages the Figure 7 eddy heat-flux profiles. The earlier `revision_20260823/` directory is retained unchanged as a historical snapshot.

The new snapshot includes final PNG/PDF figures, exact scripts, compact tables for all 90 simulations, `ENVIRONMENT.md`, `VALIDATION.md`, `revision_manifest.csv`, and `SHA256SUMS`. No new dependency beyond `requirements-publication.txt` is required.
