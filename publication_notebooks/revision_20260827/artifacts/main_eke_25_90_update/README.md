# Figure 5/7 25–90°N EKE update

This artifact contains the final figures, exact scripts, compact numerical tables, and QC notes for the August 27, 2026 EKE-domain update.

## Definitions

- EKE uses instantaneous zonal-mean departures of `u_plev` and `v_plev`.
- Horizontal averaging uses cosine-latitude area weights over 25–90°N.
- Vertical averaging uses the existing pressure-trapezoid mass weights.
- Figure 5 initial Eady rates use the same 25–90°N area–mass-weighted domain.
- Figure 7 windows are the first rising-phase crossings from 50% to 80% of each case’s maximum 25–90°N EKE over hours 1–360.
- Figure 7 `[v′T′]` profiles use the existing jet-relative ±15° diagnostic, averaged over those new case-specific windows.

## Reproduce

The scripts require the dependencies listed in `publication_notebooks/requirements-publication.txt` and the original Keeling simulation/diagnostic files. Set the data and output roots explicitly:

```bash
export PYGEN_DATA_ROOT=/data/keeling/a/mingfei5/a/data/original
export PYGEN_EKE_OUTPUT=/path/to/eke_25_90_update_output

python scripts/recompute_25_90_diagnostics.py --workers 2
python scripts/replot_figures5_7_25_90.py
```

The data root must contain the 45 standard pressure-level files, the 45 constant-`Umax` files under `priority_revision_analysis_20260720/simulations/umax30_all_bns/`, the source Section 3.2 analysis package, and the existing `analysis/upper_lower_baroclinicity/case_results/` files. The large raw model files are not included here.

## Files

- `figures/`: revised Figure 5 and Figure 7 PNG/PDF files.
- `tables/`: all-90 EKE time series, all-45 initial Eady values, all-90 windows, all-90 heat-flux profiles, and QC comparison tables.
- `scripts/`: one recomputation script and one plotting script.
- `CAPTION_DRAFTS.md`, `QC_REPORT.md`: caption and validation notes.
