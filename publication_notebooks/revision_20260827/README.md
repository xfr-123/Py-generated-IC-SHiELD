# August 27, 2026 EKE-domain revision

This snapshot updates the EKE and initial Eady-growth diagnostics used in Figures 5 and 7. It supersedes only the EKE-related portions of `revision_20260823`; the earlier snapshot remains unchanged as a historical record.

## Scientific changes

- EKE is recomputed over 25–90°N instead of 0–90°N.
- Figure 5 initial Eady growth-rate annotations are recomputed over 25–90°N with the same cosine-latitude area weighting and pressure-trapezoid mass weighting.
- Figure 7 panels (e–f) use the new EKE series.
- Figure 7 50–80% rising-phase crossing times are rediagnosed separately for all 90 cases.
- Figure 7 panels (g–h) use eddy heat flux `[v′T′]` averaged over the new case-specific windows.
- Figure 7 panels (a–d) retain the previously diagnosed jet-core-local wind and Eady profiles.

## Contents

`artifacts/main_eke_25_90_update/` contains the final PNG/PDF figures, exact plotting scripts, compact numerical tables, captions, and QC information. Raw SHiELD model files are not included.

## Reproduce on Keeling

The scripts use the existing `publication_notebooks/requirements-publication.txt` dependency set. Set the data and output roots explicitly:

```bash
export PYGEN_DATA_ROOT=/data/keeling/a/mingfei5/a/data/original
export PYGEN_EKE_OUTPUT=/path/to/eke_25_90_update_output

$HOME/anaconda3/envs/pygen_clean/bin/python \
  publication_notebooks/revision_20260827/artifacts/main_eke_25_90_update/scripts/recompute_25_90_diagnostics.py \
  --workers 2

$HOME/anaconda3/envs/pygen_clean/bin/python \
  publication_notebooks/revision_20260827/artifacts/main_eke_25_90_update/scripts/replot_figures5_7_25_90.py
```

The recomputation reads the 45 standard and 45 constant-`Umax` hourly files under `PYGEN_DATA_ROOT`, plus the existing compact jet-relative diagnostics under `paper_revision/analysis/upper_lower_baroclinicity/case_results/`.

## Verification

From this directory, run:

```bash
sha256sum -c SHA256SUMS
```

The publication notebook is `../07_revision_updates_20260827.ipynb`. Build its manifest and notebook with:

```bash
$HOME/anaconda3/envs/rwb/bin/python publication_notebooks/build_revision_update_20260827.py
```
