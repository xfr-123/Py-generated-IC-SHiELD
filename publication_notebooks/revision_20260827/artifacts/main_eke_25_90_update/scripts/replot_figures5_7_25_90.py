#!/usr/bin/env python3
"""Draw revised Figures 5 and 7 from the recomputed 25–90°N diagnostics."""
from __future__ import annotations

import hashlib
import os
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get('PYGEN_DATA_ROOT', '/data/keeling/a/mingfei5/a/data/original'))
SOURCE_PACKAGE = ROOT / 'paper_revision' / 'section32_vertical_profiles_eke_fluxes_20260820'
SOURCE_SCRIPT = SOURCE_PACKAGE / 'scripts' / 'make_section32_figures.py'
OLD_FIGURE7_PACKAGE = Path(os.environ.get('PYGEN_OLD_FIGURE7_PACKAGE', str(ROOT / 'paper_revision' / 'figure7_vT_bottom_row_20260825')))
OUTPUT = Path(os.environ.get('PYGEN_EKE_OUTPUT', str(ROOT / 'paper_revision' / 'eke_25_90_update_20260825')))
FIGURES = OUTPUT / 'figures'
TABLES = OUTPUT / 'tables'
EKE_COLUMN = 'eke_25_90N_area_mass_weighted_m2_s-2'
EADY_COLUMN = 'initial_eady_25_90N_area_mass_weighted_day-1'
B_VALUES_F5 = [2.0, 1.5, 1.0]
N_VALUES_F5 = [6, 3, 1]
S_VALUES_F5 = [0, 10, -10]
S_COLORS = {0: 'blue', 10: 'red', -10: 'green'}
S_LINESTYLES = {0: 'solid', 10: 'dashed', -10: 'dotted'}


def load_source_module():
    specification = importlib.util.spec_from_file_location('section32_plot_source_25_90', SOURCE_SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(f'Cannot import {SOURCE_SCRIPT}')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def nice_upper_limit(values, minimum, step):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return max(minimum, math.ceil(float(finite.max()) / step) * step)


def common_limits(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    span = high - low
    return (low - 0.07 * span, high + 0.07 * span) if span > 0 else (-1.0, 1.0)


def make_figure5(eke, eady):
    standard = eke[eke.ensemble.eq('standard')].copy()
    plotted = standard[standard.s.isin(S_VALUES_F5) & standard.time_h.ge(144)]
    y_max = nice_upper_limit(plotted[EKE_COLUMN], 50.0, 10.0)
    y_ticks = np.arange(0.0, y_max + 0.1, 20.0)
    if y_ticks[-1] < y_max:
        y_ticks = np.append(y_ticks, y_max)
    figure, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True, sharey=True, facecolor='white')
    panel_index = 0
    for row, b_value in enumerate(B_VALUES_F5):
        for column, n_value in enumerate(N_VALUES_F5):
            axis = axes[row, column]
            for s_value in S_VALUES_F5:
                selected = standard[
                    np.isclose(standard.b.astype(float), b_value)
                    & standard.n.eq(n_value)
                    & standard.s.eq(s_value)
                    & standard.time_h.ge(144)
                ]
                if len(selected) != 217:
                    raise RuntimeError(f'Unexpected Figure 5 samples for b={b_value}, n={n_value}, s={s_value}')
                axis.plot(selected.time_h, selected[EKE_COLUMN], color=S_COLORS[s_value],
                          linestyle=S_LINESTYLES[s_value], linewidth=2.2)
            panel_eady = eady[np.isclose(eady.b.astype(float), b_value) & eady.n.eq(n_value)].set_index('s')
            box = FancyBboxPatch((0.025, 0.515), 0.72, 0.435, transform=axis.transAxes,
                                 boxstyle='round,pad=0.012,rounding_size=0.012', facecolor='white',
                                 edgecolor='0.72', linewidth=0.8, alpha=0.92, zorder=3)
            axis.add_patch(box)
            axis.text(0.05, 0.895, 'Initial 25–90°N mean Eady rate', transform=axis.transAxes,
                      ha='left', va='center', fontsize=13.2, fontweight='semibold', zorder=4)
            axis.text(0.05, 0.835, '25–90°N; area–mass weighted (day⁻¹)', transform=axis.transAxes,
                      ha='left', va='center', fontsize=10.4, color='0.25', zorder=4)
            for y_position, s_value in zip((0.745, 0.655, 0.565), S_VALUES_F5):
                eady_value = float(panel_eady.loc[s_value, EADY_COLUMN])
                axis.plot([0.05, 0.13], [y_position, y_position], transform=axis.transAxes,
                          color=S_COLORS[s_value], linestyle=S_LINESTYLES[s_value], linewidth=2.4,
                          solid_capstyle='butt', clip_on=False, zorder=4)
                s_label = rf'$s={s_value:+d}^\circ$' if s_value else r'$s=0^\circ$'
                axis.text(0.15, y_position, rf'{s_label}   {eady_value:.3f}', transform=axis.transAxes,
                          ha='left', va='center', fontsize=12.8, color=S_COLORS[s_value], zorder=4)
            letter = chr(ord('a') + panel_index)
            panel_index += 1
            axis.set_title(rf'({letter}) $b={b_value:g}$, $n={n_value}$', fontsize=20, weight='bold', pad=6)
            axis.set_xlim(144, 360)
            axis.set_ylim(0, y_max)
            axis.set_xticks([150, 200, 250, 300, 350])
            axis.set_yticks(y_ticks)
            axis.tick_params(labelsize=20)
            for hour in range(144, 361, 48):
                axis.axvline(hour, color='lightgray', linewidth=0.5, zorder=0)
            for value in y_ticks:
                axis.axhline(value, color='lightgray', linewidth=0.5, linestyle='--', zorder=0)
    legend_handles = [Line2D([0], [0], color=S_COLORS[s_value], linestyle=S_LINESTYLES[s_value],
                             linewidth=3, label=rf'$s={s_value:+d}^\circ$' if s_value else r'$s=0^\circ$')
                      for s_value in S_VALUES_F5]
    figure.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.535, 0.992), ncol=3,
                  frameon=False, fontsize=20, handlelength=3.0, columnspacing=2.2, handletextpad=0.7)
    figure.supxlabel('Time (h)', fontsize=24, y=0.04)
    figure.supylabel('25–90°N area- and mass-weighted EKE\n' + r'(m$^2$ s$^{-2}$)', fontsize=22, x=0.012)
    figure.subplots_adjust(left=0.095, right=0.99, bottom=0.09, top=0.885, wspace=0.10, hspace=0.22)
    png = FIGURES / 'Figure5_EKE_evolution_25_90N_with_Eady.png'
    pdf = FIGURES / 'Figure5_EKE_evolution_25_90N_with_Eady.pdf'
    figure.savefig(png, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.03)
    figure.savefig(pdf, facecolor='white', bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)
    return png, pdf, y_max


def make_figure7(source, initial, eke, heat_flux):
    initial_frame = initial[initial.pressure_hpa.between(source.PRESSURE_MIN, source.PRESSURE_MAX)].copy()
    wind_values = initial_frame['initial_zonal_wind_at_jet_core_m_s-1'].to_numpy(float)
    wind_limits = (min(-2.0, float(np.nanmin(wind_values)) - 1.0), float(np.nanmax(wind_values)) + 2.0)
    eady_limits = (0.0, float(np.nanmax(initial_frame['initial_eady_growth_rate_day-1'])) * 1.08)
    heat_frame = heat_flux[heat_flux.pressure_hpa.between(source.PRESSURE_MIN, source.PRESSURE_MAX)].copy()
    heat_limits = common_limits(heat_frame.value)
    eke_ymax = nice_upper_limit(eke[EKE_COLUMN], 100.0, 20.0)
    figure, axes = plt.subplots(4, 2, figsize=(7.35, 11.9), constrained_layout=False)
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.058, top=0.895, hspace=0.47, wspace=0.20)
    labels = list('abcdefgh')
    for column, ensemble in enumerate(source.ENSEMBLE_ORDER):
        axis = axes[0, column]
        source.draw_profile_group(axis, initial_frame, ensemble, 'initial_zonal_wind_at_jet_core_m_s-1',
                                  x_limits=wind_limits, x_label=r'Zonal wind (m s$^{-1}$)')
        source.add_panel_label(axis, labels[column])
        axis.set_title(source.ENSEMBLE_LABELS[ensemble], pad=4, fontsize=10.5)
        axis = axes[1, column]
        source.draw_profile_group(axis, initial_frame, ensemble, 'initial_eady_growth_rate_day-1',
                                  x_limits=eady_limits, top_wind_axis=True,
                                  x_label=r'Initial Eady growth rate (day$^{-1}$)')
        source.add_panel_label(axis, labels[2 + column])
        axis = axes[2, column]
        ensemble_eke = eke[eke.ensemble.eq(ensemble)]
        for b_value in source.B_VALUES:
            b_subset = ensemble_eke[np.isclose(ensemble_eke.b.astype(float), float(b_value))]
            for _, case_data in b_subset.groupby('case', sort=False):
                n_value = int(case_data.n.iloc[0])
                axis.plot(case_data.time_h, case_data[EKE_COLUMN], color=source.B_COLORS[b_value],
                          ls=source.N_LINESTYLES[n_value], lw=0.45, alpha=0.22, zorder=2)
            mean_series = b_subset.groupby('time_h', as_index=False)[EKE_COLUMN].mean()
            axis.plot(mean_series.time_h, mean_series[EKE_COLUMN], color=source.B_COLORS[b_value], lw=2.2, zorder=4)
        axis.set_xlim(144, 360)
        axis.set_ylim(0, eke_ymax)
        axis.set_xticks([150, 200, 250, 300, 350])
        axis.grid(True, color='#D9D9D9', lw=0.5, alpha=0.55)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(labelsize=8.5)
        axis.set_xlabel('Time (h)')
        for hour in range(144, 361, 48):
            axis.axvline(hour, color='#BEBEBE', lw=0.6, ls='--', zorder=0)
        source.add_panel_label(axis, labels[4 + column])
        axis = axes[3, column]
        source.draw_profile_group(axis, heat_frame[heat_frame.ensemble.eq(ensemble)], ensemble, 'value',
                                  scale=1.0, x_limits=heat_limits, top_wind_axis=False,
                                  x_label=r'$\overline{v\prime T\prime}$ (K m s$^{-1}$)')
        source.add_panel_label(axis, labels[6 + column])
        if column == 1:
            for row in range(4):
                axes[row, column].tick_params(labelleft=False)
    axes[2, 0].set_ylabel('25–90°N area- and\nmass-weighted EKE\n' + r'(m$^2$ s$^{-2}$)')
    handles = source.b_legend() + [
        Patch(facecolor=source.LOWER_SHADE, edgecolor='none', alpha=0.58, label='1000–850 hPa'),
        Patch(facecolor=source.UPPER_SHADE, edgecolor='none', alpha=0.54, label='500–300 hPa'),
        Line2D([0], [0], color='#444444', lw=1.2, ls=source.N_LINESTYLES[1], label=r'$n=1$'),
        Line2D([0], [0], color='#444444', lw=1.2, ls=source.N_LINESTYLES[3], label=r'$n=3$'),
        Line2D([0], [0], color='#444444', lw=1.2, ls=source.N_LINESTYLES[6], label=r'$n=6$'),
    ]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.56, 0.945), ncol=8,
                  frameon=False, handlelength=1.8, columnspacing=0.75, fontsize=8.3)
    png = FIGURES / 'Figure7_vertical_structure_eddy_activity_25_90N_vT.png'
    pdf = FIGURES / 'Figure7_vertical_structure_eddy_activity_25_90N_vT.pdf'
    figure.savefig(png, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.04)
    figure.savefig(pdf, facecolor='white', bbox_inches='tight', pad_inches=0.04)
    plt.close(figure)
    return png, pdf, heat_limits, eke_ymax


def make_supporting_outputs(eke, summary, eady, heat_flux, f5, f7):
    old_windows_path = OLD_FIGURE7_PACKAGE / 'tables' / 'Figure7_bottom_panels_vT_windows.csv'
    if old_windows_path.exists():
        old = pd.read_csv(old_windows_path).rename(columns={
            'window_start_h': 'old_0_90N_window_start_h', 'window_end_h': 'old_0_90N_window_end_h',
            'window_count': 'old_0_90N_window_count'})
        new = summary.rename(columns={
            'eke_50pct_window_start_h': 'new_25_90N_window_start_h',
            'eke_80pct_window_end_h': 'new_25_90N_window_end_h',
            'eke_50_80pct_window_count': 'new_25_90N_window_count'})
        columns = ['ensemble', 'case', 'old_0_90N_window_start_h', 'old_0_90N_window_end_h', 'old_0_90N_window_count']
        comparison = new.merge(old[columns], on=['ensemble', 'case'], how='left', validate='one_to_one')
        comparison['start_shift_h'] = comparison.new_25_90N_window_start_h - comparison.old_0_90N_window_start_h
        comparison['end_shift_h'] = comparison.new_25_90N_window_end_h - comparison.old_0_90N_window_end_h
        comparison.to_csv(TABLES / 'Figure7_window_change_0_90N_to_25_90N.csv', index=False)
    group_1000 = (heat_flux[np.isclose(heat_flux.pressure_hpa, 1000.0)]
                  .groupby(['ensemble', 'b'], as_index=False)
                  .agg(mean_vT_K_m_s_1=('value', 'mean'), min_vT=('value', 'min'), max_vT=('value', 'max')))
    lower = (heat_flux[heat_flux.pressure_hpa.between(850.0, 1000.0)]
             .groupby(['ensemble', 'b'], as_index=False)
             .agg(mean_profile_vT_850_1000_K_m_s_1=('value', 'mean')))
    checks = group_1000.merge(lower, on=['ensemble', 'b'], how='left')
    checks.to_csv(TABLES / 'Figure7_vT_group_mean_checks.csv', index=False)
    f5_png, f5_pdf, f5_ymax = f5
    f7_png, f7_pdf, heat_limits, f7_ymax = f7
    dimensions = {}
    for path in [f5_png, f7_png]:
        with Image.open(path) as image:
            dimensions[path.name] = list(image.size)
    standard_b2 = checks[checks.ensemble.eq('standard') & np.isclose(checks.b, 2.0)].iloc[0]
    u30_b2 = checks[checks.ensemble.eq('u30') & np.isclose(checks.b, 2.0)].iloc[0]
    summary_json = {
        'requested_latitude_domain_degN': [25.0, 90.0],
        'actual_grid_centers_degN': [float(summary.latitude_grid_min_deg.min()), float(summary.latitude_grid_max_deg.max())],
        'eke_cases': int(eke.case.nunique()), 'eke_rows': int(len(eke)),
        'eady_standard_cases': int(eady.case.nunique()), 'heat_flux_cases': int(heat_flux.case.nunique()),
        'figure5_eke_ymax_m2_s-2': float(f5_ymax), 'figure7_eke_ymax_m2_s-2': float(f7_ymax),
        'figure7_heat_flux_xlim_K_m_s-1': [float(heat_limits[0]), float(heat_limits[1])],
        'png_dimensions_pixels': dimensions,
        'rechecked_standard_b2_mean_vT_at_1000hPa_K_m_s-1': float(standard_b2.mean_vT_K_m_s_1),
        'rechecked_u30_b2_mean_vT_at_1000hPa_K_m_s-1': float(u30_b2.mean_vT_K_m_s_1),
    }
    (TABLES / 'recalculation_summary.json').write_text(json.dumps(summary_json, indent=2) + '\n', encoding='utf-8')
    caption5 = ('Figure 5. Eddy kinetic energy (EKE) evolution across experiments. The 25–90°N area- and '
                'mass-weighted EKE (m² s⁻²) is shown as a function of time for different jet configurations. '
                'Insets report the corresponding initial 25–90°N area- and mass-weighted mean Eady growth rates. '
                'The curves demonstrate that more poleward-shifted (larger s), broader (smaller n), and larger-b '
                'configurations tend to produce faster and larger EKE growth, whereas more equatorward-shifted '
                '(smaller s), narrower (larger n), and smaller-b configurations yield slower, weaker growth.')
    caption7 = ('Figure 7. Vertical structure and eddy activity in the constant-u0 and constant-Umax ensembles. '
                'Left and right columns show the constant-u0 and constant-Umax=30 m s-1 ensembles, respectively. '
                '(a, b) Initial zonal-mean zonal-wind profiles at the latitude of each simulation’s initial 300-hPa '
                'jet core. (c, d) Initial Eady growth-rate profiles at the same latitude; the upper axes show the '
                'corresponding zonal-wind profiles. (e, f) 25–90°N area- and mass-weighted EKE evolution during '
                'hours 144–360. (g, h) Eddy heat flux [v′T′], averaged over the case-specific rising phase in which '
                'the 25–90°N area- and mass-weighted EKE increases from 50% to 80% of its maximum. Thin lines '
                'represent individual simulations, thick lines show b-group means, colors denote b, and line styles '
                'denote n. Blue and orange shading mark the 1000–850-hPa and 500–300-hPa layers, respectively.')
    (OUTPUT / 'CAPTION_DRAFTS.md').write_text(
        '# Caption drafts\n\n## Figure 5\n\n' + caption5 + '\n\n## Figure 7\n\n' + caption7 + '\n', encoding='utf-8')
    qc = [
        '# QC report', '',
        '- EKE was independently recomputed from hourly u_plev and v_plev after removing the instantaneous zonal mean at every time, pressure, and latitude.',
        '- Horizontal averaging uses cosine-latitude weights over requested 25–90°N; the 1° grid contributes centers from 25.5°N through 89.5°N.',
        '- Vertical averaging uses the same pressure-trapezoid mass weights as the previous figures.',
        '- Figure 5 initial Eady rates were recomputed over the same 25–90°N area–mass-weighted domain.',
        '- Figure 7 panels (a–d) retain the existing jet-core-local profiles without recomputation.',
        '- Every Figure 7 case has a newly diagnosed first 50–80% rising-phase interval based on the 25–90°N EKE series.',
        '- Panels (g–h) use hourly jet-relative ±15° eddy heat-flux diagnostics averaged over those new windows.',
        f'- Rechecked b=2 mean 1000-hPa [v′T′]: constant-u0 = {standard_b2.mean_vT_K_m_s_1:.2f} K m s⁻¹; constant-Umax = {u30_b2.mean_vT_K_m_s_1:.2f} K m s⁻¹.',
        f'- Figure 7 EKE axis includes the complete data range with an upper limit of {f7_ymax:.0f} m² s⁻².',
        '- Both PNGs were written at 300 dpi and matching vector PDFs were generated.', ''
    ]
    (OUTPUT / 'QC_REPORT.md').write_text('\n'.join(qc), encoding='utf-8')
    readme = f'''# Figures 5 and 7: 25–90°N EKE update

This package changes the EKE domain from 0–90°N to 25–90°N. Figure 5 also recomputes its area–mass-weighted initial Eady annotations over 25–90°N. Figure 7 retains panels (a–d), rediagnoses every case-specific 50–80% EKE-growth window, and reaverages the hourly jet-relative ±15° eddy heat flux for panels (g–h).

## Reproduce

```bash
MPLCONFIGDIR=/data/keeling/a/mingfei5/local/tmp/matplotlib \\
  /data/keeling/a/mingfei5/anaconda3/envs/pygen_clean/bin/python \\
  {OUTPUT / 'scripts' / 'recompute_25_90_diagnostics.py'} --workers 2
MPLCONFIGDIR=/data/keeling/a/mingfei5/local/tmp/matplotlib \\
  /data/keeling/a/mingfei5/anaconda3/envs/pygen_clean/bin/python \\
  {OUTPUT / 'scripts' / 'replot_figures5_7_25_90.py'}
```

Use `--force` on the first command to rebuild all per-case checkpoints.

## Outputs

- `figures/`: revised Figure 5 and Figure 7 PNG/PDF files.
- `tables/`: exact EKE series, Eady annotations, new windows, reaveraged heat-flux profiles, and comparison checks.
- `checkpoints/`: resumable per-case EKE calculations.
- `CAPTION_DRAFTS.md`, `QC_REPORT.md`, and `SHA256SUMS`.
'''
    (OUTPUT / 'README.md').write_text(readme, encoding='utf-8')


def write_manifest():
    files = sorted(path for path in OUTPUT.rglob('*') if path.is_file() and path.name != 'SHA256SUMS'
                   and 'checkpoints' not in path.parts and '__pycache__' not in path.parts)
    lines = [f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUTPUT)}' for path in files]
    (OUTPUT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    source = load_source_module()
    eke = pd.read_csv(TABLES / 'eke_25_90N_area_mass_weighted_timeseries_all90.csv')
    summary = pd.read_csv(TABLES / 'eke_25_90N_windows_all90.csv')
    eady = pd.read_csv(TABLES / 'initial_eady_25_90N_area_mass_weighted_standard_all45.csv')
    heat_flux = pd.read_csv(TABLES / 'eddy_heat_flux_profiles_50_80pct_peak_eke_25_90N_all90.csv')
    initial = pd.read_csv(SOURCE_PACKAGE / 'data' / 'initial_profiles_at_jet_core.csv')
    f5 = make_figure5(eke, eady)
    f7 = make_figure7(source, initial, eke, heat_flux)
    make_supporting_outputs(eke, summary, eady, heat_flux, f5, f7)
    write_manifest()
    print(f5[0]); print(f5[1]); print(f7[0]); print(f7[1])


if __name__ == '__main__':
    main()
