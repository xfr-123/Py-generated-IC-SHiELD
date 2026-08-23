#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path('/data/keeling/a/mingfei5/a/data/original')
BASE = ROOT / 'eddy' / 'controlled_umax30_bns_analysis_20260722'
OUT = BASE / 'original_style_comparison'
FIGURES = OUT / 'figures'
TABLES = OUT / 'tables'
ARCHIVE_HEADLINE = ROOT / 'priority_revision_analysis_20260720' / 'tables' / 'headline_responses_and_initial_predictors.csv'
CONTROLLED_HEADLINE = BASE / 'tables' / 'umax30_all45_headline_responses.csv'
CONTROLLED_RWB = TABLES / 'controlled_rwb_direct_counts_all45.csv'
ARCHIVE_PERSISTENCE = TABLES / 'archive_anticyclone_persistence_coarse5_all45.csv'
FIGURE12_ARCHIVE_PERSISTENCE = TABLES / 'Figure7_archive_exact_original_contrack_no_area.csv'

COLOR_CYC = '#fc9274'
COLOR_ANT = '#9ecae1'
B_VALUES = [1.0, 1.5, 2.0]
N_VALUES = [6, 3, 1]
S_VALUES = [-10, -5, 0, 5, 10]
RED_WHITE_CMAP = LinearSegmentedColormap.from_list('white_red', ['#ffffff', '#ff0000'])


def original_rwb_counts() -> pd.DataFrame:
    rows = []
    values = {
        'n': {6: (388, 501), 3: (446, 839), 1: (269, 1678)},
        'b': {1.0: (622, 867), 1.5: (311, 979), 2.0: (170, 1172)},
        's': {-10: (355, 53), -5: (358, 277), 0: (270, 611), 5: (120, 994), 10: (0, 1083)},
    }
    for parameter, groups in values.items():
        for value, (cyclonic, anticyclonic) in groups.items():
            rows.append({'ensemble': 'Original archive', 'parameter': parameter, 'value': value, 'cyclonic': cyclonic, 'anticyclonic': anticyclonic, 'total': cyclonic + anticyclonic})
    return pd.DataFrame(rows)


def controlled_rwb_counts() -> pd.DataFrame:
    data = pd.read_csv(CONTROLLED_RWB)
    rows = []
    for parameter, order in [('n', [6, 3, 1]), ('b', B_VALUES), ('s', S_VALUES)]:
        grouped = data.groupby(parameter)[['cyclonic', 'anticyclonic']].sum().reindex(order)
        for value, row in grouped.iterrows():
            rows.append({'ensemble': 'Fixed Umax=30 m s-1', 'parameter': parameter, 'value': value, 'cyclonic': int(row.cyclonic), 'anticyclonic': int(row.anticyclonic), 'total': int(row.sum())})
    return pd.DataFrame(rows)


def rwb_layout(data: pd.DataFrame):
    order = [('n', [6, 3, 1]), ('b', B_VALUES), ('s', S_VALUES)]
    y = []
    labels = []
    cyclonic = []
    anticyclonic = []
    group_spans = []
    position = 0.0
    for parameter, values in order:
        start = len(y)
        for value in values:
            row = data[(data.parameter == parameter) & np.isclose(data.value.astype(float), float(value))].iloc[0]
            total = row.cyclonic + row.anticyclonic
            y.append(position)
            labels.append(rf'${parameter}$ = {value:g}')
            cyclonic.append(row.cyclonic / total if total else 0)
            anticyclonic.append(row.anticyclonic / total if total else 0)
            position += 1.0
        group_spans.append((parameter, start, len(y)))
        position += 0.7
    return np.array(y), labels, np.array(cyclonic), np.array(anticyclonic), group_spans


def draw_rwb_panel(axis, data: pd.DataFrame, title: str, show_y: bool = True, show_arrows: bool = True):
    y, labels, cyclonic, anticyclonic, spans = rwb_layout(data)
    counts = []
    for parameter, values in [('n', [6, 3, 1]), ('b', B_VALUES), ('s', S_VALUES)]:
        for value in values:
            row = data[(data.parameter == parameter) & np.isclose(data.value.astype(float), float(value))].iloc[0]
            counts.append((int(row.cyclonic), int(row.anticyclonic)))
    axis.barh(y, cyclonic, height=0.7, color=COLOR_CYC, edgecolor='white')
    axis.barh(y, anticyclonic, height=0.7, left=cyclonic, color=COLOR_ANT, edgecolor='white')
    for position, left, right, (left_count, right_count) in zip(y, cyclonic, anticyclonic, counts):
        if left_count > 0:
            axis.text(left / 2, position, str(left_count), ha='center', va='center', fontsize=10)
        if right_count > 0:
            axis.text(left + right / 2, position, str(right_count), ha='center', va='center', fontsize=10)
    for (_, _, end), (_, next_start, _) in zip(spans[:-1], spans[1:]):
        axis.axhline(0.5 * (y[end - 1] + y[next_start]), color='black', linewidth=1.2)
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.grid(axis='x', linestyle=':', alpha=0.45)
    axis.set_yticks(y)
    if show_y:
        axis.set_yticklabels(labels)
    else:
        axis.tick_params(axis='y', labelleft=False)
    axis.set_xlabel('Fraction of Events')
    axis.set_title(title, fontsize=14)
    if show_arrows:
        directions = [('broader jet →', spans[0]), ('higher jet →', spans[1]), ('poleward-shifted jet →', spans[2])]
        for text, (_, start, end) in directions:
            center = 0.5 * (y[start] + y[end - 1])
            axis.text(1.015, center, text, rotation=-90, va='center', ha='left', transform=axis.get_yaxis_transform(), fontsize=10, fontweight='bold')


def make_rwb_figures(original: pd.DataFrame, controlled: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(8.2, 10.2), dpi=220)
    draw_rwb_panel(axis, controlled, 'Fixed-$U_{max}$ RWB Frequency\n($\\theta=330$ K, $|PV|=2.0$ PVU)')
    axis.invert_yaxis()
    figure.legend([Patch(color=COLOR_CYC), Patch(color=COLOR_ANT)], ['Cyclonic', 'Anticyclonic'], loc='lower center', ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.10, right=0.88)
    figure.savefig(FIGURES / 'Figure5_controlled_original_style.png', dpi=240, bbox_inches='tight')
    figure.savefig(FIGURES / 'Figure5_controlled_original_style.pdf', bbox_inches='tight')
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15.2, 10.2), sharey=True, dpi=220)
    draw_rwb_panel(axes[0], original, r'(a) constant $u_0$', show_y=True, show_arrows=False)
    draw_rwb_panel(axes[1], controlled, r'(b) constant $U_{\max}=30$ m s$^{-1}$', show_y=False, show_arrows=True)
    axes[0].invert_yaxis()
    figure.suptitle('Relative Frequency of Cyclonic vs Anticyclonic Wave Breaking\n($\\theta=330$ K, $|PV|=2.0$ PVU)', fontsize=19)
    figure.legend([Patch(color=COLOR_CYC), Patch(color=COLOR_ANT)], ['Cyclonic', 'Anticyclonic'], loc='lower center', ncol=2, frameon=False, fontsize=12)
    figure.subplots_adjust(bottom=0.10, right=0.93, top=0.88, wspace=0.06)
    figure.savefig(FIGURES / 'Figure5_original_vs_controlled_same_style.png', dpi=240, bbox_inches='tight')
    figure.savefig(FIGURES / 'Figure5_original_vs_controlled_same_style.pdf', bbox_inches='tight')
    plt.close(figure)


def heatmap_arrays(data: pd.DataFrame, value: str):
    return [data[np.isclose(data.s, s)].pivot(index='b', columns='n', values=value).reindex(index=B_VALUES, columns=N_VALUES).to_numpy(float) for s in S_VALUES]


def annotate_heatmap(axis, array, vmin, vmax, integer=True, fontsize=9, high_value_white=False):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = array[i, j]
            fraction = 0.5 if vmax == vmin else (value - vmin) / (vmax - vmin)
            use_white = fraction > 0.68 if high_value_white else fraction < 0.32
            axis.text(j, i, f'{value:.0f}' if integer else f'{value:.1f}', ha='center', va='center', fontsize=fontsize, color='white' if use_white else 'black')


def make_single_heatmap_figure(data, value, title, colorbar_label, stem, integer=True):
    arrays = heatmap_arrays(data, value)
    finite = np.concatenate([array.ravel() for array in arrays])
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    figure, axes = plt.subplots(1, 5, figsize=(15.7, 3.6), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for axis, s, array in zip(axes, S_VALUES, arrays):
        image = axis.imshow(array, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        annotate_heatmap(axis, array, vmin, vmax, integer)
        axis.set_title(f'$s={s}°$')
        axis.set_xticks(range(3), N_VALUES)
        axis.set_yticks(range(3), [f'{b:g}' for b in B_VALUES])
        axis.set_xlabel('$n$')
    axes[0].set_ylabel('$b$')
    figure.colorbar(image, ax=axes, label=colorbar_label, shrink=0.84)
    figure.suptitle(title, fontsize=14)
    figure.savefig(FIGURES / f'{stem}.png', dpi=240, bbox_inches='tight')
    figure.savefig(FIGURES / f'{stem}.pdf', bbox_inches='tight')
    plt.close(figure)


def make_comparison_heatmap(
    original,
    controlled,
    original_value,
    controlled_value,
    title,
    colorbar_label,
    stem,
    integer=True,
    color_limits=None,
):
    original_arrays = heatmap_arrays(original, original_value)
    controlled_arrays = heatmap_arrays(controlled, controlled_value)
    finite = np.concatenate([array.ravel() for array in original_arrays + controlled_arrays])
    if color_limits is None:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = color_limits
    figure = plt.figure(figsize=(13.4, 5.55), constrained_layout=True, facecolor='white')
    grid = figure.add_gridspec(
        3,
        6,
        width_ratios=[0.27, 1, 1, 1, 1, 1],
        height_ratios=[1, 1, 0.11],
        wspace=0.06,
        hspace=0.08,
    )
    axes = np.empty((2, 5), dtype=object)
    row_label_axes = [figure.add_subplot(grid[row, 0]) for row in range(2)]
    for row_label_axis in row_label_axes:
        row_label_axis.axis('off')
    for row in range(2):
        for column in range(5):
            axes[row, column] = figure.add_subplot(
                grid[row, column + 1],
                sharex=axes[0, 0] if row or column else None,
                sharey=axes[0, 0] if row or column else None,
            )
    colorbar_axis = figure.add_subplot(grid[2, 1:])
    image = None
    row_specs = [
        ('Standard', '#555555', '#f2f2f2', original_arrays),
        (r'Fixed $U_{max}=30$ m s$^{-1}$', '#C63B2D', '#fcecea', controlled_arrays),
    ]
    for row, (label, label_color, label_background, arrays) in enumerate(row_specs):
        row_label_axes[row].set_facecolor(label_background)
        row_label_axes[row].axvline(0.98, color=label_color, linewidth=3.0)
        row_label_axes[row].text(
            0.48,
            0.5,
            label,
            rotation=90,
            ha='center',
            va='center',
            fontsize=13.5,
            color=label_color,
            weight='bold',
        )
        for column, (s, array) in enumerate(zip(S_VALUES, arrays)):
            axis = axes[row, column]
            image = axis.imshow(array, origin='lower', cmap=RED_WHITE_CMAP, vmin=vmin, vmax=vmax, aspect='equal')
            axis.set_box_aspect(1)
            annotate_heatmap(axis, array, vmin, vmax, integer, fontsize=12.5, high_value_white=True)
            if row == 0:
                axis.set_title(f'$s={s}°$', fontsize=12.5, pad=5)
            axis.set_xticks(range(3), N_VALUES)
            axis.set_yticks(range(3), [f'{b:g}' for b in B_VALUES])
            axis.tick_params(axis='both', labelsize=11, length=3.5, width=0.9)
            if row == 1:
                axis.set_xlabel('$n$', fontsize=12.5, labelpad=2)
            else:
                axis.tick_params(axis='x', labelbottom=False)
            if column == 0:
                axis.set_ylabel('$b$', fontsize=12.5, labelpad=5)
            else:
                axis.tick_params(axis='y', labelleft=False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.9)
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label(colorbar_label, fontsize=12, labelpad=4)
    colorbar.ax.tick_params(labelsize=10.5, length=3)
    figure.suptitle(title, fontsize=14.5, weight='semibold')
    figure.savefig(FIGURES / f'{stem}.png', dpi=320, facecolor='white', bbox_inches='tight', pad_inches=0.04)
    figure.savefig(FIGURES / f'{stem}.pdf', facecolor='white', bbox_inches='tight', pad_inches=0.04)
    plt.close(figure)


def make_requested_common_heatmaps() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    archive = pd.read_csv(ARCHIVE_HEADLINE)
    controlled = pd.read_csv(CONTROLLED_HEADLINE)
    archive_persistence = pd.read_csv(FIGURE12_ARCHIVE_PERSISTENCE)
    make_comparison_heatmap(
        archive,
        controlled,
        'eddy_minus10_coalescence_count',
        'eddy_minus10_coalescence_count',
        'Eddy-Pressure Coalescence Before and After Jet-Strength Control',
        'Closed $-10$-hPa eddy-contour events',
        'figure10_standard_common_coalescence',
    )
    make_comparison_heatmap(
        archive_persistence,
        controlled,
        'figure7_persistence_hours',
        'anticyclone_overlap_duration_hours',
        'Anticyclone Persistence Before and After Jet-Strength Control',
        'Maximum overlap-connected duration (h)',
        'figure12_standard_common_persistence',
        color_limits=(0.0, 200.0),
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f'{value:.3f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def write_report(original_rwb, controlled_rwb, archive, controlled, archive_persistence):
    combined_rwb = pd.concat([original_rwb, controlled_rwb], ignore_index=True)
    combined_rwb.to_csv(TABLES / 'Figure5_original_controlled_group_counts.csv', index=False)
    archive_coalescence = archive[['case', 'b', 'n', 's', 'eddy_minus10_coalescence_count']].copy()
    controlled_coalescence = controlled[['case', 'b', 'n', 's', 'eddy_minus10_coalescence_count']].copy()
    archive_coalescence.to_csv(TABLES / 'Figure10_archive_eddy_minus10_counts.csv', index=False)
    controlled_coalescence.to_csv(TABLES / 'Figure10_controlled_eddy_minus10_counts.csv', index=False)
    controlled_persistence = controlled[['case', 'b', 'n', 's', 'anticyclone_overlap_duration_hours']].copy()
    controlled_persistence.to_csv(TABLES / 'Figure13_controlled_persistence_coarse5.csv', index=False)

    rwb_summary = combined_rwb.copy()
    rwb_summary['cyclonic_fraction'] = rwb_summary.cyclonic / rwb_summary.total
    rwb_summary['anticyclonic_fraction'] = rwb_summary.anticyclonic / rwb_summary.total
    report = [
        '# Original-Style Figure Comparison for the Complete Fixed-$U_{max}$ Ensemble', '',
        'This package reproduces the visual forms of manuscript Figures 5, 10, and 13 using the complete 45-member fixed-$U_{max}=30$ m s$^{-1}$ ensemble and places them beside consistently calculated archive results.', '',
        '## Figure 5: RWB orientation fractions', '',
        'The controlled RWB counts are recomputed directly from the hourly 330-K PV fields with the same WaveBreaking settings used by the original notebook: five smoothing passes, $|PV|=2$ PVU, `range_group=5`, and `min_exp=1`. They are therefore not inferred from the linked-track subset.', '',
        '![Figure 5 comparison](figures/Figure5_original_vs_controlled_same_style.png)', '',
        '## Figure 10: eddy-pressure coalescence', '',
        'The Figure-10 layout is retained, but both rows use the revised closed $-10$-hPa eddy-pressure contour criterion. This makes the archive-versus-controlled comparison robust to the zonal-mean-pressure objection; it should not be labeled as the original full-field 980/990-hPa criterion.', '',
        '![Figure 10 comparison](figures/Figure10_original_vs_controlled_eddy_minus10_same_style.png)', '',
        '## Figure 13: anticyclone persistence', '',
        'Both rows use the same hourly 5° tracker, $p_s\\ge1010$ hPa, minimum area $10^5$ km$^2$, and consecutive-overlap threshold 0.8. The archive persistence was recomputed specifically for this comparison.', '',
        '![Figure 13 comparison](figures/Figure13_original_vs_controlled_persistence_same_style.png)', '',
        '## RWB grouped counts', '', markdown_table(rwb_summary[['ensemble', 'parameter', 'value', 'cyclonic', 'anticyclonic', 'total', 'cyclonic_fraction', 'anticyclonic_fraction']]), '',
        '## Interpretation', '',
        '- The original-style RWB plot directly shows whether strength control changes both the total number of overturning detections and their cyclonic–anticyclonic partition.',
        '- The Figure-10-style comparison shows the reversal of the $b$ relationship for revised eddy-pressure coalescence while retaining the strong $n$ and $s$ structure.',
        '- The Figure-13-style comparison isolates persistence changes from tracker-resolution differences because both ensembles use the same coarse overlap definition.', '',
    ]
    (OUT / 'original_style_figure_comparison_report.md').write_text('\n'.join(report), encoding='utf-8')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    archive = pd.read_csv(ARCHIVE_HEADLINE)
    controlled = pd.read_csv(CONTROLLED_HEADLINE)
    archive_persistence = pd.read_csv(ARCHIVE_PERSISTENCE)
    original_rwb = original_rwb_counts()
    controlled_rwb = controlled_rwb_counts()
    make_rwb_figures(original_rwb, controlled_rwb)
    make_single_heatmap_figure(controlled, 'eddy_minus10_coalescence_count', 'Fixed-$U_{max}$ Eddy-Pressure Coalescence', 'Closed $-10$-hPa eddy-contour events', 'Figure10_controlled_original_style_eddy_minus10')
    make_comparison_heatmap(archive, controlled, 'eddy_minus10_coalescence_count', 'eddy_minus10_coalescence_count', 'Eddy-Pressure Coalescence Before and After Jet-Strength Control', 'Closed $-10$-hPa eddy-contour events', 'Figure10_original_vs_controlled_eddy_minus10_same_style')
    make_single_heatmap_figure(controlled, 'anticyclone_overlap_duration_hours', 'Fixed-$U_{max}$ Anticyclone Persistence', 'Overlap-connected duration (h)', 'Figure13_controlled_original_style_persistence')
    make_comparison_heatmap(archive_persistence, controlled, 'anticyclone_overlap_duration_hours_coarse5', 'anticyclone_overlap_duration_hours', 'Anticyclone Persistence Before and After Jet-Strength Control', 'Overlap-connected duration (h)', 'Figure13_original_vs_controlled_persistence_same_style')
    write_report(original_rwb, controlled_rwb, archive, controlled, archive_persistence)
    print(f'Wrote original-style comparison package to {OUT}')


if __name__ == '__main__':
    main()
