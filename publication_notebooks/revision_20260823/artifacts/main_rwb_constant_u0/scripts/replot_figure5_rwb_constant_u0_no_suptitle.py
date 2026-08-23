#!/usr/bin/env python3
from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'eddy' / 'controlled_umax30_bns_analysis_20260722' / 'original_style_comparison' / 'scripts' / 'make_original_style_comparison_figures.py'
OUTPUT = ROOT / 'publication_notebooks' / 'outputs'

spec = importlib.util.spec_from_file_location('rwb_counts_source', SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

COLOR_CYC = '#fc9274'
COLOR_ANT = '#9ecae1'


def main() -> None:
    data = source.original_rwb_counts()
    y, labels, cyclonic, anticyclonic, spans = source.rwb_layout(data)
    counts = []
    for parameter, values in [('n', [6, 3, 1]), ('b', [1.0, 1.5, 2.0]), ('s', [-10, -5, 0, 5, 10])]:
        for value in values:
            row = data[(data.parameter == parameter) & np.isclose(data.value.astype(float), float(value))].iloc[0]
            counts.append((int(row.cyclonic), int(row.anticyclonic)))

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'axes.labelsize': 18,
        'xtick.labelsize': 16.5,
        'ytick.labelsize': 16.5,
        'legend.fontsize': 16.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    figure, axis = plt.subplots(figsize=(8, 10), facecolor='white')
    axis.set_facecolor('white')
    axis.barh(y, cyclonic, height=0.7, color=COLOR_CYC, edgecolor='white')
    axis.barh(y, anticyclonic, height=0.7, left=cyclonic, color=COLOR_ANT, edgecolor='white')

    for position, left, right, (left_count, right_count) in zip(y, cyclonic, anticyclonic, counts):
        if left_count > 0:
            axis.text(left / 2, position, str(left_count), ha='center', va='center', fontsize=15)
        if right_count > 0:
            axis.text(left + right / 2, position, str(right_count), ha='center', va='center', fontsize=15)

    for (_, _, end), (_, next_start, _) in zip(spans[:-1], spans[1:]):
        axis.axhline(0.5 * (y[end - 1] + y[next_start]), color='black', linewidth=1.5)

    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlabel('Fraction of Events')
    axis.grid(axis='x', linestyle=':', alpha=0.6)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.invert_yaxis()

    directions = [('broader jet →', spans[0]), ('larger b →', spans[1]), ('poleward-shifted jet →', spans[2])]
    for text, (_, start, end) in directions:
        center = 0.5 * (y[start] + y[end - 1])
        axis.text(1.03, center, text, rotation=-90, va='center', ha='center', transform=axis.get_yaxis_transform(), fontsize=16.5, fontweight='bold')

    handles = [Patch(color=COLOR_CYC, label='Cyclonic'), Patch(color=COLOR_ANT, label='Anticyclonic')]
    axis.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    figure.tight_layout(rect=[0, 0.08, 1, 0.995])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = [
        OUTPUT / 'Figure5_RWB_object_time_statistics.png',
        OUTPUT / 'Figure5_RWB_object_time_statistics.pdf',
        OUTPUT / 'Figure6_RWB_fraction_constant_u0_no_suptitle.png',
        OUTPUT / 'Figure6_RWB_fraction_constant_u0_no_suptitle.pdf',
    ]
    for target in targets:
        if target.suffix == '.png':
            figure.savefig(target, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            figure.savefig(target, bbox_inches='tight', facecolor='white')
        print(target)
    plt.close(figure)


if __name__ == '__main__':
    main()
