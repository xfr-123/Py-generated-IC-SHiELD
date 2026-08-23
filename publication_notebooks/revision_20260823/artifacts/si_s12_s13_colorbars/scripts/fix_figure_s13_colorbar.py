#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/keeling/a/mingfei5/a/data/original')
OUTPUT = ROOT / 'paper_revision/figure_s13_colorbar_fix_20260820'
FIGURES = OUTPUT / 'figures'
SOURCE_SCRIPT = ROOT / 'eddy/initial_state_analysis_20260722/scripts/build_initial_state_report_analysis.py'


def load_source_module():
    spec = importlib.util.spec_from_file_location('figure_s13_source', SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {SOURCE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_source_module()


def make_figure() -> tuple[Path, Path]:
    case_rows = [
        [('BCwave_b2n1', 'n=1'), ('BCwave_b2n3', 'n=3'), ('BCwave_b2n6', 'n=6')],
        [('BCwave_b2n3s-10', 's=−10°'), ('BCwave_b2n3', 's=0°'), ('BCwave_b2n3s10', 's=+10°')],
    ]

    figure = plt.figure(figsize=(14.8, 8.6), facecolor='white')
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.115,
        right=0.91,
        top=0.90,
        bottom=0.12,
        hspace=0.20,
        wspace=0.09,
    )
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            share_axis = axes[0, 0] if row or column else None
            axes[row, column] = figure.add_subplot(grid[row, column], sharex=share_axis, sharey=share_axis)
    colorbar_axis = figure.add_subplot(grid[:, 3])

    image = None
    for row, cases in enumerate(case_rows):
        for column, (case, label) in enumerate(cases):
            fields = source.initial_fields(source.ARCHIVE_COARSE / f'{case}.nc')
            lat = fields['lat']
            pressure = fields['pressure']
            image = axes[row, column].contourf(
                lat,
                pressure,
                fields['u'],
                levels=np.arange(0, 34, 2),
                cmap='viridis',
                extend='max',
            )
            eady_contours = axes[row, column].contour(
                lat,
                pressure,
                fields['eady'],
                levels=[0.4, 0.8, 1.2, 1.6],
                colors='#ffcc33',
                linewidths=0.8,
            )
            axes[row, column].clabel(eady_contours, fmt='%.1f', fontsize=7)
            axes[row, column].contour(
                lat,
                pressure,
                fields['theta'],
                levels=np.arange(280, 371, 20),
                colors='white',
                linewidths=0.4,
            )
            axes[row, column].set_title(label, fontsize=11)
            axes[row, column].set_xlim(15, 75)
            axes[row, column].set_ylim(1000, 100)
            axes[row, column].set_yticks([1000, 850, 700, 500, 300, 200, 100])
            axes[row, column].grid(alpha=0.12)
            if column == 0:
                axes[row, column].set_ylabel('Pressure (hPa)')
            else:
                axes[row, column].tick_params(labelleft=False)
            if row == 1:
                axes[row, column].set_xlabel('Latitude (°N)')
            else:
                axes[row, column].tick_params(labelbottom=False)

    axes[0, 0].text(
        -0.25,
        0.5,
        'Width variation\n(b=2, s=0)',
        transform=axes[0, 0].transAxes,
        rotation=90,
        va='center',
        ha='center',
        fontweight='bold',
    )
    axes[1, 0].text(
        -0.25,
        0.5,
        'Latitude shift\n(b=2, n=3)',
        transform=axes[1, 0].transAxes,
        rotation=90,
        va='center',
        ha='center',
        fontweight='bold',
    )

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation='vertical')
    colorbar.set_label('Initial zonal-mean wind (m s$^{-1}$)', fontsize=10, labelpad=8)
    colorbar.ax.tick_params(labelsize=9, pad=2)
    figure.text(
        0.51,
        0.055,
        'Contours: yellow = Eady growth rate (day$^{-1}$); white = potential temperature',
        ha='center',
        va='center',
        fontsize=9.5,
    )
    figure.suptitle('Initial cross-section changes associated with n and s', fontsize=15, y=0.97)

    png_path = FIGURES / 'Figure_S13_initial_cross_sections_n_s_variations_colorbar_fixed.png'
    pdf_path = FIGURES / 'Figure_S13_initial_cross_sections_n_s_variations_colorbar_fixed.pdf'
    figure.savefig(png_path, dpi=300, facecolor='white')
    figure.savefig(pdf_path, facecolor='white')
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = make_figure()
    print(png_path)
    print(pdf_path)


if __name__ == '__main__':
    main()
