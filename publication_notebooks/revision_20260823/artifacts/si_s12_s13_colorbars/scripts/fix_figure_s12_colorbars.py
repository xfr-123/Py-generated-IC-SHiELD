#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/data/keeling/a/mingfei5/a/data/original')
OUTPUT = ROOT / 'paper_revision/figure_s12_colorbar_fix_20260820'
FIGURES = OUTPUT / 'figures'
SOURCE_SCRIPT = ROOT / 'eddy/initial_state_analysis_20260722/scripts/build_initial_state_report_analysis.py'
ARCHIVE_TABLE = ROOT / 'priority_revision_analysis_20260720/tables/headline_responses_and_initial_predictors.csv'


def load_source_module():
    spec = importlib.util.spec_from_file_location('figure_s12_source', SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {SOURCE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_source_module()


def plot_figure_s12(archive: pd.DataFrame) -> tuple[Path, Path]:
    cases = [
        ('BCwave_b1n3', 1.0),
        ('BCwave_b15n3', 1.5),
        ('BCwave_b2n3', 2.0),
    ]
    bundles = []
    for case, b_value in cases:
        fields = source.initial_fields(source.ARCHIVE_COARSE / f'{case}.nc')
        row = archive.loc[archive['case'].eq(case)].iloc[0]
        fields['bt_conversion'] = source.growth_phase_conversion(
            source.ARCHIVE_COARSE / f'{case}.nc',
            int(row.growth_onset_index),
            int(row.peak_eke_time_index),
        )
        fields['case'] = case
        fields['b'] = b_value
        bundles.append(fields)

    latitude_mask = (bundles[0]['lat'] >= 15.0) & (bundles[0]['lat'] <= 75.0)
    pressure_mask = (bundles[0]['pressure'] >= 100.0) & (bundles[0]['pressure'] <= 1000.0)
    eady_max = max(
        np.nanpercentile(bundle['eady'][np.ix_(pressure_mask, latitude_mask)], 99)
        for bundle in bundles
    )
    rossby_max = max(
        np.nanpercentile(bundle['rossby_depth'][np.ix_(pressure_mask, latitude_mask)], 97)
        for bundle in bundles
    )
    shear_max = max(
        np.nanpercentile(np.abs(bundle['dudy'][np.ix_(pressure_mask, latitude_mask)]), 99)
        for bundle in bundles
    )
    conversion_max = max(
        np.nanpercentile(np.abs(bundle['bt_conversion'][np.ix_(pressure_mask, latitude_mask)]), 99)
        for bundle in bundles
    )

    figure = plt.figure(figsize=(16.3, 17.0), facecolor='white')
    grid = figure.add_gridspec(
        5,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.115,
        right=0.925,
        top=0.94,
        bottom=0.055,
        hspace=0.18,
        wspace=0.10,
    )
    axes = np.empty((5, 3), dtype=object)
    for row in range(5):
        for column in range(3):
            share_axis = axes[0, 0] if row or column else None
            axes[row, column] = figure.add_subplot(grid[row, column], sharex=share_axis, sharey=share_axis)
    colorbar_axes = [figure.add_subplot(grid[row, 3]) for row in range(5)]
    row_mappables = [None] * 5

    for column, bundle in enumerate(bundles):
        lat = bundle['lat']
        pressure = bundle['pressure']
        eta_c = source.eta_core(bundle['b'])
        core_pressure = 1000.0 * eta_c

        image = axes[0, column].contourf(
            lat,
            pressure,
            bundle['u'],
            levels=np.arange(0, 34, 2),
            cmap='viridis',
            extend='max',
        )
        axes[0, column].contour(
            lat,
            pressure,
            bundle['theta'],
            levels=np.arange(270, 381, 10),
            colors='white',
            linewidths=0.45,
        )
        axes[0, column].axhline(core_pressure, color='#ffcc33', linewidth=2.0, linestyle='--')
        axes[0, column].plot(
            bundle['jet_lat'],
            core_pressure,
            marker='*',
            color='#ffcc33',
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.4,
        )
        axes[0, column].set_title(
            f"b={bundle['b']:g}: $U_{{max}}$={np.nanmax(bundle['u']):.1f} m s$^{{-1}}$\n"
            f"$\\eta_c$={eta_c:.3f}, $p_c$≈{core_pressure:.0f} hPa"
        )
        if column == 0:
            row_mappables[0] = image

        image = axes[1, column].contourf(
            lat,
            pressure,
            bundle['eady'],
            levels=np.linspace(0, max(0.4, eady_max), 13),
            cmap='magma',
            extend='max',
        )
        axes[1, column].contour(
            lat,
            pressure,
            bundle['u'],
            levels=[5, 10, 15, 20, 25],
            colors='white',
            linewidths=0.45,
        )
        if column == 0:
            row_mappables[1] = image

        image = axes[2, column].contourf(
            lat,
            pressure,
            bundle['rossby_depth'],
            levels=np.linspace(0, max(5.0, rossby_max), 13),
            cmap='cividis',
            extend='max',
        )
        axes[2, column].contour(
            lat,
            pressure,
            bundle['u'],
            levels=[5, 10, 15, 20, 25],
            colors='white',
            linewidths=0.45,
        )
        axes[2, column].text(
            0.03,
            0.06,
            f"L={bundle['jet_half_width_km']:.0f} km\nH_R={bundle['rossby_penetration_theory_km']:.1f} km",
            transform=axes[2, column].transAxes,
            fontsize=8,
            bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
        )
        if column == 0:
            row_mappables[2] = image

        shear_scaled = bundle['dudy'] * 1.0e5
        image = axes[3, column].contourf(
            lat,
            pressure,
            shear_scaled,
            levels=np.linspace(-shear_max * 1.0e5, shear_max * 1.0e5, 17),
            cmap='RdBu_r',
            extend='both',
        )
        axes[3, column].contour(
            lat,
            pressure,
            bundle['u'],
            levels=[5, 10, 15, 20, 25],
            colors='0.25',
            linewidths=0.4,
        )
        if column == 0:
            row_mappables[3] = image

        conversion_scaled = bundle['bt_conversion'] * 1.0e5
        image = axes[4, column].contourf(
            lat,
            pressure,
            conversion_scaled,
            levels=np.linspace(-conversion_max * 1.0e5, conversion_max * 1.0e5, 17),
            cmap='PuOr_r',
            extend='both',
        )
        axes[4, column].contour(
            lat,
            pressure,
            bundle['u'],
            levels=[5, 10, 15, 20, 25],
            colors='0.25',
            linewidths=0.4,
        )
        if column == 0:
            row_mappables[4] = image

    row_labels = [
        'Initial zonal wind\n(m s$^{-1}$)',
        'Initial Eady growth\n(day$^{-1}$)',
        'Local Rossby depth\n$fL/N$ (km)',
        'Initial $\\partial\\bar u/\\partial y$\n($10^{-5}$ s$^{-1}$)',
        'Growth-phase BT conversion\n($10^{-5}$ m$^2$ s$^{-3}$)',
    ]
    for row in range(5):
        for column in range(3):
            axis = axes[row, column]
            axis.set_xlim(15, 75)
            axis.set_ylim(1000, 100)
            axis.set_yticks([1000, 850, 700, 500, 300, 200, 100])
            axis.grid(alpha=0.12)
            if column == 0:
                axis.set_ylabel('Pressure (hPa)')
            else:
                axis.tick_params(labelleft=False)
            if row == 4:
                axis.set_xlabel('Latitude (°N)')
            else:
                axis.tick_params(labelbottom=False)
        axes[row, 0].text(
            -0.29,
            0.5,
            row_labels[row],
            transform=axes[row, 0].transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
        )

    colorbar_labels = [
        'Zonal wind (m s$^{-1}$)',
        'Eady growth rate (day$^{-1}$)',
        'Rossby penetration depth (km)',
        '$\\partial\\bar u/\\partial y$ ($10^{-5}$ s$^{-1}$)',
        "$-[u'v']\\,\\partial\\bar u/\\partial y$ ($10^{-5}$ m$^2$ s$^{-3}$)",
    ]
    for colorbar_axis, mappable, label in zip(colorbar_axes, row_mappables, colorbar_labels):
        colorbar = figure.colorbar(mappable, cax=colorbar_axis, orientation='vertical')
        colorbar.set_label(label, fontsize=9, labelpad=7)
        colorbar.ax.tick_params(labelsize=8, pad=2)

    figure.suptitle(
        'Initial-state and growth-phase cross sections for the b variation (n=3, s=0)',
        fontsize=15,
        y=0.985,
    )
    png_path = FIGURES / 'Figure_S12_initial_cross_sections_b_variation_colorbar_fixed.png'
    pdf_path = FIGURES / 'Figure_S12_initial_cross_sections_b_variation_colorbar_fixed.pdf'
    figure.savefig(png_path, dpi=300, facecolor='white')
    figure.savefig(pdf_path, facecolor='white')
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    archive = pd.read_csv(ARCHIVE_TABLE)
    png_path, pdf_path = plot_figure_s12(archive)
    print(png_path)
    print(pdf_path)


if __name__ == '__main__':
    main()
