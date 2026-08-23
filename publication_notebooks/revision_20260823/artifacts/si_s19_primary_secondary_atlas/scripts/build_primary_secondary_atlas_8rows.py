#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/data/keeling/a/mingfei5/a/data/original')
OUTPUT_BASE = ROOT / 'paper_revision/primary_secondary_atlas_8rows_20260820'
SOURCE_SCRIPT = ROOT / 'eddy/expanded_A1_cyclone_interaction_atlas_20260723/scripts/build_expanded_A1_atlas.py'
MANIFEST = ROOT / 'eddy/primary_secondary_first_binary_coalescence_A1_20260725/tables/Figure_A1_primary_secondary_first_binary_manifest.csv'
ROWS_PER_PAGE = 8
PAGE_SIZE_INCHES = (11.69, 16.54)


def load_source_module():
    spec = importlib.util.spec_from_file_location('expanded_a1_source_8rows', SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {SOURCE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


atlas = load_source_module()
atlas.ROWS_PER_PAGE = ROWS_PER_PAGE


def load_event_catalog() -> pd.DataFrame:
    events = pd.read_csv(MANIFEST)
    if not np.all(events.n_source_ids == 2):
        raise RuntimeError('The selected manifest contains a non-binary event')
    if events.case.duplicated().any():
        raise RuntimeError('The selected manifest contains more than one event per case')
    events['available_minus24'] = events.time_index >= 24
    events['available_plus24'] = events.time_index + 24 <= 359
    events['available_plus48'] = events.time_index + 48 <= 359
    events = events.sort_values(['s', 'b', 'n', 'case']).reset_index(drop=True)
    events['atlas_event_id'] = np.arange(1, len(events) + 1)
    events['page'] = (events.index // ROWS_PER_PAGE) + 1
    events['row_on_page'] = (events.index % ROWS_PER_PAGE) + 1
    events['hemisphere'] = np.where(events.child_centroid_lat >= 0, 'NH', 'SH')
    return events


def precursor_labels(row: pd.Series):
    center_longitude = float(row.child_centroid_lon)
    primary_delta = float(atlas.longitude_delta(np.array([row.primary_marker_lon]), center_longitude)[0])
    secondary_delta = float(atlas.longitude_delta(np.array([row.secondary_marker_lon]), center_longitude)[0])
    return (
        ('P', float(row.primary_marker_lat), center_longitude + primary_delta),
        ('S', float(row.secondary_marker_lat), center_longitude + secondary_delta),
    )


atlas.precursor_labels = precursor_labels


def make_page(page_events: pd.DataFrame, page_number: int, total_pages: int, case_cache):
    displayed_rows = len(page_events)
    top_margin_inches = 0.75
    bottom_margin_inches = 1.65
    row_height_inches = (PAGE_SIZE_INCHES[1] - top_margin_inches - bottom_margin_inches) / ROWS_PER_PAGE
    page_height = top_margin_inches + bottom_margin_inches + row_height_inches * displayed_rows
    figure, axes = plt.subplots(
        displayed_rows,
        len(atlas.OFFSETS),
        figsize=(PAGE_SIZE_INCHES[0], page_height),
        sharex=False,
        sharey=False,
        facecolor='white',
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=1.0 - top_margin_inches / page_height,
        bottom=bottom_margin_inches / page_height,
        hspace=0.30,
        wspace=0.10,
    )
    filled = None
    for column, offset in enumerate(atlas.OFFSETS):
        axes[0, column].set_title(f'{offset:+d} h', fontsize=10)
    for row_index in range(displayed_rows):
        event = page_events.iloc[row_index]
        fields = case_cache[event['case']]
        for column, offset in enumerate(atlas.OFFSETS):
            result = atlas.panel(axes[row_index, column], event, offset, fields, column)
            if result is not None:
                filled = result
            if row_index == displayed_rows - 1:
                axes[row_index, column].set_xlabel('Longitude (°E)', fontsize=7.5, labelpad=2.5)
    if filled is not None:
        colorbar_axis = figure.add_axes([0.24, 0.50 / page_height, 0.52, 0.15 / page_height])
        colorbar = figure.colorbar(filled, cax=colorbar_axis, orientation='horizontal')
        colorbar.set_label(
            r'Hemisphere-signed smoothed 850-hPa cyclonic vorticity ($10^{-5}$ s$^{-1}$)',
            fontsize=8,
            labelpad=3,
        )
        colorbar.ax.tick_params(labelsize=7, pad=2)
    figure.suptitle(
        f'Figure A1 (continued): first primary–secondary 990-hPa binary coalescence in each case — page {page_number}/{total_pages}',
        fontsize=11.5,
        y=1.0 - 0.30 / page_height,
    )
    return figure

def build_atlas(events: pd.DataFrame, limit_pages: int | None = None):
    pages_dir = OUTPUT_BASE / 'pages'
    figures_dir = OUTPUT_BASE / 'figures'
    tables_dir = OUTPUT_BASE / 'tables'
    pages_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(tables_dir / 'Figure_A1_primary_secondary_atlas_8rows_crosswalk.csv', index=False)
    total_pages = math.ceil(len(events) / ROWS_PER_PAGE)
    rendered_pages = min(total_pages, limit_pages) if limit_pages is not None else total_pages

    needed_cases = events[events.page <= rendered_pages].groupby('case')
    case_cache = {}
    for case, case_events in needed_cases:
        print(f'Loading {case}: {len(case_events)} event', flush=True)
        case_cache[case] = atlas.load_case_fields(case, atlas.needed_times(case_events))

    combined_pdf = figures_dir / 'Figure_A1_primary_secondary_first_binary_coalescence_atlas_8rows.pdf'
    page_pdfs = []
    for page_number in range(1, rendered_pages + 1):
        page_events = events[events.page == page_number]
        figure = make_page(page_events, page_number, total_pages, case_cache)
        page_stem = f'Figure_A1_primary_secondary_8rows_page_{page_number:02d}_of_{total_pages:02d}'
        page_png = pages_dir / f'{page_stem}.png'
        page_pdf = pages_dir / f'{page_stem}.pdf'
        figure.savefig(page_png, dpi=240, facecolor='white')
        figure.savefig(page_pdf, facecolor='white')
        page_pdfs.append(page_pdf)
        plt.close(figure)
        print(f'Wrote page {page_number}/{total_pages}', flush=True)

    temporary_pdf = combined_pdf.with_suffix('.tmp.pdf')
    subprocess.run([
        'gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4', f'-sOutputFile={temporary_pdf}',
        *[str(path) for path in page_pdfs],
    ], check=True)
    temporary_pdf.replace(combined_pdf)
    return combined_pdf, total_pages, rendered_pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-pages', type=int, default=None)
    args = parser.parse_args()
    events = load_event_catalog()
    pdf, total_pages, rendered_pages = build_atlas(events, args.limit_pages)
    print(
        f'Events={len(events)}, cases={events.case.nunique()}, rows_per_page={ROWS_PER_PAGE}, '
        f'pages={total_pages}, rendered={rendered_pages}, PDF={pdf}'
    )


if __name__ == '__main__':
    main()
