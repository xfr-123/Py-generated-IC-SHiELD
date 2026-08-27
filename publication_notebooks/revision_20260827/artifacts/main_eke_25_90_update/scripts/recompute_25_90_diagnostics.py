#!/usr/bin/env python3
"""Recompute 25–90°N EKE, initial Eady means, and EKE-window heat fluxes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('PYGEN_DATA_ROOT', '/data/keeling/a/mingfei5/a/data/original'))
SOURCE_PACKAGE = ROOT / 'paper_revision' / 'section32_vertical_profiles_eke_fluxes_20260820'
SOURCE_SCRIPT = SOURCE_PACKAGE / 'scripts' / 'make_section32_figures.py'
OUTPUT = Path(os.environ.get('PYGEN_EKE_OUTPUT', str(ROOT / 'paper_revision' / 'eke_25_90_update_20260825')))
TABLES = OUTPUT / 'tables'
CHECKPOINTS = OUTPUT / 'checkpoints'
LATITUDE_MIN = 25.0
LATITUDE_MAX = 90.0
TIME_BLOCK = 12
FLUX_WIDTH = 15.0
EKE_COLUMN = 'eke_25_90N_area_mass_weighted_m2_s-2'
EADY_COLUMN = 'initial_eady_25_90N_area_mass_weighted_day-1'
GRAVITY = 9.80665
RD = 287.05
CP = 1004.0
KAPPA = RD / CP
OMEGA = 7.2921159e-5


def load_source_module():
    specification = importlib.util.spec_from_file_location('section32_source_25_90', SOURCE_SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(f'Cannot import {SOURCE_SCRIPT}')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def fill_float(values: object, dtype=np.float32) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        return np.ma.filled(values, np.nan).astype(dtype, copy=False)
    array = np.asarray(values, dtype=dtype)
    array[array <= -1.0e9] = np.nan
    return array


def pressure_weights(pressure_hpa: np.ndarray) -> np.ndarray:
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    if not np.all(np.diff(pressure_hpa) > 0.0):
        raise ValueError('Pressure levels must increase monotonically')
    weights = np.empty_like(pressure_hpa)
    weights[0] = 0.5 * (pressure_hpa[1] - pressure_hpa[0])
    weights[-1] = 0.5 * (pressure_hpa[-1] - pressure_hpa[-2])
    weights[1:-1] = 0.5 * (pressure_hpa[2:] - pressure_hpa[:-2])
    return weights / weights.sum()


def source_hour_indices(dataset: netCDF4.Dataset, ensemble: str) -> tuple[np.ndarray, np.ndarray]:
    source_time = fill_float(dataset.variables['time'][:], dtype=np.float64)
    desired_time = np.arange(1.0, 361.0)
    if ensemble == 'standard' and np.allclose(source_time[:360], desired_time):
        indices = np.arange(360, dtype=int)
    else:
        indices = np.array([int(np.argmin(np.abs(source_time - hour))) for hour in desired_time])
    if not np.allclose(source_time[indices], desired_time, atol=1.0e-5):
        raise RuntimeError(f'Could not identify hourly outputs in {dataset.filepath()}')
    return indices, desired_time


def read_time_block(variable, source_indices, start, stop, latitude_slice):
    selected = source_indices[start:stop]
    differences = np.diff(selected)
    if selected.size == 1 or np.all(differences == differences[0]):
        step = 1 if selected.size == 1 else int(differences[0])
        values = variable[int(selected[0]):int(selected[-1] + step):step, :, latitude_slice, :]
    else:
        values = variable[selected.tolist(), :, latitude_slice, :]
    return fill_float(values)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def checkpoint_paths(ensemble: str, case: str) -> dict[str, Path]:
    stem = f'{ensemble}__{case}'
    return {
        'eke': CHECKPOINTS / f'{stem}__eke_25_90.csv',
        'summary': CHECKPOINTS / f'{stem}__summary_25_90.json',
        'eady': CHECKPOINTS / f'{stem}__eady_25_90.json',
        'done': CHECKPOINTS / f'{stem}.done',
    }


def compute_initial_eady(zonal_wind, temperature, pressure_hpa, latitude):
    pressure_order = np.argsort(pressure_hpa)
    latitude_order = np.argsort(latitude)
    pressure = np.asarray(pressure_hpa, dtype=float)[pressure_order]
    selected_latitude = np.asarray(latitude, dtype=float)[latitude_order]
    zonal_wind = np.asarray(zonal_wind, dtype=float)[pressure_order][:, latitude_order]
    temperature = np.asarray(temperature, dtype=float)[pressure_order][:, latitude_order]
    theta = temperature * (1000.0 / pressure[:, None]) ** KAPPA
    log_pressure = np.log(pressure)
    du_dlogp = np.gradient(zonal_wind, log_pressure, axis=0, edge_order=2)
    dtheta_dlogp = np.gradient(theta, log_pressure, axis=0, edge_order=2)
    du_dz = -GRAVITY / (RD * temperature) * du_dlogp
    dtheta_dz = -GRAVITY / (RD * temperature) * dtheta_dlogp
    n_squared = GRAVITY / theta * dtheta_dz
    buoyancy_frequency = np.sqrt(np.where(n_squared > 1.0e-8, n_squared, np.nan))
    coriolis = 2.0 * OMEGA * np.sin(np.deg2rad(selected_latitude))
    eady = 0.31 * np.abs(coriolis)[None, :] * np.abs(du_dz) / buoyancy_frequency * 86400.0
    area_weights = np.cos(np.deg2rad(selected_latitude))
    area_weights /= area_weights.sum()
    weights = pressure_weights(pressure)[:, None] * area_weights[None, :]
    valid = np.isfinite(eady)
    denominator = np.sum(np.where(valid, weights, 0.0))
    if denominator <= 0.0:
        raise RuntimeError('No valid initial Eady values')
    return float(np.nansum(np.where(valid, eady * weights, 0.0)) / denominator), float(denominator)


def calculate_case(row: dict[str, object], force: bool = False) -> str:
    ensemble = str(row['ensemble'])
    case = str(row['case'])
    paths = checkpoint_paths(ensemble, case)
    required = [paths['eke'], paths['summary'], paths['done']]
    if ensemble == 'standard':
        required.append(paths['eady'])
    if not force and all(path.exists() for path in required):
        return f'cached {ensemble} {case}'

    source_path = Path(str(row['source_path']))
    with netCDF4.Dataset(source_path) as dataset:
        pressure = fill_float(dataset.variables['plev'][:], dtype=np.float64)
        latitude = fill_float(dataset.variables['grid_yt'][:], dtype=np.float64)
        latitude_indices = np.flatnonzero((latitude >= LATITUDE_MIN) & (latitude <= LATITUDE_MAX))
        latitude_slice = slice(int(latitude_indices[0]), int(latitude_indices[-1]) + 1)
        selected_latitude = latitude[latitude_indices]
        area_weights = np.cos(np.deg2rad(selected_latitude))
        area_weights /= area_weights.sum()
        mass_weights = pressure_weights(pressure)
        source_indices, hourly_time = source_hour_indices(dataset, ensemble)
        u_variable = dataset.variables['u_plev']
        v_variable = dataset.variables['v_plev']
        eke_values = np.full(360, np.nan, dtype=float)

        for start in range(0, 360, TIME_BLOCK):
            stop = min(start + TIME_BLOCK, 360)
            u_values = read_time_block(u_variable, source_indices, start, stop, latitude_slice)
            u_values -= np.nanmean(u_values, axis=-1, keepdims=True)
            zonal_u_variance = np.nanmean(u_values * u_values, axis=-1)
            del u_values
            v_values = read_time_block(v_variable, source_indices, start, stop, latitude_slice)
            v_values -= np.nanmean(v_values, axis=-1, keepdims=True)
            zonal_v_variance = np.nanmean(v_values * v_values, axis=-1)
            del v_values
            zonal_eke = 0.5 * (zonal_u_variance + zonal_v_variance)
            level_area_mean = np.nansum(zonal_eke * area_weights[None, None, :], axis=-1)
            eke_values[start:stop] = np.nansum(level_area_mean * mass_weights[None, :], axis=-1)

        eady_payload = None
        if ensemble == 'standard':
            first_u = read_time_block(u_variable, source_indices, 0, 1, latitude_slice)[0]
            first_t = read_time_block(dataset.variables['t_plev'], source_indices, 0, 1, latitude_slice)[0]
            eady_value, valid_fraction = compute_initial_eady(
                np.nanmean(first_u, axis=-1), np.nanmean(first_t, axis=-1), pressure, selected_latitude
            )
            eady_payload = {
                'ensemble': ensemble, 'case': case, 'b': float(row['b']), 'n': int(row['n']), 's': int(row['s']),
                EADY_COLUMN: eady_value, 'eady_valid_weight_fraction': valid_fraction,
                'latitude_requested_min_deg': LATITUDE_MIN, 'latitude_requested_max_deg': LATITUDE_MAX,
                'latitude_grid_min_deg': float(selected_latitude.min()), 'latitude_grid_max_deg': float(selected_latitude.max()),
                'pressure_top_hpa': float(pressure.min()), 'pressure_bottom_hpa': float(pressure.max()),
            }

    if not np.isfinite(eke_values).all():
        raise RuntimeError(f'Non-finite EKE values for {ensemble} {case}')
    peak_index = int(np.argmax(eke_values))
    peak_value = float(eke_values[peak_index])
    rising_values = eke_values[:peak_index + 1]
    start_candidates = np.flatnonzero(rising_values >= 0.50 * peak_value)
    start_index = int(start_candidates[0]) if start_candidates.size else 0
    end_candidates = np.flatnonzero(rising_values[start_index:] >= 0.80 * peak_value)
    end_index = int(start_index + end_candidates[0]) if end_candidates.size else peak_index
    eke_frame = pd.DataFrame({
        'ensemble': ensemble, 'case': case, 'b': float(row['b']), 'n': int(row['n']), 's': int(row['s']),
        'time_h': hourly_time, EKE_COLUMN: eke_values,
    })
    summary = {
        'ensemble': ensemble, 'case': case, 'b': float(row['b']), 'n': int(row['n']), 's': int(row['s']),
        'eke_peak_25_90N_area_mass_weighted_m2_s-2': peak_value,
        'eke_peak_time_h': float(hourly_time[peak_index]),
        'eke_50pct_window_start_h': float(hourly_time[start_index]),
        'eke_80pct_window_end_h': float(hourly_time[end_index]),
        'eke_50_80pct_window_count': int(end_index - start_index + 1),
        'latitude_requested_min_deg': LATITUDE_MIN, 'latitude_requested_max_deg': LATITUDE_MAX,
        'latitude_grid_min_deg': float(selected_latitude.min()), 'latitude_grid_max_deg': float(selected_latitude.max()),
    }
    atomic_write_csv(eke_frame, paths['eke'])
    atomic_write_json(summary, paths['summary'])
    if eady_payload is not None:
        atomic_write_json(eady_payload, paths['eady'])
    paths['done'].write_text('complete\n', encoding='utf-8')
    return f'computed {ensemble} {case}'


def collect_checkpoints(metadata):
    eke_frames, summaries, eady_rows = [], [], []
    for row in metadata.itertuples(index=False):
        paths = checkpoint_paths(str(row.ensemble), str(row.case))
        eke_frames.append(pd.read_csv(paths['eke']))
        summaries.append(json.loads(paths['summary'].read_text(encoding='utf-8')))
        if row.ensemble == 'standard':
            eady_rows.append(json.loads(paths['eady'].read_text(encoding='utf-8')))
    eke = pd.concat(eke_frames, ignore_index=True).sort_values(['ensemble', 'b', 'n', 's', 'time_h'])
    summary = pd.DataFrame(summaries).sort_values(['ensemble', 'b', 'n', 's'])
    eady = pd.DataFrame(eady_rows).sort_values(['b', 'n', 's'])
    return eke, summary, eady


def recompute_heat_flux_profiles(metadata, summary):
    source = load_source_module()
    window_lookup = summary.set_index(['ensemble', 'case'])
    rows = []
    for index, metadata_row in enumerate(metadata.itertuples(index=False), start=1):
        ensemble, case = str(metadata_row.ensemble), str(metadata_row.case)
        window = window_lookup.loc[(ensemble, case)]
        start_hour = float(window.eke_50pct_window_start_h)
        end_hour = float(window.eke_80pct_window_end_h)
        with netCDF4.Dataset(source.CASE_RESULTS / f'{ensemble}__{case}.nc') as result:
            result_time = fill_float(result.variables['time'][:], dtype=np.float64)
            selected = np.flatnonzero((result_time >= start_hour - 1e-5) & (result_time <= end_hour + 1e-5))
            expected_count = int(window.eke_50_80pct_window_count)
            if selected.size != expected_count:
                raise RuntimeError(f'Window mismatch for {ensemble} {case}: {selected.size} != {expected_count}')
            widths = fill_float(result.variables['jet_half_width'][:], dtype=np.float64)
            width_index = int(np.argmin(np.abs(widths - FLUX_WIDTH)))
            values = fill_float(result.variables['eddy_heat_flux_vT'][selected, width_index, :], dtype=np.float64)
            means = np.nanmean(values, axis=0)
            pressures = fill_float(result.variables['plev'][:], dtype=np.float64)
        for pressure, value in zip(pressures, means):
            rows.append({
                'ensemble': ensemble, 'case': case, 'b': float(metadata_row.b), 'n': int(metadata_row.n), 's': int(metadata_row.s),
                'variable': 'eddy_heat_flux_vT', 'pressure_hpa': float(pressure), 'value': float(value),
                'window_start_h': start_hour, 'window_end_h': end_hour, 'window_count': expected_count,
                'eke_domain': '25–90°N area–mass weighted', 'jet_relative_half_width_deg': FLUX_WIDTH,
            })
        print(f'heat flux [{index:02d}/{len(metadata):02d}] {ensemble} {case}', flush=True)
    return pd.DataFrame(rows).sort_values(['ensemble', 'b', 'n', 's', 'pressure_hpa'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    source = load_source_module()
    metadata = source.case_metadata()
    rows = metadata.to_dict('records')
    if args.workers <= 1:
        for index, row in enumerate(rows, start=1):
            print(f'[{index:02d}/{len(rows):02d}] {calculate_case(row, args.force)}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(calculate_case, row, args.force): row for row in rows}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                print(f'[{completed:02d}/{len(rows):02d}] {future.result()}', flush=True)
    eke, summary, eady = collect_checkpoints(metadata)
    if len(eke) != 90 * 360 or eke.case.nunique() != 90:
        raise RuntimeError('Incomplete all-90 EKE table')
    if eady.case.nunique() != 45:
        raise RuntimeError('Incomplete standard Eady table')
    heat_flux = recompute_heat_flux_profiles(metadata, summary)
    if len(heat_flux) != 90 * 15 or heat_flux.case.nunique() != 90:
        raise RuntimeError('Incomplete all-90 heat-flux table')
    eke.to_csv(TABLES / 'eke_25_90N_area_mass_weighted_timeseries_all90.csv', index=False)
    summary.to_csv(TABLES / 'eke_25_90N_windows_all90.csv', index=False)
    eady.to_csv(TABLES / 'initial_eady_25_90N_area_mass_weighted_standard_all45.csv', index=False)
    heat_flux.to_csv(TABLES / 'eddy_heat_flux_profiles_50_80pct_peak_eke_25_90N_all90.csv', index=False)
    print('diagnostic tables complete', flush=True)


if __name__ == '__main__':
    main()
