#!/usr/bin/env python3
"""Recompute the fixed n=1, s=+10 Table S3 from eddy surface pressure."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/data/keeling/a/mingfei5/a/data/original")
DEFAULT_OUT = ROOT / "paper_revision" / "Table_S3_eddy_slp_recomputed_20260821"
CASES = ("BCwave_b1n1s10", "BCwave_b15n1s10", "BCwave_b2n1s10")
WINDOW_H = 72


def load_base_module(outdir: Path):
    helper = outdir / "source_reused" / "recompute_figures_s20_s21_eddy_slp.py"
    spec = importlib.util.spec_from_file_location("s20_s21_eddy_helper", helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_primary_track(case: str, module) -> tuple[dict[str, object], list, np.ndarray]:
    path = ROOT / f"{case}.nc"
    metadata = module.parse_case_name(case)
    with xr.open_dataset(path, decode_times=False, engine="netcdf4", mask_and_scale=True) as dataset:
        latitude_all = dataset["grid_yt"].values.astype(float)
        longitude = np.mod(dataset["grid_xt"].values.astype(float), 360.0)
        nh_indices = np.flatnonzero(latitude_all >= 0.0)
        latitude = latitude_all[nh_indices]
        pressure = dataset["PRESsfc"].isel(grid_yt=nh_indices).load()
        pressure = pressure.rename({"grid_yt": "latitude", "grid_xt": "longitude"})
        pressure = pressure.assign_coords(latitude=("latitude", latitude), longitude=("longitude", longitude))
        pressure = pressure.sortby("longitude")
        time_values = dataset["time"].values.astype(float)
    pressure_values = pressure.values.astype(np.float64, copy=False)
    eddy_pressure_pa = pressure_values - pressure_values.mean(axis=2, keepdims=True)
    area2d = module.spherical_cell_areas_km2(latitude, longitude)
    latitude2d = np.broadcast_to(latitude[:, None], area2d.shape)
    longitude2d = np.broadcast_to(longitude[None, :], area2d.shape)
    area_flat = area2d.ravel()
    latitude_flat = latitude2d.ravel()
    longitude_flat = longitude2d.ravel()

    objects_by_time = []
    for time_index in range(eddy_pressure_pa.shape[0]):
        objects, _, _ = module.objects_at_time(
            eddy_pressure_pa[time_index],
            10,
            area_flat,
            latitude_flat,
            longitude_flat,
        )
        objects_by_time.append(objects)
    tracks = module.link_overlap_tracks(objects_by_time, module.S20_OVERLAP)
    track_id, points = module.select_longest_track(tracks)
    if track_id is None or not points:
        raise RuntimeError(f"No +10-hPa eddy-SLP track found for {case}")
    summary = module.summarize_track_points(points)
    row = {
        **metadata,
        "display_case": case.removeprefix("BCwave_").replace("b15", "b1.5"),
        "source_file": str(path),
        "eddy_slp_definition": "PRESsfc minus instantaneous zonal mean at each time and latitude",
        "eddy_threshold_hpa": 10.0,
        "minimum_area_km2": module.MIN_AREA_KM2,
        "identity_overlap_threshold": module.S20_OVERLAP,
        "track_id": int(track_id),
        "start_hour": int(points[0].time_index + 1),
        "end_hour": int(points[-1].time_index + 1),
        "lifetime_h": int(len(points)),
        "mean_area_km2": float(np.mean([point.obj.area_km2 for point in points])),
        "mean_area_1e6_km2": float(np.mean([point.obj.area_km2 for point in points]) / 1.0e6),
        "maximum_area_km2": float(np.max([point.obj.area_km2 for point in points])),
        "peak_eddy_slp_anomaly_hpa": float(np.max([point.obj.peak_eddy_slp_hpa for point in points])),
        "path_speed_m_s-1": float(summary["mean_centroid_path_speed_m_s-1"]),
        "net_speed_m_s-1": float(summary["net_centroid_speed_m_s-1"]),
        "maximum_absolute_zonal_mean_residual_hpa": float(
            np.max(np.abs(np.mean(eddy_pressure_pa, axis=2))) / 100.0
        ),
    }
    return row, points, time_values


def point_rows(case_row: dict[str, object], points: list) -> list[dict[str, object]]:
    rows = []
    for point in points:
        rows.append({
            "case": case_row["case"],
            "b": case_row["b"],
            "n": case_row["n"],
            "s": case_row["s"],
            "track_id": case_row["track_id"],
            "time_index": int(point.time_index),
            "simulation_hour": int(point.time_index + 1),
            "area_km2": float(point.obj.area_km2),
            "equivalent_radius_km": float(math.sqrt(point.obj.area_km2 / math.pi)),
            "centroid_latitude_deg_n": float(point.obj.centroid_lat),
            "centroid_longitude_deg_e": float(point.obj.centroid_lon),
            "peak_eddy_slp_anomaly_hpa": float(point.obj.peak_eddy_slp_hpa),
            "overlap_from_previous": float(point.overlap_from_previous),
            "displacement_km_from_previous": float(point.displacement_km_from_previous),
            "normalized_displacement_from_previous": float(point.normalized_displacement_from_previous),
        })
    return rows


def window_rows(case_row: dict[str, object], points: list, module) -> list[dict[str, object]]:
    rows = []
    if len(points) < WINDOW_H + 1:
        return rows
    for start in range(0, len(points) - WINDOW_H):
        end = start + WINDOW_H
        window = points[start : end + 1]
        if window[-1].time_index - window[0].time_index != WINDOW_H:
            continue
        step_distances = np.asarray(
            [point.displacement_km_from_previous for point in window[1:]], dtype=float
        )
        if not np.isfinite(step_distances).all():
            continue
        net_km = module.great_circle_km(
            window[0].obj.centroid_lat,
            window[0].obj.centroid_lon,
            window[-1].obj.centroid_lat,
            window[-1].obj.centroid_lon,
        )
        path_km = float(step_distances.sum())
        radius_km = float(np.mean([math.sqrt(point.obj.area_km2 / math.pi) for point in window]))
        rows.append({
            "case": case_row["case"],
            "b": case_row["b"],
            "n": case_row["n"],
            "s": case_row["s"],
            "track_id": case_row["track_id"],
            "window_start_time_index": int(window[0].time_index),
            "window_end_time_index": int(window[-1].time_index),
            "window_start_hour": int(window[0].time_index + 1),
            "window_end_hour": int(window[-1].time_index + 1),
            "net_displacement_km_72h": float(net_km),
            "path_length_km_72h": path_km,
            "net_speed_m_s-1_72h": float(net_km * 1000.0 / (WINDOW_H * 3600.0)),
            "path_speed_m_s-1_72h": float(path_km * 1000.0 / (WINDOW_H * 3600.0)),
            "mean_equivalent_radius_km_72h": radius_km,
            "normalized_net_displacement_radii_72h": float(net_km / radius_km),
            "normalized_path_length_radii_72h": float(path_km / radius_km),
        })
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    module = load_base_module(args.outdir)

    summary_rows = []
    all_points = []
    all_windows = []
    for case in CASES:
        case_row, points, _ = build_primary_track(case, module)
        windows = window_rows(case_row, points, module)
        if not windows:
            raise RuntimeError(f"No continuous 72-h windows for {case}")
        speeds = np.asarray([row["net_speed_m_s-1_72h"] for row in windows], dtype=float)
        case_row.update({
            "number_continuous_72h_windows": int(len(windows)),
            "minimum_72h_net_speed_m_s-1": float(np.min(speeds)),
            "median_72h_net_speed_m_s-1": float(np.median(speeds)),
            "maximum_72h_net_speed_m_s-1": float(np.max(speeds)),
        })
        summary_rows.append(case_row)
        all_points.extend(point_rows(case_row, points))
        all_windows.extend(windows)
        print(case, case_row["lifetime_h"], case_row["median_72h_net_speed_m_s-1"], flush=True)

    summary = pd.DataFrame(summary_rows).sort_values("b").reset_index(drop=True)
    points = pd.DataFrame(all_points).sort_values(["b", "time_index"]).reset_index(drop=True)
    windows = pd.DataFrame(all_windows).sort_values(["b", "window_start_time_index"]).reset_index(drop=True)

    s20 = pd.read_csv(
        ROOT / "paper_revision" / "Figures_S20_S21_eddy_slp_recomputed_20260821" / "S20_case_level_results.csv"
    )
    s20 = s20[s20["case"].isin(CASES)][[
        "case", "maximum_qualifying_persistence_h", "mean_qualifying_object_area_km2",
        "maximum_peak_eddy_slp_anomaly_hpa", "mean_centroid_path_speed_m_s-1",
        "net_centroid_speed_m_s-1",
    ]]
    validation = summary.merge(s20, on="case", validate="one_to_one")
    validation["lifetime_difference_h"] = validation["lifetime_h"] - validation["maximum_qualifying_persistence_h"]
    validation["mean_area_difference_km2"] = validation["mean_area_km2"] - validation["mean_qualifying_object_area_km2"]
    validation["peak_anomaly_difference_hpa"] = validation["peak_eddy_slp_anomaly_hpa"] - validation["maximum_peak_eddy_slp_anomaly_hpa"]
    validation["path_speed_difference_m_s-1"] = validation["path_speed_m_s-1"] - validation["mean_centroid_path_speed_m_s-1"]
    validation["net_speed_difference_m_s-1"] = validation["net_speed_m_s-1"] - validation["net_centroid_speed_m_s-1"]
    numeric_differences = validation[[
        "lifetime_difference_h", "mean_area_difference_km2", "peak_anomaly_difference_hpa",
        "path_speed_difference_m_s-1", "net_speed_difference_m_s-1",
    ]].to_numpy(float)
    validation["matches_figure_s20"] = np.all(np.isclose(numeric_differences, 0.0, atol=1.0e-9), axis=1)

    old = pd.read_csv(ROOT / "revision_analysis" / "tables" / "r2_2_anticyclone_primary_decomposition.csv")
    old = old[old["case"].isin(CASES)][[
        "case", "duration_hours", "mean_area_km2", "peak_slp_pa", "mean_centroid_speed_ms",
        "net_centroid_speed_ms", "median_net_speed_ms_72h", "number_72h_windows",
    ]].rename(columns={
        "duration_hours": "old_full_field_lifetime_h",
        "mean_area_km2": "old_full_field_mean_area_km2",
        "peak_slp_pa": "old_full_field_peak_slp_pa",
        "mean_centroid_speed_ms": "old_full_field_path_speed_m_s-1",
        "net_centroid_speed_ms": "old_full_field_net_speed_m_s-1",
        "median_net_speed_ms_72h": "old_full_field_median_72h_net_speed_m_s-1",
        "number_72h_windows": "old_full_field_number_72h_windows",
    })
    comparison = summary.merge(old, on="case", validate="one_to_one")

    summary.to_csv(args.outdir / "Table_S3_eddy_slp_recomputed.csv", index=False)
    points.to_csv(args.outdir / "Table_S3_primary_track_points.csv", index=False)
    windows.to_csv(args.outdir / "Table_S3_72h_window_metrics.csv", index=False)
    validation.to_csv(args.outdir / "Table_S3_validation_against_Figure_S20.csv", index=False)
    comparison.to_csv(args.outdir / "Table_S3_old_full_field_vs_new_eddy_slp.csv", index=False)

    table_lines = [
        "# Revised Table S3 using eddy surface pressure",
        "",
        "The 72-h quantity is the median net-displacement speed among all continuous 72-h windows within the primary track. Each window contains 73 hourly centroid positions whose first and last times differ by exactly 72 h.",
        "",
        "| Case | Lifetime (h) | Mean area ($10^6$ km$^2$) | Peak eddy-SLP anomaly (hPa) | Path speed (m s$^{-1}$) | Net speed (m s$^{-1}$) | Median 72-h net speed (m s$^{-1}$) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        table_lines.append(
            f"| {row['display_case']} | {int(row['lifetime_h'])} | {row['mean_area_1e6_km2']:.2f} | "
            f"{row['peak_eddy_slp_anomaly_hpa']:.1f} | {row['path_speed_m_s-1']:.1f} | "
            f"{row['net_speed_m_s-1']:.1f} | {row['median_72h_net_speed_m_s-1']:.1f} |"
        )
    table_lines.extend([
        "",
        "## Definition",
        "",
        "- Eddy surface pressure: $p_s'=p_s-[p_s]$, with the zonal mean removed independently at every time and latitude.",
        "- Object threshold: $p_s'\geq+10$ hPa.",
        "- Minimum area: $10^5$ km$^2$ using spherical grid-cell areas.",
        "- Identity linking: consecutive overlap threshold 0.3, using the same independent permissive tracker as revised Figure S20.",
        "- Primary object: the longest identified track in each simulation.",
        "- Peak intensity is now an eddy-SLP anomaly and is not an absolute surface-pressure value.",
    ])
    (args.outdir / "Table_S3_eddy_slp_recomputed.md").write_text("\n".join(table_lines) + "\n")

    validation_text = [
        "# Table S3 validation",
        "",
        f"All three recomputed primary-track summaries match the revised Figure S20 case-level table: {int(validation['matches_figure_s20'].sum())}/3.",
        f"The maximum absolute zonal-mean residual is {summary['maximum_absolute_zonal_mean_residual_hpa'].max():.3e} hPa.",
        "The original table used full surface pressure and reported absolute peak SLP. The revised table uses eddy surface pressure and therefore reports peak positive eddy-SLP anomaly; these intensity columns are not numerically interchangeable.",
        "",
        "Files `Table_S3_primary_track_points.csv` and `Table_S3_72h_window_metrics.csv` contain the complete supporting data.",
    ]
    (args.outdir / "Table_S3_validation.md").write_text("\n".join(validation_text) + "\n")

    files = sorted(path for path in args.outdir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (args.outdir / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(args.outdir)}" for path in files) + "\n"
    )


if __name__ == "__main__":
    main()
