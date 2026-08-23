#!/usr/bin/env python3
"""Replot the three anticyclone diagnostics using eddy surface pressure.

The original figures used full-field surface pressure.  This script instead
uses p_s* = p_s - [p_s] for surface-pressure contours, Hovmöller shading, and
high-pressure-object tracking.  Because p_s* has a zero instantaneous zonal
mean, the high-pressure object threshold is +10 hPa rather than 1010 hPa.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path as MplPath
from scipy import ndimage

try:
    import cartopy.crs as ccrs
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Cartopy is required for Figure 11") from exc


ROOT = Path("/data/keeling/a/mingfei5/a/data/original")
OUT = ROOT / "paper_revision" / "hovmoller_2hpa_pm10_blabels_replot_20260819"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
SOURCE = OUT / "source"

EARTH_RADIUS_KM = 6371.0
OMEGA = 7.2921e-5
OMEGA_LEVELS = np.array([-0.24, -0.20, -0.16, -0.08, -0.04, 0.04, 0.08, 0.16, 0.20, 0.24])
EDDY_CONTOUR_LEVELS = np.array([-40, -30, -20, -10, 10, 20, 30, 40], dtype=float)
EDDY_HOV_LEVELS = np.concatenate((
    np.arange(-10.0, 0.0, 2.0),
    np.arange(2.0, 12.0, 2.0),
))
TEMP_LEVELS = np.arange(-30, 31, 3, dtype=float)
PRESSURE_TRACK_THRESHOLD_HPA = 10.0
OVERLAP_THRESHOLD = 0.80
MIN_PERSISTENCE_H = 72
B_FILE_VALUES = [2, 15, 1]
B_DISPLAY_VALUES = [2.0, 1.5, 1.0]
N_VALUES = [6, 3, 1]
S_VALUES = [-10, -5, 0, 5, 10]


@dataclass
class PressureObject:
    pixels: np.ndarray
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    peak_anomaly_hpa: float


@dataclass
class TrackPoint:
    time_index: int
    obj: PressureObject
    overlap_from_previous: float = math.nan


def ensure_directories() -> None:
    for directory in (FIGURES, TABLES, LOGS, SOURCE):
        directory.mkdir(parents=True, exist_ok=True)


def case_name(b_file_value: int, n_value: int, shift: int | None = None) -> str:
    stem = f"BCwave_b{b_file_value}n{n_value}"
    if shift not in (None, 0):
        stem += f"s{shift}"
    return stem


def make_white_center_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    coolwarm = plt.get_cmap("coolwarm")
    negative_intervals = int(np.sum(EDDY_HOV_LEVELS < 0) - 1)
    positive_intervals = int(np.sum(EDDY_HOV_LEVELS > 0) - 1)
    negative = [coolwarm(value) for value in np.linspace(0.03, 0.45, negative_intervals)]
    positive = [coolwarm(value) for value in np.linspace(0.55, 0.97, positive_intervals)]
    cmap = ListedColormap(negative + [[1.0, 1.0, 1.0, 1.0]] + positive)
    return cmap, BoundaryNorm(EDDY_HOV_LEVELS, ncolors=cmap.N, clip=False)


def cyclic_connected_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, nlabels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if nlabels == 0:
        return labels, 0
    parent = np.arange(nlabels + 1, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for row in range(labels.shape[0]):
        left_label = int(labels[row, 0])
        if left_label == 0:
            continue
        for neighbor_row in (row - 1, row, row + 1):
            if 0 <= neighbor_row < labels.shape[0]:
                right_label = int(labels[neighbor_row, -1])
                if right_label > 0:
                    union(left_label, right_label)
    root_map = np.arange(nlabels + 1, dtype=np.int32)
    for label in range(1, nlabels + 1):
        root_map[label] = find(label)
    root_labels = root_map[labels]
    roots = np.unique(root_labels[root_labels > 0])
    compact = np.zeros(nlabels + 1, dtype=np.int32)
    for new_label, root in enumerate(roots, start=1):
        compact[int(root)] = new_label
    return compact[root_labels], len(roots)


def spherical_cell_areas_km2(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    latitude_edges = np.empty(latitude.size + 1, dtype=float)
    latitude_edges[1:-1] = 0.5 * (latitude[:-1] + latitude[1:])
    latitude_edges[0] = max(-90.0, latitude[0] - 0.5 * (latitude[1] - latitude[0]))
    latitude_edges[-1] = min(90.0, latitude[-1] + 0.5 * (latitude[-1] - latitude[-2]))
    longitude_step = np.median(np.diff(np.unwrap(np.deg2rad(longitude))))
    strip = EARTH_RADIUS_KM**2 * longitude_step * (
        np.sin(np.deg2rad(latitude_edges[1:]))
        - np.sin(np.deg2rad(latitude_edges[:-1]))
    )
    return np.broadcast_to(strip[:, None], (latitude.size, longitude.size)).copy()


def spherical_centroid(
    pixels: np.ndarray,
    latitude_flat: np.ndarray,
    longitude_flat: np.ndarray,
    area_flat: np.ndarray,
) -> tuple[float, float]:
    weights = area_flat[pixels]
    latitude_rad = np.deg2rad(latitude_flat[pixels])
    longitude_rad = np.deg2rad(longitude_flat[pixels])
    x = np.sum(weights * np.cos(latitude_rad) * np.cos(longitude_rad))
    y = np.sum(weights * np.cos(latitude_rad) * np.sin(longitude_rad))
    z = np.sum(weights * np.sin(latitude_rad))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0 or not np.isfinite(norm):
        return float("nan"), float("nan")
    return (
        math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
        math.degrees(math.atan2(y, x)) % 360.0,
    )


def objects_at_time(
    eddy_pressure_hpa: np.ndarray,
    area: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    threshold_hpa: float,
) -> list[PressureObject]:
    mask = np.isfinite(eddy_pressure_hpa) & (eddy_pressure_hpa >= threshold_hpa)
    labels, nlabels = cyclic_connected_labels(mask)
    if nlabels == 0:
        return []
    labels_flat = labels.ravel()
    field_flat = eddy_pressure_hpa.ravel()
    area_flat = area.ravel()
    latitude2d, longitude2d = np.meshgrid(latitude, longitude, indexing="ij")
    latitude_flat = latitude2d.ravel()
    longitude_flat = longitude2d.ravel()
    objects: list[PressureObject] = []
    for label in range(1, nlabels + 1):
        pixels = np.flatnonzero(labels_flat == label).astype(np.int32)
        centroid_lat, centroid_lon = spherical_centroid(
            pixels, latitude_flat, longitude_flat, area_flat
        )
        objects.append(
            PressureObject(
                pixels=pixels,
                area_km2=float(np.sum(area_flat[pixels])),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                peak_anomaly_hpa=float(np.nanmax(field_flat[pixels])),
            )
        )
    return objects


def overlap_fraction(first: PressureObject, second: PressureObject, area_flat: np.ndarray) -> float:
    common = np.intersect1d(first.pixels, second.pixels, assume_unique=True)
    if common.size == 0:
        return 0.0
    return float(np.sum(area_flat[common]) / min(first.area_km2, second.area_km2))


def track_objects(
    objects_by_time: list[list[PressureObject]],
    area: np.ndarray,
    overlap_threshold: float,
) -> dict[int, list[TrackPoint]]:
    area_flat = area.ravel()
    tracks: dict[int, list[TrackPoint]] = {}
    active: dict[int, PressureObject] = {}
    next_track_id = 1
    for time_index, current_objects in enumerate(objects_by_time):
        candidates: list[tuple[float, int, int]] = []
        for track_id, previous_object in active.items():
            for object_index, current_object in enumerate(current_objects):
                overlap = overlap_fraction(previous_object, current_object, area_flat)
                if overlap >= overlap_threshold:
                    candidates.append((overlap, track_id, object_index))
        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_objects: set[int] = set()
        assignments: dict[int, tuple[int, float]] = {}
        for overlap, track_id, object_index in candidates:
            if track_id in used_tracks or object_index in used_objects:
                continue
            used_tracks.add(track_id)
            used_objects.add(object_index)
            assignments[object_index] = (track_id, overlap)
        new_active: dict[int, PressureObject] = {}
        for object_index, current_object in enumerate(current_objects):
            if object_index in assignments:
                track_id, overlap = assignments[object_index]
            else:
                track_id = next_track_id
                next_track_id += 1
                overlap = math.nan
            tracks.setdefault(track_id, []).append(
                TrackPoint(time_index=time_index, obj=current_object, overlap_from_previous=overlap)
            )
            new_active[track_id] = current_object
        active = new_active
    return tracks


def eddy_pressure_from_dataset(dataset: xr.Dataset, time_indices: np.ndarray | slice | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pressure = dataset["PRESsfc"] / 100.0
    selected_time = dataset["time"]
    if time_indices is not None:
        pressure = pressure.isel(time=time_indices)
        selected_time = selected_time.isel(time=time_indices)
    pressure = pressure.load()
    eddy_pressure = pressure - pressure.mean(dim="grid_xt")
    return (
        eddy_pressure.values.astype(np.float32),
        pressure["grid_yt"].values.astype(float),
        pressure["grid_xt"].values.astype(float),
        selected_time.load().values.astype(float),
    )


def polar_boundary(ax) -> None:
    angle = np.linspace(0, 2 * np.pi, 400)
    vertices = np.column_stack([0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)])
    codes = np.full(len(angle), MplPath.LINETO, dtype=MplPath.code_type)
    codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
    ax.set_boundary(MplPath(vertices, codes), transform=ax.transAxes)


def add_polar_grid(ax, column: int) -> None:
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linestyle="--",
        linewidth=0.45,
        color="0.55",
        alpha=0.7,
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
    )
    gridlines.xlocator = mticker.FixedLocator([-150, -90, 0, 90, 150, 180])
    gridlines.xformatter = mticker.FuncFormatter(
        lambda value, position: (
            "90°W" if value == -90 and column == 0 else
            "0°" if value == 0 else
            "90°E" if value == 90 and column == 2 else
            "180°" if value == 180 else ""
        )
    )
    gridlines.xlabels_top = True
    gridlines.xlabels_bottom = False
    gridlines.ylabels_left = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style = {"size": 10}


def find_high_centers(field: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> list[tuple[float, float]]:
    latitude_mask = (latitude >= 30.0) & (latitude <= 82.0)
    subset = field[latitude_mask]
    mask = np.isfinite(subset) & (subset >= PRESSURE_TRACK_THRESHOLD_HPA)
    labels, nlabels = cyclic_connected_labels(mask)
    if nlabels == 0:
        return []
    area = spherical_cell_areas_km2(latitude[latitude_mask], longitude)
    area_flat = area.ravel()
    latitude2d, longitude2d = np.meshgrid(latitude[latitude_mask], longitude, indexing="ij")
    candidates: list[tuple[float, float, float]] = []
    for label in range(1, nlabels + 1):
        pixels = np.flatnonzero(labels.ravel() == label)
        if pixels.size < 4:
            continue
        peak = float(np.nanmax(subset.ravel()[pixels]))
        area_value = float(np.sum(area_flat[pixels]))
        lat_center, lon_center = spherical_centroid(
            pixels, latitude2d.ravel(), longitude2d.ravel(), area_flat
        )
        candidates.append((area_value, peak, lat_center, lon_center))
    candidates.sort(reverse=True)
    return [(row[2], row[3]) for row in candidates[:2]]


def find_regional_peak(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
) -> tuple[float, float] | None:
    latitude_mask = (latitude >= latitude_bounds[0]) & (latitude <= latitude_bounds[1])
    longitude_mask = (longitude >= longitude_bounds[0]) & (longitude <= longitude_bounds[1])
    subset = field[latitude_mask][:, longitude_mask]
    if subset.size == 0 or not np.isfinite(subset).any():
        return None
    row, column = np.unravel_index(np.nanargmax(subset), subset.shape)
    if subset[row, column] < PRESSURE_TRACK_THRESHOLD_HPA:
        return None
    return float(latitude[latitude_mask][row]), float(longitude[longitude_mask][column])


def plot_figure11() -> Path:
    case = "BCwave_b2n1s10"
    path = ROOT / f"{case}.nc"
    with xr.open_dataset(path, decode_times=False) as dataset:
        pressure_eddy, latitude, longitude, _ = eddy_pressure_from_dataset(dataset)
        omega = dataset["omg_plev"].sel(plev=500).isel(time=[168, 192, 216, 240, 264, 288]).load().values
        temperature = dataset["t_plev"].sel(plev=1000).isel(time=[240, 288, 336]).load().values
        time_values = dataset["time"].values.astype(float)
    top_indices = [168, 192, 216, 240, 264, 288]
    bottom_indices = [240, 288, 336]
    figure = plt.figure(figsize=(12.4, 16.2), facecolor="white")
    grid = figure.add_gridspec(
        6, 3, height_ratios=[1.0, 1.0, 0.18, 0.64, 0.64, 0.64],
        left=0.06, right=0.90, bottom=0.045, top=0.955,
        wspace=0.12, hspace=0.34,
    )
    top_axes = []
    omega_image = None
    letters = list("abcdefghi")
    for panel_index, (time_index, omega_field) in enumerate(zip(top_indices, omega)):
        row, column = divmod(panel_index, 3)
        axis = figure.add_subplot(grid[row, column], projection=ccrs.NorthPolarStereo(central_longitude=0))
        top_axes.append(axis)
        axis.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
        polar_boundary(axis)
        add_polar_grid(axis, column)
        omega_plot = np.where(np.abs(omega_field) >= 0.08, omega_field, 0.0)
        omega_image = axis.contourf(
            longitude, latitude, omega_plot, levels=OMEGA_LEVELS,
            cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree(), alpha=0.68,
        )
        pressure_field = pressure_eddy[time_index]
        axis.contour(
            longitude, latitude, pressure_field, levels=EDDY_CONTOUR_LEVELS,
            colors="black", linewidths=0.90, transform=ccrs.PlateCarree(),
        )
        if panel_index >= 3:
            centers = [
                find_regional_peak(pressure_field, latitude, longitude, (35.0, 75.0), (180.0, 270.0)),
                find_regional_peak(pressure_field, latitude, longitude, (35.0, 75.0), (90.0, 180.0)),
            ]
            centers = [center for center in centers if center is not None]
        else:
            centers = find_high_centers(pressure_field, latitude, longitude)
        for center_lat, center_lon in centers:
            label = axis.text(
                center_lon, center_lat, "H", transform=ccrs.PlateCarree(),
                color="#d7191c", fontsize=23, fontweight="bold",
                ha="center", va="center", zorder=11,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white")])
        if panel_index in (3, 5):
            box_lons = [100, 250, 250, 100, 100]
            box_lats = [30, 30, 80, 80, 30]
            axis.plot(box_lons, box_lats, transform=ccrs.PlateCarree(), color="black", linewidth=2.0, zorder=10)
        day = int(round(time_values[time_index] / 24.0))
        axis.set_title(f"({letters[panel_index]}) Day {day}", fontsize=15, pad=10)

    bottom_axes = []
    temperature_image = None
    latitude_mask = (latitude >= 35.0) & (latitude <= 65.0)
    longitude_mask = (longitude >= 100.0) & (longitude <= 250.0)
    pressure_contour_latitude_mask = latitude_mask
    for offset, (time_index, temperature_field) in enumerate(zip(bottom_indices, temperature)):
        axis = figure.add_subplot(grid[3 + offset, :])
        bottom_axes.append(axis)
        temperature_subset = temperature_field[latitude_mask]
        temperature_anomaly = temperature_subset - temperature_subset.mean(axis=-1, keepdims=True)
        longitude_subset = longitude[longitude_mask]
        temperature_anomaly = temperature_anomaly[:, longitude_mask]
        temperature_image = axis.contourf(
            longitude_subset, latitude[latitude_mask], temperature_anomaly,
            levels=TEMP_LEVELS, cmap="RdBu_r", extend="both",
        )
        pressure_subset = pressure_eddy[time_index][pressure_contour_latitude_mask][:, longitude_mask]
        negative_levels = EDDY_CONTOUR_LEVELS[EDDY_CONTOUR_LEVELS < 0]
        positive_levels = EDDY_CONTOUR_LEVELS[EDDY_CONTOUR_LEVELS > 0]
        axis.contour(longitude_subset, latitude[latitude_mask], pressure_subset, levels=positive_levels, colors="black", linewidths=0.88)
        negative_contours = axis.contour(longitude_subset, latitude[latitude_mask], pressure_subset, levels=negative_levels, colors="black", linewidths=0.76, linestyles="dashed")
        axis.clabel(negative_contours, inline=True, fontsize=9, fmt="%g")
        high_center = find_regional_peak(
            pressure_eddy[time_index], latitude, longitude, (35.0, 65.0), (100.0, 250.0)
        )
        if high_center is not None:
            high_latitude, high_longitude = high_center
            label = axis.text(
                high_longitude, high_latitude, "H", color="#d7191c",
                fontsize=17, fontweight="bold", ha="center", va="center", zorder=9,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
        axis.set_xlim(100, 250)
        axis.set_ylim(35, 65)
        axis.set_yticks([35, 45, 55, 65])
        axis.tick_params(labelsize=10.5)
        axis.grid(True, color="0.55", alpha=0.35, linewidth=0.50)
        axis.set_ylabel("Latitude", fontsize=12)
        axis.set_title(f"({letters[6 + offset]}) Day {int(round(time_values[time_index] / 24.0))}", fontsize=14, pad=4)
        if offset == 2:
            axis.set_xlabel("Longitude", fontsize=12)
    figure.supxlabel("", y=0.01)
    omega_cbar_axis = figure.add_axes([0.915, 0.59, 0.020, 0.29])
    omega_cbar = figure.colorbar(omega_image, cax=omega_cbar_axis)
    omega_cbar.set_label("500-hPa ω (Pa s⁻¹)", fontsize=10, rotation=90, labelpad=8)
    omega_cbar.ax.tick_params(labelsize=9, length=3)
    temp_cbar_axis = figure.add_axes([0.915, 0.105, 0.020, 0.36])
    temp_cbar = figure.colorbar(temperature_image, cax=temp_cbar_axis)
    temp_cbar.set_label("1000-hPa temperature anomaly (K)", fontsize=11, rotation=90, labelpad=9)
    temp_cbar.ax.tick_params(labelsize=9, length=3)
    output = FIGURES / "Figure11_eddy_persistent_anticyclone.png"
    figure.savefig(output, dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    pd.DataFrame(
        {
            "panel": letters,
            "field": ["500-hPa omega"] * 6 + ["1000-hPa temperature anomaly"] * 3,
            "day": [7, 8, 9, 10, 11, 12, 10, 12, 14],
            "case": [case] * 9,
            "surface_pressure_field": ["eddy p_s*"] * 9,
            "pressure_contour_levels_hpa": ["-40,-30,-20,-10,10,20,30,40"] * 9,
            "high_center_label_method": ["largest connected-object centroid"] * 3 + ["regional eddy-pressure maxima"] * 6,
        }
    ).to_csv(TABLES / "Figure11_eddy_panel_metadata.csv", index=False)
    return output


def load_hovmoller(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(ROOT / f"{case}.nc", decode_times=False) as dataset:
        pressure = (dataset["PRESsfc"] / 100.0).isel(time=slice(200, None)).load()
        eddy_pressure = pressure - pressure.mean(dim="grid_xt")
        selected = eddy_pressure.sel(grid_yt=slice(0, 50))
        weights = np.cos(np.deg2rad(selected["grid_yt"]))
        weights = weights / weights.sum(dim="grid_yt")
        average = (selected * weights).sum(dim="grid_yt").transpose("time", "grid_xt")
        return average.values.astype(np.float32), average["grid_xt"].values.astype(float), dataset["time"].isel(time=slice(200, None)).values.astype(float)


def plot_figure12() -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(11.0, 8.8), sharex=True, sharey=True, facecolor="white")
    cmap, norm = make_white_center_cmap()
    image = None
    records = []
    letters = list("abcdefghi")
    for row, (b_file_value, b_display_value) in enumerate(zip(B_FILE_VALUES, B_DISPLAY_VALUES)):
        for column, n_value in enumerate(N_VALUES):
            axis = axes[row, column]
            case = case_name(b_file_value, n_value, 10)
            values, longitude, time = load_hovmoller(case)
            image = axis.contourf(
                longitude, time, values, levels=EDDY_HOV_LEVELS,
                cmap=cmap, norm=norm, extend="both",
            )
            axis.set_title(f"({letters[row * 3 + column]})  b={b_display_value:g}, n={n_value}", fontsize=11, fontweight="bold", pad=4)
            axis.set_xlim(0, 360)
            axis.set_ylim(200, 360)
            axis.set_xticks([0, 90, 180, 270, 360])
            axis.set_yticks([200, 250, 300, 350])
            axis.tick_params(labelsize=8)
            axis.grid(True, color="0.55", alpha=0.35, linewidth=0.45)
            for time_index, time_value in enumerate(time):
                for lon_index, lon_value in enumerate(longitude):
                    records.append({
                        "case": case,
                        "b": b_display_value,
                        "n": n_value,
                        "s": 10,
                        "time_h": time_value,
                        "longitude_deg_e": lon_value,
                        "eddy_pressure_anomaly_hpa": values[time_index, lon_index],
                    })
    for row, b_display_value in enumerate(B_DISPLAY_VALUES):
        axes[row, 0].text(
            -0.10, 0.50, f"b={b_display_value:g}",
            transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=9.5, fontweight="bold",
            clip_on=False,
        )
    figure.supxlabel("Longitude", fontsize=11, y=0.035)
    figure.supylabel("Time (h)", fontsize=11, x=0.025)
    colorbar_axis = figure.add_axes([0.91, 0.17, 0.022, 0.66])
    colorbar = figure.colorbar(image, cax=colorbar_axis, boundaries=EDDY_HOV_LEVELS, ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], spacing="proportional")
    colorbar.set_label("0–50°N mean eddy surface pressure anomaly (hPa)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.105, right=0.88, bottom=0.09, top=0.94, wspace=0.10, hspace=0.24)
    output = FIGURES / "Figure12_eddy_high_pressure_hovmoller_2hpa_pm10_blabels.png"
    figure.savefig(output, dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    pd.DataFrame.from_records(records).to_csv(TABLES / "Figure12_eddy_hovmoller_values.csv", index=False)
    return output


def parse_case_metadata(case: str) -> tuple[float, int, int]:
    match = re.fullmatch(r"BCwave_b(15|1|2)n([136])(?:s(-?10|-?5))?", case)
    if match is None:
        raise ValueError(case)
    b_value = 1.5 if match.group(1) == "15" else float(match.group(1))
    n_value = int(match.group(2))
    shift = int(match.group(3)) if match.group(3) else 0
    return b_value, n_value, shift


def process_persistence_case(case: str) -> dict[str, float | int | str]:
    with xr.open_dataset(ROOT / f"{case}.nc", decode_times=False) as dataset:
        pressure = (dataset["PRESsfc"] / 100.0).load().values.astype(np.float32)
        latitude_all = dataset["grid_yt"].values.astype(float)
        longitude = dataset["grid_xt"].values.astype(float)
        time = dataset["time"].values.astype(float)
    northern = latitude_all >= 0.0
    latitude = latitude_all[northern]
    eddy_pressure = pressure[:, northern, :] - pressure[:, northern, :].mean(axis=-1, keepdims=True)
    area = spherical_cell_areas_km2(latitude, longitude)
    objects_by_time = [
        objects_at_time(eddy_pressure[time_index], area, latitude, longitude, PRESSURE_TRACK_THRESHOLD_HPA)
        for time_index in range(eddy_pressure.shape[0])
    ]
    tracks = track_objects(objects_by_time, area, OVERLAP_THRESHOLD)
    durations = []
    qualifying_track_count = 0
    peak_values = []
    mean_areas = []
    max_areas = []
    for points in tracks.values():
        duration_hours = float(points[-1].time_index - points[0].time_index)
        if duration_hours >= MIN_PERSISTENCE_H:
            qualifying_track_count += 1
            durations.append(duration_hours)
            peak_values.append(max(point.obj.peak_anomaly_hpa for point in points))
            mean_areas.append(float(np.mean([point.obj.area_km2 for point in points])))
            max_areas.append(float(np.max([point.obj.area_km2 for point in points])))
    b_value, n_value, shift = parse_case_metadata(case)
    return {
        "case": case,
        "b": b_value,
        "n": n_value,
        "s": shift,
        "threshold_hpa": PRESSURE_TRACK_THRESHOLD_HPA,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "minimum_persistence_h": MIN_PERSISTENCE_H,
        "maximum_qualifying_persistence_h": max(durations, default=0.0),
        "number_qualifying_tracks": qualifying_track_count,
        "mean_qualifying_object_area_km2": float(np.mean(mean_areas)) if mean_areas else np.nan,
        "maximum_qualifying_object_area_km2": float(np.max(max_areas)) if max_areas else np.nan,
        "maximum_peak_eddy_pressure_anomaly_hpa": float(np.max(peak_values)) if peak_values else np.nan,
        "number_of_time_steps": len(time),
        "analysis_start_h": float(time[0]),
        "analysis_end_h": float(time[-1]),
    }


def plot_figure13(case_results: pd.DataFrame) -> Path:
    matrices = []
    for shift in S_VALUES:
        matrix = np.full((3, 3), np.nan)
        subset = case_results[case_results["s"] == shift]
        for row, b_value in enumerate([1.0, 1.5, 2.0]):
            for column, n_value in enumerate(N_VALUES):
                match = subset[(subset["b"] == b_value) & (subset["n"] == n_value)]
                matrix[row, column] = match["maximum_qualifying_persistence_h"].iloc[0] if not match.empty else np.nan
        matrices.append(matrix)
    finite_max = float(np.nanmax(np.concatenate([matrix.ravel() for matrix in matrices])))
    vmax = max(200.0, math.ceil(finite_max / 20.0) * 20.0)
    cmap = ListedColormap(["#fffdfb", "#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"])
    norm = BoundaryNorm(np.linspace(0, vmax, cmap.N + 1), cmap.N)
    figure, axes = plt.subplots(len(S_VALUES), 1, figsize=(6.6, 14.8), facecolor="white")
    axes = np.atleast_1d(axes)
    letters = list("abcde")
    image = None
    for panel_index, (axis, shift, matrix) in enumerate(zip(axes, S_VALUES, matrices)):
        image = axis.imshow(matrix, origin="lower", cmap=cmap, norm=norm, aspect="equal")
        axis.set_xticks(np.arange(3), [str(value) for value in N_VALUES])
        axis.set_yticks(np.arange(3), ["1", "1.5", "2"])
        axis.tick_params(labelsize=9)
        shift_label = "0°" if shift == 0 else f"{shift:+g}°"
        axis.set_title(f"({letters[panel_index]})  s = {shift_label}", fontsize=11, fontweight="bold", pad=5)
        axis.set_xlabel("Jet-width parameter n  (broader jet →)", fontsize=9)
        axis.set_ylabel("b", fontsize=9)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=9)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
    figure.suptitle("Eddy high-pressure persistence (constant-u₀ ensemble)", fontsize=13, y=0.995)
    figure.text(0.015, 0.50, "Maximum overlap-connected duration (h)", rotation=90, va="center", ha="center", fontsize=10)
    colorbar_axis = figure.add_axes([0.89, 0.16, 0.026, 0.68])
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=np.linspace(0, vmax, 6))
    colorbar.set_label("Duration (h)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.16, right=0.86, bottom=0.04, top=0.965, hspace=0.38)
    output = FIGURES / "Figure13_eddy_anticyclone_overlap_duration.png"
    figure.savefig(output, dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    heatmap_rows = []
    for shift, matrix in zip(S_VALUES, matrices):
        for row, b_value in enumerate([1.0, 1.5, 2.0]):
            for column, n_value in enumerate(N_VALUES):
                heatmap_rows.append({"s": shift, "b": b_value, "n": n_value, "maximum_qualifying_persistence_h": matrix[row, column]})
    pd.DataFrame(heatmap_rows).to_csv(TABLES / "Figure13_eddy_persistence_heatmap_values.csv", index=False)
    return output


def write_metadata() -> None:
    metadata = {
        "surface_pressure_definition": "p_s* = p_s - instantaneous zonal mean of p_s",
        "high_pressure_tracking_threshold_hpa": PRESSURE_TRACK_THRESHOLD_HPA,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "minimum_persistence_h": MIN_PERSISTENCE_H,
        "tracking_domain": "0-90 N",
        "longitude_connectivity": "cyclic",
        "object_connectivity": "8-neighbor",
        "figure11_case": "BCwave_b2n1s10",
        "figure11_top_days": [7, 8, 9, 10, 11, 12],
        "figure11_bottom_days": [10, 12, 14],
        "figure12_cases": [case_name(b, n, 10) for b in B_FILE_VALUES for n in N_VALUES],
        "figure12_latitude_average": "cosine-weighted 0-50 N",
        "figure12_time_window_h": [201, 360],
        "figure13_cases": 45,
        "note": "The eddy-field persistence values are not numerically comparable to full-field 1010-hPa values without accounting for the changed pressure reference.",
    }
    (TABLES / "anticyclone_eddyfield_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    ensure_directories()
    output = plot_figure12()
    metadata = {
        "source_diagnostic": "existing anticyclone eddy-field Hovmoller implementation",
        "surface_pressure_definition": "p_s* = p_s - instantaneous zonal mean of p_s",
        "latitude_average": "cosine-weighted 0-50 N",
        "time_window_h": [201, 360],
        "cases": [case_name(b, n, 10) for b in B_FILE_VALUES for n in N_VALUES],
        "contour_interval_hpa": 2.0,
        "white_band_hpa": [-2.0, 2.0],
        "levels_hpa": EDDY_HOV_LEVELS.tolist(),
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
    }
    (TABLES / "Figure12_eddy_hovmoller_2hpa_pm10_blabels_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
