#!/usr/bin/env python3
"""Publication-quality Figure 11 revision using the existing eddy diagnostics."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path("/data/keeling/a/mingfei5/a/data/original")
OUT = ROOT / "paper_revision" / "anticyclone_figure11_eddyfield_replot_20260819_v7"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
SOURCE = OUT / "source"

HELPER_PATH = SOURCE / "legacy_figure11_helpers.py"
spec = importlib.util.spec_from_file_location("legacy_figure11_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
assert spec.loader is not None
spec.loader.exec_module(helpers)

ccrs = helpers.ccrs
OMEGA_LEVELS = helpers.OMEGA_LEVELS
EDDY_CONTOUR_LEVELS = helpers.EDDY_CONTOUR_LEVELS
TEMP_LEVELS = helpers.TEMP_LEVELS
PRESSURE_TRACK_THRESHOLD_HPA = helpers.PRESSURE_TRACK_THRESHOLD_HPA


def rounded_geographic_box(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    radius: float = 4.0, points_per_arc: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth rounded rectangle in lon-lat coordinates."""
    radius = min(radius, 0.24 * (lon_max - lon_min), 0.24 * (lat_max - lat_min))
    pieces: list[np.ndarray] = []
    pieces.append(np.column_stack([np.linspace(lon_min + radius, lon_max - radius, 30), np.full(30, lat_min)]))
    theta = np.linspace(-90.0, 0.0, points_per_arc)
    pieces.append(np.column_stack([
        lon_max - radius + radius * np.cos(np.deg2rad(theta)),
        lat_min + radius + radius * np.sin(np.deg2rad(theta)),
    ]))
    pieces.append(np.column_stack([np.full(30, lon_max), np.linspace(lat_min + radius, lat_max - radius, 30)]))
    theta = np.linspace(0.0, 90.0, points_per_arc)
    pieces.append(np.column_stack([
        lon_max - radius + radius * np.cos(np.deg2rad(theta)),
        lat_max - radius + radius * np.sin(np.deg2rad(theta)),
    ]))
    pieces.append(np.column_stack([np.linspace(lon_max - radius, lon_min + radius, 30), np.full(30, lat_max)]))
    theta = np.linspace(90.0, 180.0, points_per_arc)
    pieces.append(np.column_stack([
        lon_min + radius + radius * np.cos(np.deg2rad(theta)),
        lat_max - radius + radius * np.sin(np.deg2rad(theta)),
    ]))
    pieces.append(np.column_stack([np.full(30, lon_min), np.linspace(lat_max - radius, lat_min + radius, 30)]))
    theta = np.linspace(180.0, 270.0, points_per_arc)
    pieces.append(np.column_stack([
        lon_min + radius + radius * np.cos(np.deg2rad(theta)),
        lat_min + radius + radius * np.sin(np.deg2rad(theta)),
    ]))
    points = np.vstack(pieces)
    points = np.vstack([points, points[0]])
    return points[:, 0], points[:, 1]


def find_regional_high_centers(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
    count: int = 2,
) -> list[tuple[float, float]]:
    """Find peak positions of the largest connected positive eddy-pressure regions."""
    lat_mask = (latitude >= latitude_bounds[0]) & (latitude <= latitude_bounds[1])
    lon_mask = (longitude >= longitude_bounds[0]) & (longitude <= longitude_bounds[1])
    subset = np.asarray(field[np.ix_(lat_mask, lon_mask)], dtype=float)
    if subset.size == 0 or not np.isfinite(subset).any():
        return []
    mask = np.isfinite(subset) & (subset >= PRESSURE_TRACK_THRESHOLD_HPA)
    labels, nlabels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    candidates: list[tuple[int, float, float, float]] = []
    for label_id in range(1, nlabels + 1):
        pixels = np.flatnonzero(labels.ravel() == label_id)
        if pixels.size < 4:
            continue
        peak_position = pixels[int(np.nanargmax(subset.ravel()[pixels]))]
        row, column = np.unravel_index(peak_position, subset.shape)
        candidates.append((
            int(pixels.size),
            float(subset.ravel()[peak_position]),
            float(latitude[lat_mask][row]),
            float(longitude[lon_mask][column]),
        ))
    candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return [(row[2], row[3]) for row in candidates[:count]]

def label_high_center(axis, longitude: float, latitude: float, fontsize: float = 20.0, geographic: bool = True) -> None:
    kwargs = {"transform": ccrs.PlateCarree()} if geographic else {}
    label = axis.text(
        longitude, latitude, "H", **kwargs,
        color="#d7191c", fontsize=fontsize, fontweight="bold",
        ha="center", va="center", zorder=12,
    )
    label.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white")])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    case = "BCwave_b2n1s10"
    path = ROOT / f"{case}.nc"
    with xr.open_dataset(path, decode_times=False) as dataset:
        pressure_eddy, latitude, longitude, _ = helpers.eddy_pressure_from_dataset(dataset)
        omega = dataset["omg_plev"].sel(plev=500).isel(time=[168, 192, 216, 240, 264, 288]).load().values
        temperature = dataset["t_plev"].sel(plev=1000).isel(time=[240, 288, 336]).load().values
        time_values = dataset["time"].values.astype(float)

    top_indices = [168, 192, 216, 240, 264, 288]
    bottom_indices = [240, 288, 336]
    figure = plt.figure(figsize=(13.4, 19.2), facecolor="white")
    grid = figure.add_gridspec(
        6, 3,
        height_ratios=[1.04, 1.04, 0.0, 0.86, 0.86, 0.86],
        left=0.045, right=0.875, bottom=0.035, top=0.955,
        wspace=0.23, hspace=0.18,
    )
    letters = list("abcdefghi")
    omega_image = None
    panel_metadata: list[dict[str, object]] = []

    for panel_index, (time_index, omega_field) in enumerate(zip(top_indices, omega)):
        row, column = divmod(panel_index, 3)
        axis = figure.add_subplot(grid[row, column], projection=ccrs.NorthPolarStereo(central_longitude=0))
        axis.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
        helpers.polar_boundary(axis)
        helpers.add_polar_grid(axis, column)
        axis.gridlines().xlabel_style = {"size": 11}
        omega_plot = np.where(np.abs(omega_field) >= 0.08, omega_field, 0.0)
        omega_image = axis.contourf(
            longitude, latitude, omega_plot, levels=OMEGA_LEVELS,
            cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree(), alpha=0.68,
        )
        pressure_field = pressure_eddy[time_index]
        axis.contour(
            longitude, latitude, pressure_field, levels=EDDY_CONTOUR_LEVELS,
            colors="black", linewidths=1.02, transform=ccrs.PlateCarree(),
        )
        if panel_index >= 3:
            centers = [
                helpers.find_regional_peak(pressure_field, latitude, longitude, (35.0, 75.0), (180.0, 270.0)),
                helpers.find_regional_peak(pressure_field, latitude, longitude, (35.0, 75.0), (90.0, 180.0)),
            ]
            centers = [center for center in centers if center is not None]
            center_pairs = [(lat, lon) for lat, lon in centers]
        else:
            center_pairs = helpers.find_high_centers(pressure_field, latitude, longitude)
        for center_lat, center_lon in center_pairs:
            label_high_center(axis, center_lon, center_lat, fontsize=23)
        if panel_index in (3, 5):
            box_lon, box_lat = rounded_geographic_box(100.0, 250.0, 35.0, 65.0, radius=4.0)
            axis.plot(
                box_lon, box_lat, transform=ccrs.PlateCarree(), color="black",
                linewidth=2.15, solid_capstyle="round", solid_joinstyle="round", zorder=10,
            )
        day = int(round(time_values[time_index] / 24.0))
        axis.set_title(f"({letters[panel_index]}) Day {day}", fontsize=17, pad=11, fontweight="semibold")
        panel_metadata.append({
            "panel": letters[panel_index], "day": day, "field": "500-hPa omega",
            "high_center_count": len(center_pairs),
            "box_bounds": "100-250E, 35-65N" if panel_index in (3, 5) else "",
        })

    temperature_image = None
    latitude_mask = (latitude >= 35.0) & (latitude <= 65.0)
    longitude_mask = (longitude >= 100.0) & (longitude <= 250.0)
    for offset, (time_index, temperature_field) in enumerate(zip(bottom_indices, temperature)):
        axis = figure.add_subplot(grid[3 + offset, :])
        axis.set_box_aspect(0.24)
        temperature_subset = temperature_field[latitude_mask]
        temperature_anomaly = temperature_subset - temperature_subset.mean(axis=-1, keepdims=True)
        longitude_subset = longitude[longitude_mask]
        temperature_anomaly = temperature_anomaly[:, longitude_mask]
        temperature_image = axis.contourf(
            longitude_subset, latitude[latitude_mask], temperature_anomaly,
            levels=TEMP_LEVELS, cmap="RdBu_r", extend="both",
        )
        pressure_subset = pressure_eddy[time_index][latitude_mask][:, longitude_mask]
        negative_levels = EDDY_CONTOUR_LEVELS[EDDY_CONTOUR_LEVELS < 0]
        positive_levels = EDDY_CONTOUR_LEVELS[EDDY_CONTOUR_LEVELS > 0]
        axis.contour(longitude_subset, latitude[latitude_mask], pressure_subset, levels=positive_levels, colors="black", linewidths=1.0)
        negative_contours = axis.contour(longitude_subset, latitude[latitude_mask], pressure_subset, levels=negative_levels, colors="black", linewidths=0.86, linestyles="dashed")
        axis.clabel(negative_contours, inline=True, fontsize=10.5, fmt="%g")
        centers = find_regional_high_centers(
            pressure_eddy[time_index], latitude, longitude,
            (35.0, 65.0), (100.0, 250.0), count=2,
        )
        for center_lat, center_lon in centers:
            label_high_center(axis, center_lon, center_lat, fontsize=20, geographic=False)
        axis.set_xlim(100, 250)
        axis.set_ylim(35, 65)
        axis.set_yticks([35, 45, 55, 65])
        axis.tick_params(labelsize=13, length=4, width=0.9)
        axis.grid(True, color="0.55", alpha=0.32, linewidth=0.55)
        axis.set_ylabel("Latitude", fontsize=15, labelpad=7)
        axis.set_title(f"({letters[6 + offset]}) Day {int(round(time_values[time_index] / 24.0))}", fontsize=17, pad=7, fontweight="semibold")
        if offset == 2:
            axis.set_xlabel("Longitude", fontsize=15, labelpad=6)
        panel_metadata.append({
            "panel": letters[6 + offset], "day": int(round(time_values[time_index] / 24.0)),
            "field": "1000-hPa temperature anomaly", "high_center_count": len(centers),
            "box_bounds": "100-250E, 35-65N",
        })

    omega_cbar_axis = figure.add_axes([0.895, 0.585, 0.019, 0.325])
    omega_cbar = figure.colorbar(omega_image, cax=omega_cbar_axis)
    omega_cbar.set_label("500-hPa $\u03c9$ (Pa s$^{-1}$)", fontsize=12, rotation=90, labelpad=10)
    omega_cbar.ax.tick_params(labelsize=10, length=3)
    temp_cbar_axis = figure.add_axes([0.895, 0.075, 0.019, 0.405])
    temp_cbar = figure.colorbar(temperature_image, cax=temp_cbar_axis)
    temp_cbar.set_label("1000-hPa temperature anomaly (K)", fontsize=12, rotation=90, labelpad=10)
    temp_cbar.ax.tick_params(labelsize=10, length=3)

    output = FIGURES / "Figure11_eddy_persistent_anticyclone.png"
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)

    pd.DataFrame(panel_metadata).to_csv(TABLES / "Figure11_eddy_panel_metadata.csv", index=False)
    (TABLES / "Figure11_qc_summary.json").write_text(json.dumps({
        "case": case,
        "box_bounds": "100-250E, 35-65N",
        "box_style": "smooth rounded geographic path, radius 4 degrees",
        "bottom_high_center_method": "peak grid points of two largest connected regions above +10 hPa in 35-65N, 100-250E",
        "bottom_high_center_counts": [row["high_center_count"] for row in panel_metadata[-3:]],
        "figure_size_inches": [13.4, 19.2],
        "output_png": str(output),
        "output_pdf": str(output.with_suffix('.pdf')),
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
