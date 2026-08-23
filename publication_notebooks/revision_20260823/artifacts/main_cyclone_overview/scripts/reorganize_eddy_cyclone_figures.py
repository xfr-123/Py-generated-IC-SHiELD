#!/usr/bin/env python3
"""Reorganize the eddy-field cyclone-interaction diagnostics.

This script leaves the existing Figure 7 and Figure 8 files untouched. It
replots their underlying diagnostics in two new figures:

1. an overview-plus-zoom composite containing the 3 x 3 overview and the
   three-panel b2n3s10 interaction sequence; and
2. a separate three-panel wind-speed time-series figure.

The pressure diagnostic is the eddy surface pressure p_s - [p_s], and all
event labels are taken from the existing eddy-field event tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

ROOT = Path("/data/keeling/a/mingfei5/a/data/original")
SOURCE_FIGURE_DIR = ROOT / "paper_revision/eddy_field_replot_20260818/figures"
SOURCE_TABLE_DIR = ROOT / "paper_revision/eddy_field_replot_20260818/tables"
EVENTS_STANDARD = ROOT / "eddy/eddy_full_coalescence_review_analysis_20260723/case_results"
OUTPUT = ROOT / "paper_revision/cyclone_interaction_reorganization_20260820"
FIGURES = OUTPUT / "figures"
METADATA = OUTPUT / "metadata"

OMEGA_LEVELS = np.array([-0.24, -0.20, -0.16, -0.08, -0.04, 0.04, 0.08, 0.16, 0.20, 0.24])
OMEGA_TICKS = [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24]
EDDY_LEVELS = np.arange(-80.0, 0.0, 10.0)
EDDY_CONNECTION_LEVEL = -10.0
SPEED_LEVELS = np.arange(0.0, 60.1, 2.5)
LATITUDE_MIN = 30.0
LOCAL_LON_MIN = 80.0
LOCAL_LON_MAX = 200.0
LOCAL_LAT_MIN = 55.0
LOCAL_LAT_MAX = 80.0
COLORS_B = {1.0: "#0072B2", 1.5: "#E69F00", 2.0: "#6A3D9A"}


def open_case(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=False, engine="netcdf4")


def get_lat_lon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    return ds["grid_yt"].values.astype(float), ds["grid_xt"].values.astype(float)


def eddy_slp_hpa(ds: xr.Dataset, time_index: int) -> np.ndarray:
    pressure = ds["PRESsfc"].isel(time=time_index).values.astype(float) / 100.0
    return pressure - np.nanmean(pressure, axis=-1, keepdims=True)


def read_events(case: str) -> pd.DataFrame:
    path = EVENTS_STANDARD / f"{case}_events.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    threshold = pd.to_numeric(table["threshold_hpa"], errors="coerce")
    return table[(table["mode"] == "eddy") & np.isclose(threshold, -10.0)].copy()


def nearest_event_row(event_table: pd.DataFrame, time_index: int, min_area: float = 1.0e6):
    if event_table.empty:
        return None
    table = event_table.copy()
    large = table[table["child_area_km2"] >= min_area]
    if not large.empty:
        table = large
    table["distance"] = np.abs(table["time_index"].astype(float) - float(time_index))
    return table.sort_values(["distance", "time_index"]).iloc[0]


def annotate_event(axis, row, show_secondary: bool = True) -> None:
    if row is None:
        return
    labels = [("P", "parent_1_centroid_lat", "parent_1_centroid_lon", "#b2182b")]
    if show_secondary:
        labels.append(("S", "parent_2_centroid_lat", "parent_2_centroid_lon", "#2166ac"))
    for label, lat_name, lon_name, color in labels:
        lat = float(row.get(lat_name, np.nan))
        lon = float(row.get(lon_name, np.nan))
        if not np.isfinite(lat) or not np.isfinite(lon) or lat < LATITUDE_MIN:
            continue
        axis.text(
            lon,
            lat,
            label,
            transform=ccrs.PlateCarree(),
            color=color,
            fontsize=12.0,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=20,
        )


def polar_boundary(axis) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    vertices = np.column_stack([0.5 + 0.5 * np.cos(theta), 0.5 + 0.5 * np.sin(theta)])
    codes = np.full(len(theta), MplPath.LINETO, dtype=MplPath.code_type)
    codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
    axis.set_boundary(MplPath(vertices, codes), transform=axis.transAxes)


def lon_formatter(column: int):
    def formatter(value, position):
        if value == -90 and column == 0:
            return "90°W"
        if value == 0:
            return "0°"
        if value == 90 and column == 2:
            return "90°E"
        if value == 180:
            return "180°"
        return ""

    return formatter


def rounded_geographic_box(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    radius: float = 4.0, points_per_arc: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth rounded rectangle in longitude-latitude coordinates."""
    radius = min(radius, 0.24 * (lon_max - lon_min), 0.24 * (lat_max - lat_min))
    pieces = [
        np.column_stack([np.linspace(lon_min + radius, lon_max - radius, 30), np.full(30, lat_min)]),
    ]
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


def configure_polar_axis(axis, column: int, grid_label_size: float = 7.6, show_labels: bool = True) -> None:
    axis.set_extent([-180, 180, LATITUDE_MIN, 90], ccrs.PlateCarree())
    polar_boundary(axis)
    gridlines = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linestyle="--",
        linewidth=0.42,
        color="0.55",
        alpha=0.58,
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
    )
    gridlines.xlocator = mticker.FixedLocator([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    if show_labels:
        gridlines.xformatter = FuncFormatter(lon_formatter(column))
    else:
        gridlines.xformatter = FuncFormatter(lambda value, position: "")
    gridlines.xlabels_top = show_labels
    gridlines.xlabels_bottom = False
    gridlines.ylabels_left = False
    gridlines.ylabels_right = False
    # Cartopy 0.25 uses the explicit top/bottom label properties.
    gridlines.top_labels = show_labels
    gridlines.bottom_labels = False
    gridlines.left_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": grid_label_size}


def plot_eddy_pressure(axis, longitude, latitude, eddy, linewidth: float = 0.75) -> None:
    axis.contour(
        longitude,
        latitude,
        eddy,
        transform=ccrs.PlateCarree(),
        levels=EDDY_LEVELS,
        colors="black",
        linewidths=linewidth,
        zorder=8,
    )
    axis.contour(
        longitude,
        latitude,
        eddy,
        transform=ccrs.PlateCarree(),
        levels=[EDDY_CONNECTION_LEVEL],
        colors="#1464a0",
        linewidths=2.0,
        zorder=9,
    )


def plot_local_eddy_pressure(axis, longitude, latitude, eddy, linewidth: float = 0.8) -> None:
    axis.contour(
        longitude, latitude, eddy, levels=EDDY_LEVELS, colors="black",
        linewidths=linewidth, zorder=8,
    )
    axis.contour(
        longitude, latitude, eddy, levels=[EDDY_CONNECTION_LEVEL],
        colors="#1464a0", linewidths=2.0, zorder=9,
    )


def panel_label(axis, label: str) -> None:
    axis.text(
        0.035,
        0.965,
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.3},
        zorder=30,
    )


def draw_overview_maps(figure, grid) -> None:
    scenarios = [
        ("BCwave_b2n3", r"$b=2,\,n=3,\,s=0^\circ$"),
        ("BCwave_b2n3s10", r"$b=2,\,n=3,\,s=+10^\circ$"),
    ]
    days = [10, 12, 14]
    letters = "abcdef"
    axes = []
    for row_index, (case, case_label) in enumerate(scenarios):
        event_table = read_events(case)
        with open_case(ROOT / f"{case}.nc") as ds:
            latitude, longitude = get_lat_lon(ds)
            for column_index, day in enumerate(days):
                axis = figure.add_subplot(
                    grid[row_index, column_index],
                    projection=ccrs.NorthPolarStereo(central_longitude=0),
                )
                axes.append(axis)
                configure_polar_axis(axis, column_index, show_labels=False)
                time_index = day * 24
                omega = ds["omg_plev"].sel(plev=500).isel(time=time_index).values.astype(float)
                omega[np.abs(omega) < 0.08] = 0.0
                eddy = eddy_slp_hpa(ds, time_index)
                axis.contourf(
                    longitude,
                    latitude,
                    omega,
                    transform=ccrs.PlateCarree(),
                    levels=OMEGA_LEVELS,
                    cmap="RdBu_r",
                    extend="both",
                    alpha=0.70,
                    zorder=1,
                )
                plot_eddy_pressure(axis, longitude, latitude, eddy)
                if row_index == 1 and column_index == 0:
                    box_lon, box_lat = rounded_geographic_box(
                        LOCAL_LON_MIN, LOCAL_LON_MAX, LOCAL_LAT_MIN, LOCAL_LAT_MAX, radius=4.0,
                    )
                    axis.plot(
                        box_lon, box_lat, transform=ccrs.PlateCarree(), color="black",
                        linewidth=1.8, solid_capstyle="round", solid_joinstyle="round", zorder=12,
                    )
                axis.set_title(
                    f"({letters[row_index * 3 + column_index]}) Day {day}",
                    fontsize=10.2, pad=4.0, fontweight="semibold",
                )
    return axes, [case_label for _, case_label in scenarios]


def draw_zoom_maps(figure, grid):
    case = "BCwave_b2n3s10"
    days = [9, 10, 11]
    letters = "ghi"
    axes = []
    with open_case(ROOT / f"{case}.nc") as ds:
        latitude, longitude = get_lat_lon(ds)
        latitude_mask = (latitude >= LOCAL_LAT_MIN) & (latitude <= LOCAL_LAT_MAX)
        longitude_mask = (longitude >= LOCAL_LON_MIN) & (longitude <= LOCAL_LON_MAX)
        latitude_subset = latitude[latitude_mask]
        longitude_subset = longitude[longitude_mask]
        for offset, day in enumerate(days):
            axis = figure.add_subplot(grid[4 + offset, :])
            axes.append(axis)
            time_index = day * 24
            speed = np.hypot(
                ds["u_plev"].sel(plev=1000).isel(time=time_index).values.astype(float),
                ds["v_plev"].sel(plev=1000).isel(time=time_index).values.astype(float),
            )
            speed_subset = speed[latitude_mask][:, longitude_mask]
            eddy_subset = eddy_slp_hpa(ds, time_index)[latitude_mask][:, longitude_mask]
            axis.set_box_aspect(0.30)
            axis.contourf(
                longitude_subset, latitude_subset, speed_subset,
                levels=SPEED_LEVELS, cmap="Reds", extend="max", zorder=1,
            )
            plot_local_eddy_pressure(axis, longitude_subset, latitude_subset, eddy_subset, linewidth=0.8)
            axis.set_xlim(LOCAL_LON_MIN, LOCAL_LON_MAX)
            axis.set_ylim(LOCAL_LAT_MIN, LOCAL_LAT_MAX)
            axis.set_xticks([80, 100, 120, 140, 160, 180, 200])
            axis.set_yticks([55, 65, 75, 80])
            axis.tick_params(axis="both", labelsize=8.0, length=3.0, width=0.75)
            axis.grid(True, color="0.55", alpha=0.32, linewidth=0.5)
            axis.set_title(
                f"({letters[offset]}) Day {day}",
                fontsize=10.5, pad=5.0, fontweight="semibold",
            )
            axis.set_ylabel("Latitude", fontsize=9.2, labelpad=3)
            if offset == len(days) - 1:
                axis.set_xlabel("Longitude", fontsize=9.2, labelpad=3)
            else:
                axis.tick_params(axis="x", labelbottom=False)
    return axes


def add_colorbars(figure, grid):
    omega_mapper = ScalarMappable(norm=mcolors.Normalize(vmin=-0.24, vmax=0.24), cmap=plt.get_cmap("RdBu_r"))
    omega_mapper.set_array([])
    omega_cax = figure.add_subplot(grid[2, :])
    omega_cbar = figure.colorbar(omega_mapper, cax=omega_cax, orientation="horizontal", ticks=OMEGA_TICKS)
    omega_cbar.set_label("500-hPa vertical velocity (Pa s$^{-1}$)", fontsize=9.3, labelpad=2)
    omega_cbar.ax.tick_params(labelsize=7.6, length=2.5)

    speed_mapper = ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=60.0), cmap=plt.get_cmap("Reds"))
    speed_mapper.set_array([])
    speed_cax = figure.add_subplot(grid[7, :])
    speed_cbar = figure.colorbar(speed_mapper, cax=speed_cax, orientation="horizontal", ticks=[0, 15, 30, 45, 60])
    speed_cbar.set_label("1000-hPa wind speed (m s$^{-1}$)", fontsize=9.3, labelpad=2)
    speed_cbar.ax.tick_params(labelsize=7.6, length=2.5)


def build_overview_zoom_figure() -> Path:
    figure = plt.figure(figsize=(7.35, 12.25), dpi=300, facecolor="white")
    grid = figure.add_gridspec(
        8,
        3,
        height_ratios=[1.0, 1.0, 0.075, 0.08, 0.65, 0.65, 0.65, 0.075],
        left=0.105,
        right=0.985,
        top=0.985,
        bottom=0.045,
        wspace=0.025,
        hspace=0.16,
    )
    overview_axes, overview_labels = draw_overview_maps(figure, grid)
    zoom_axes = draw_zoom_maps(figure, grid)
    add_colorbars(figure, grid)
    figure.canvas.draw()
    for row_index, label in enumerate(overview_labels):
        row_axes = overview_axes[row_index * 3:(row_index + 1) * 3]
        row_position = row_axes[0].get_position()
        figure.text(
            0.052,
            0.5 * (row_position.y0 + row_position.y1),
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.4,
            fontweight="semibold",
        )
    output_png = FIGURES / "Figure9_eddy_cyclone_overview_plus_zoom.png"
    output_pdf = FIGURES / "Figure9_eddy_cyclone_overview_plus_zoom.pdf"
    figure.savefig(output_png, dpi=300, facecolor="white")
    figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)
    return output_png


def summarize_series(series: pd.DataFrame):
    edges = np.arange(-30.0, 30.01, 6.0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    arrays = []
    for _, group in series.groupby("case"):
        values = np.full(centers.shape, np.nan)
        for index in range(len(centers)):
            selected = group[(group["relative_hour"] >= edges[index]) & (group["relative_hour"] < edges[index + 1])]
            if not selected.empty:
                values[index] = selected["wind_speed_m_s"].mean()
        arrays.append(values)
    array = np.vstack(arrays)
    return centers, np.nanmean(array, axis=0), np.nanstd(array, axis=0, ddof=1)


def draw_group(axis, data: pd.DataFrame, key: str, values, colors, labels):
    for value in values:
        subset = data[np.isclose(data[key].astype(float), value)]
        if subset.empty:
            continue
        centers, mean, standard_deviation = summarize_series(subset)
        finite = np.isfinite(mean)
        axis.fill_between(
            centers[finite],
            (mean - 0.5 * standard_deviation)[finite],
            (mean + 0.5 * standard_deviation)[finite],
            color=colors[value],
            alpha=0.19,
            linewidth=0,
        )
        axis.plot(centers[finite], mean[finite], color=colors[value], linewidth=2.2, label=labels[value])
        axis.hlines(np.nanmean(mean), -24, 24, color=colors[value], linestyle="--", linewidth=1.1, alpha=0.9)


def build_time_series_figure() -> Path:
    data_path = SOURCE_TABLE_DIR / "Figure8_eddy_wind_speed_series_standard.csv"
    data = pd.read_csv(data_path)
    figure = plt.figure(figsize=(6.10, 7.80), dpi=300, facecolor="white")
    grid = figure.add_gridspec(3, 1, left=0.165, right=0.975, top=0.965, bottom=0.090, hspace=0.30)
    axes = [figure.add_subplot(grid[index, 0]) for index in range(3)]
    specs = [
        ("s", [-10, 0, 10], {-10: "#7f7f7f", 0: "#ff7f0e", 10: "#9467bd"}, {-10: "s = −10°", 0: "s = 0°", 10: "s = +10°"}, "Jet-latitude shift"),
        ("n", [1, 3, 6], {1: "#7f7f7f", 3: "#ff7f0e", 6: "#9467bd"}, {1: "n = 1", 3: "n = 3", 6: "n = 6"}, "Jet-width parameter"),
        ("b", [1.0, 1.5, 2.0], COLORS_B, {1.0: "b = 1", 1.5: "b = 1.5", 2.0: "b = 2"}, "Vertical-profile parameter"),
    ]
    letters = "abc"
    for index, (axis, spec) in enumerate(zip(axes, specs)):
        key, values, colors, labels, title = spec
        draw_group(axis, data, key, values, colors, labels)
        axis.axvline(0, color="black", linewidth=1.2)
        axis.set_xlim(-24, 24)
        axis.set_xticks([-24, -12, 0, 12, 24])
        axis.set_ylim(28, 57)
        axis.set_yticks([30, 35, 40, 45, 50, 55])
        axis.set_ylabel("1000-hPa wind speed\n(m s$^{-1}$)", fontsize=10.1)
        axis.tick_params(axis="both", labelsize=8.8, width=0.8, length=3.2)
        axis.grid(True, color="0.82", linewidth=0.55, alpha=0.75)
        axis.set_title(title, fontsize=10.8, loc="left", pad=5, fontweight="semibold")
        axis.text(0.012, 0.94, f"({letters[index]})", transform=axis.transAxes, fontsize=11.3, fontweight="bold", va="top", ha="left", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2})
        legend_handles, legend_labels = axis.get_legend_handles_labels()
        axis.legend(
            legend_handles,
            legend_labels,
            loc="lower left",
            bbox_to_anchor=(0.012, 0.018),
            ncol=3,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            fontsize=7.7,
            handlelength=1.8,
            columnspacing=0.9,
            handletextpad=0.45,
            borderpad=0.25,
        )
        if index < 2:
            axis.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Relative time since first eddy-contour connection (h)", fontsize=10.4, labelpad=4)
    output_png = FIGURES / "Figure10_eddy_cyclone_wind_evolution_timeseries.png"
    output_pdf = FIGURES / "Figure10_eddy_cyclone_wind_evolution_timeseries.pdf"
    figure.savefig(output_png, dpi=300, facecolor="white")
    figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)
    return output_png


def write_metadata(overview: Path, time_series: Path) -> None:
    metadata = {
        "interpretation": "Figure 7 overview plus Figure 8 left-side b2n3s10 zoom; Figure 8 right-side time series is separate",
        "source_figure_directory": str(SOURCE_FIGURE_DIR),
        "source_table_directory": str(SOURCE_TABLE_DIR),
        "overview_zoom_png": str(overview),
        "overview_zoom_pdf": str(overview.with_suffix(".pdf")),
        "time_series_png": str(time_series),
        "time_series_pdf": str(time_series.with_suffix(".pdf")),
        "overview_cases": ["BCwave_b2n3", "BCwave_b2n3s10"],
        "overview_days": [10, 12, 14],
        "zoom_case": "BCwave_b2n3s10",
        "zoom_days": [9, 10, 11],
        "zoom_bounds": "80-200E, 55-80N",
        "zoom_projection": "three stacked rectangular longitude-latitude axes spanning the figure width",
        "overview_zoom_box": "80-200E, 55-80N on panel (d) only",
        "field": "eddy surface pressure p_s - [p_s]",
        "time_series_source": "Figure8_eddy_wind_speed_series_standard.csv",
        "panel_scheme": {"overview": "(a)-(f)", "zoom": "(g)-(i)", "time_series": "(a)-(c)"},
        "figure9_p_s_annotations": "omitted for manual annotation",
        "figure9_in_figure_subtitles": False,
    }
    (METADATA / "reorganization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Eddy-field cyclone-interaction reorganization\n\n"
        "The combined figure uses the b2n3 and b2n3s10 eddy-field polar maps as the overall evolution and a rectangular 80-200E, 55-80N b2n3s10 local sequence matching the Figure 10 left-panel style. The Figure 8 right-side grouped wind-speed time series is provided as a separate figure. P/S labels and in-figure section subtitles are intentionally omitted for manual annotation. Existing figures and documents are not overwritten.\n\n"
        "All map fields are replotted from the original NetCDF variables using the existing eddy surface-pressure definition and event annotation tables. The time-series values are read from the existing Figure 8 eddy-field table and summarized with its six-hour bins and ±0.5 standard-deviation envelopes.\n",
        encoding="utf-8",
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    overview = build_overview_zoom_figure()
    time_series = build_time_series_figure()
    write_metadata(overview, time_series)
    print(overview)
    print(overview.with_suffix(".pdf"))
    print(time_series)
    print(time_series.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
