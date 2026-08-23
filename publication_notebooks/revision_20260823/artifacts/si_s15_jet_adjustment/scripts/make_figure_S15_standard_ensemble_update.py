#!/usr/bin/env python3
"""Build revised Supplementary Figure S15 for the 45-case standard ensemble."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_PACKAGE = ROOT / "standard_jet_baroclinicity_analysis_20260813.zip"
SOURCE_PREFIX = "paper_revision/supplemental_analysis/standard_jet_baroclinicity_evolution"

HOURLY_MEMBER = f"{SOURCE_PREFIX}/standard_jet_baroclinicity_hourly_all45.csv"
CASE_SUMMARY_MEMBER = f"{SOURCE_PREFIX}/standard_jet_baroclinicity_case_summary_all45.csv"
GROUP_SUMMARY_MEMBER = f"{SOURCE_PREFIX}/standard_jet_baroclinicity_group_summary_b_n_s.csv"
METHODS_MEMBER = f"{SOURCE_PREFIX}/standard_jet_baroclinicity_methods.md"

PNG_PATH = OUTPUT_DIR / "Figure_S15_standard_ensemble_jet_adjustment.png"
PDF_PATH = OUTPUT_DIR / "Figure_S15_standard_ensemble_jet_adjustment.pdf"
TABLE_PATH = OUTPUT_DIR / "Figure_S15_plotted_values_all45.csv"
REPORT_PATH = OUTPUT_DIR / "Figure_S15_methods_verification_caption_response.md"
MANIFEST_PATH = OUTPUT_DIR / "SHA256SUMS.txt"

REPRESENTATIVE_CASES = [
    ("BCwave_b2n3s-10", -10),
    ("BCwave_b2n3", 0),
    ("BCwave_b2n3s10", 10),
]

B_COLORS = {1.0: "#0072B2", 1.5: "#E69F00", 2.0: "#6A3D9A"}
N_MARKERS = {1: "o", 3: "s", 6: "^"}
N_OFFSETS = {1: -0.58, 3: 0.0, 6: 0.58}

DELTA_LEVELS = np.arange(-10.0, 10.1, 1.0)
DELTA_TICKS = [-10, -5, 0, 5, 10]
PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200, 100]
LATITUDE_TICKS = [20, 30, 40, 50, 60, 70, 80]


def read_package_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as stream:
        return pd.read_csv(stream)


def read_package_text(archive: zipfile.ZipFile, member: str) -> str:
    with archive.open(member) as stream:
        return stream.read().decode("utf-8")


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - target)))


def read_representative_case(case: str) -> dict[str, np.ndarray]:
    path = ROOT / f"{case}.nc"
    with Dataset(path, mode="r") as dataset:
        latitude = np.asarray(dataset.variables["grid_yt"][:], dtype=np.float64)
        pressure = np.asarray(dataset.variables["plev"][:], dtype=np.float64)
        time_hours = np.asarray(dataset.variables["time"][:], dtype=np.float64)
        k300 = nearest_index(pressure, 300.0)
        zonal_mean_initial = np.nanmean(
            np.ma.filled(dataset.variables["u_plev"][0, :, :, :], np.nan),
            axis=-1,
        )
        zonal_mean_day15 = np.nanmean(
            np.ma.filled(dataset.variables["u_plev"][-1, :, :, :], np.nan),
            axis=-1,
        )
        zonal_mean_u300 = np.nanmean(
            np.ma.filled(dataset.variables["u_plev"][:, k300, :, :], np.nan),
            axis=-1,
        )

    latitude_order = np.argsort(latitude)
    latitude = latitude[latitude_order]
    zonal_mean_initial = zonal_mean_initial[:, latitude_order]
    zonal_mean_day15 = zonal_mean_day15[:, latitude_order]
    zonal_mean_u300 = zonal_mean_u300[:, latitude_order]
    latitude_mask = (latitude >= 20.0) & (latitude <= 80.0)
    latitude_indices = np.flatnonzero(latitude_mask)
    maximum_indices = np.nanargmax(zonal_mean_u300[:, latitude_mask], axis=1)
    jet_latitudes = latitude[latitude_indices[maximum_indices]]

    return {
        "latitude": latitude,
        "pressure": pressure,
        "days": time_hours / 24.0,
        "initial": zonal_mean_initial,
        "day15": zonal_mean_day15,
        "u300": zonal_mean_u300,
        "jet_latitudes": jet_latitudes,
    }


def snapshot(hourly: pd.DataFrame, hour: float, suffix: str) -> pd.DataFrame:
    selected = hourly.loc[np.isclose(hourly["time_hour"], hour)].copy()
    if len(selected) != 45 or selected["case"].nunique() != 45:
        raise RuntimeError(f"Expected 45 unique cases at {hour:g} h; found {len(selected)} rows")
    return selected.rename(
        columns={
            "time_hour": f"{suffix}_time_hour",
            "time_day": f"{suffix}_time_day",
            "u_max_300_ms": f"{suffix}_u_max_300_ms",
            "jet_latitude_300_degN": f"{suffix}_jet_latitude_300_degN",
            "max_abs_dtheta850_dy_K_per_1000km": (
                f"{suffix}_max_abs_dtheta850_dy_K_per_1000km"
            ),
            "max_abs_dtheta850_dy_latitude_degN": (
                f"{suffix}_max_abs_dtheta850_dy_latitude_degN"
            ),
        }
    )


def strict_ordering_flags(table: pd.DataFrame) -> tuple[dict[tuple[float, int], bool], dict[tuple[int, int], bool]]:
    latitude_flags: dict[tuple[float, int], bool] = {}
    for (b_value, n_value), group in table.groupby(["b", "n"], sort=True):
        ordered = group.sort_values("s")
        latitude_flags[(float(b_value), int(n_value))] = bool(
            np.all(np.diff(ordered["day15_jet_latitude_300_degN"].to_numpy()) > 0)
        )

    wind_flags: dict[tuple[int, int], bool] = {}
    for (n_value, s_value), group in table.groupby(["n", "s"], sort=True):
        ordered = group.sort_values("b")
        wind_flags[(int(n_value), int(s_value))] = bool(
            np.all(np.diff(ordered["day15_u_max_300_ms"].to_numpy()) > 0)
        )
    return latitude_flags, wind_flags


def construct_plotted_table(hourly: pd.DataFrame) -> pd.DataFrame:
    identifying = ["case", "b", "n", "s"]
    initial = snapshot(hourly, 1.0, "initial")
    day8 = snapshot(hourly, 192.0, "day8")
    day15 = snapshot(hourly, 360.0, "day15")
    table = initial.merge(day8, on=identifying, validate="one_to_one")
    table = table.merge(day15, on=identifying, validate="one_to_one")

    for suffix in ("day8", "day15"):
        table[f"{suffix}_delta_u_max_300_ms"] = (
            table[f"{suffix}_u_max_300_ms"] - table["initial_u_max_300_ms"]
        )
        table[f"{suffix}_delta_jet_latitude_300_deg"] = (
            table[f"{suffix}_jet_latitude_300_degN"]
            - table["initial_jet_latitude_300_degN"]
        )
        table[f"{suffix}_delta_max_abs_dtheta850_dy_K_per_1000km"] = (
            table[f"{suffix}_max_abs_dtheta850_dy_K_per_1000km"]
            - table["initial_max_abs_dtheta850_dy_K_per_1000km"]
        )

    baseline = initial.set_index("case")
    hourly_with_initial = hourly.join(
        baseline[["initial_u_max_300_ms", "initial_jet_latitude_300_degN"]],
        on="case",
        validate="many_to_one",
    )
    hourly_with_initial["abs_u_change"] = np.abs(
        hourly_with_initial["u_max_300_ms"]
        - hourly_with_initial["initial_u_max_300_ms"]
    )
    hourly_with_initial["abs_latitude_change"] = np.abs(
        hourly_with_initial["jet_latitude_300_degN"]
        - hourly_with_initial["initial_jet_latitude_300_degN"]
    )
    interval = hourly_with_initial.groupby("case", sort=False).agg(
        analysis_max_abs_u_max_300_change_ms=("abs_u_change", "max"),
        analysis_max_abs_jet_latitude_300_change_deg=("abs_latitude_change", "max"),
    )
    table = table.merge(interval, left_on="case", right_index=True, validate="one_to_one")

    latitude_flags, wind_flags = strict_ordering_flags(table)
    table["fixed_b_n_group"] = table.apply(
        lambda row: f"b={row.b:g}, n={int(row.n)}", axis=1
    )
    table["day15_jet_latitude_strict_s_order_retained"] = table.apply(
        lambda row: latitude_flags[(float(row.b), int(row.n))], axis=1
    )
    table["fixed_n_s_group"] = table.apply(
        lambda row: f"n={int(row.n)}, s={int(row.s)}", axis=1
    )
    table["day15_u_max_300_strict_b_order_retained"] = table.apply(
        lambda row: wind_flags[(int(row.n), int(row.s))], axis=1
    )
    return table.sort_values(["b", "n", "s", "case"]).reset_index(drop=True)


def verify_source_summaries(
    table: pd.DataFrame,
    case_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    methods_text: str,
) -> None:
    if "Initial value: first available model output at 1 h" not in methods_text:
        raise RuntimeError("Source methods file does not document the 1-h initial definition")

    source = case_summary.set_index("case").loc[table["case"]]
    comparisons = {
        "day15_delta_u_max_300_ms": "delta_u_max_300_ms",
        "day15_delta_jet_latitude_300_deg": "delta_jet_latitude_300_deg",
        "day15_delta_max_abs_dtheta850_dy_K_per_1000km": (
            "delta_max_abs_dtheta850_dy_K_per_1000km"
        ),
        "analysis_max_abs_u_max_300_change_ms": "analysis_max_abs_u_max_300_change_ms",
        "analysis_max_abs_jet_latitude_300_change_deg": (
            "analysis_max_abs_jet_latitude_300_change_deg"
        ),
    }
    for recomputed_column, source_column in comparisons.items():
        if not np.allclose(
            table[recomputed_column].to_numpy(dtype=float),
            source[source_column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError(f"Hourly recomputation disagrees with {source_column}")

    recomputed_initial_by_b = table.groupby("b")["initial_u_max_300_ms"].mean()
    source_initial_by_b = group_summary.loc[
        (group_summary["group_parameter"] == "b")
        & (group_summary["metric"] == "initial_u_max_300_ms")
    ].set_index("group_value")["mean"]
    if not np.allclose(
        recomputed_initial_by_b.sort_index().to_numpy(),
        source_initial_by_b.sort_index().to_numpy(),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("Hourly initial b means disagree with the group summary")


def descriptive_statistics(table: pd.DataFrame) -> dict[str, float | int | dict]:
    day8_u = table["day8_delta_u_max_300_ms"]
    day8_latitude = table["day8_delta_jet_latitude_300_deg"]
    day8_gradient = table["day8_delta_max_abs_dtheta850_dy_K_per_1000km"]
    day15_u = table["day15_delta_u_max_300_ms"]
    day15_latitude = table["day15_delta_jet_latitude_300_deg"]
    day15_gradient = table["day15_delta_max_abs_dtheta850_dy_K_per_1000km"]
    max_u = table["analysis_max_abs_u_max_300_change_ms"]
    max_latitude = table["analysis_max_abs_jet_latitude_300_change_deg"]

    initial_by_b = table.groupby("b")["initial_u_max_300_ms"].mean().to_dict()
    latitude_flags, wind_flags = strict_ordering_flags(table)

    grouped = {}
    for parameter in ("b", "n", "s"):
        grouped[parameter] = table.groupby(parameter).agg(
            day15_mean_delta_u_ms=("day15_delta_u_max_300_ms", "mean"),
            day15_mean_delta_latitude_deg=("day15_delta_jet_latitude_300_deg", "mean"),
            interval_mean_max_abs_u_change_ms=(
                "analysis_max_abs_u_max_300_change_ms",
                "mean",
            ),
            interval_mean_max_abs_latitude_change_deg=(
                "analysis_max_abs_jet_latitude_300_change_deg",
                "mean",
            ),
        )

    return {
        "day8_mean_signed_u": float(day8_u.mean()),
        "day8_mean_abs_u": float(day8_u.abs().mean()),
        "day8_min_abs_u": float(day8_u.abs().min()),
        "day8_max_abs_u": float(day8_u.abs().max()),
        "day8_mean_signed_latitude": float(day8_latitude.mean()),
        "day8_mean_abs_latitude": float(day8_latitude.abs().mean()),
        "day8_min_latitude": float(day8_latitude.min()),
        "day8_max_latitude": float(day8_latitude.max()),
        "day8_mean_gradient": float(day8_gradient.mean()),
        "day15_mean_signed_u": float(day15_u.mean()),
        "day15_mean_abs_u": float(day15_u.abs().mean()),
        "day15_min_signed_u": float(day15_u.min()),
        "day15_max_signed_u": float(day15_u.max()),
        "day15_mean_latitude": float(day15_latitude.mean()),
        "day15_median_latitude": float(day15_latitude.median()),
        "day15_min_latitude": float(day15_latitude.min()),
        "day15_max_latitude": float(day15_latitude.max()),
        "day15_mean_gradient": float(day15_gradient.mean()),
        "day15_min_gradient": float(day15_gradient.min()),
        "day15_max_gradient": float(day15_gradient.max()),
        "interval_median_max_abs_u": float(max_u.median()),
        "interval_min_max_abs_u": float(max_u.min()),
        "interval_max_max_abs_u": float(max_u.max()),
        "interval_mean_max_abs_latitude": float(max_latitude.mean()),
        "interval_min_max_abs_latitude": float(max_latitude.min()),
        "interval_max_max_abs_latitude": float(max_latitude.max()),
        "latitude_order_groups_retained": int(sum(latitude_flags.values())),
        "latitude_order_groups_total": int(len(latitude_flags)),
        "wind_order_groups_retained": int(sum(wind_flags.values())),
        "wind_order_groups_total": int(len(wind_flags)),
        "initial_by_b": initial_by_b,
        "grouped": grouped,
    }


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.018,
        0.975,
        label,
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        zorder=10,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="0.88", linewidth=0.45, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=7.3, width=0.7, length=3)


def plot_bottom_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    day8_column: str,
    day15_column: str,
    ylabel: str,
    panel_label: str,
    ylim: tuple[float, float],
) -> None:
    axis.axhline(0.0, color="0.35", linewidth=0.75, linestyle="--", zorder=0)
    for row in table.itertuples(index=False):
        x_position = float(row.s) + N_OFFSETS[int(row.n)]
        day8_value = float(getattr(row, day8_column))
        day15_value = float(getattr(row, day15_column))
        color = B_COLORS[float(row.b)]
        marker = N_MARKERS[int(row.n)]
        axis.plot(
            [x_position, x_position],
            [day8_value, day15_value],
            color="0.78",
            linewidth=0.45,
            zorder=1,
        )
        axis.scatter(
            x_position,
            day8_value,
            marker=marker,
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=0.85,
            alpha=0.68,
            zorder=2,
        )
        axis.scatter(
            x_position,
            day15_value,
            marker=marker,
            s=24,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    axis.set_xlim(-12.0, 12.0)
    axis.set_ylim(*ylim)
    axis.set_xticks([-10, -5, 0, 5, 10])
    axis.set_xlabel(r"Prescribed shift, $s$ (degrees)", fontsize=7.3, labelpad=2)
    axis.set_ylabel(ylabel, fontsize=7.0, labelpad=2)
    style_axis(axis)
    add_panel_label(axis, panel_label)


def create_figure(table: pd.DataFrame) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(6.5, 8.65), facecolor="white")
    outer_grid = figure.add_gridspec(
        7,
        1,
        height_ratios=[1.04, 0.045, 1.04, 0.085, 0.045, 0.075, 1.12],
        left=0.105,
        right=0.985,
        bottom=0.145,
        top=0.975,
        hspace=0.14,
    )
    top_grid = outer_grid[0].subgridspec(1, 3, wspace=0.14)
    middle_grid = outer_grid[2].subgridspec(1, 3, wspace=0.14)
    bottom_grid = outer_grid[6].subgridspec(1, 3, wspace=0.47)

    top_axes = [figure.add_subplot(top_grid[0, column]) for column in range(3)]
    middle_axes = [figure.add_subplot(middle_grid[0, column]) for column in range(3)]
    bottom_axes = [figure.add_subplot(bottom_grid[0, column]) for column in range(3)]
    top_colorbar_axis = figure.add_subplot(outer_grid[1])
    middle_colorbar_axis = figure.add_subplot(outer_grid[4])

    top_image = None
    middle_image = None
    for column, (case, shift) in enumerate(REPRESENTATIVE_CASES):
        data = read_representative_case(case)
        latitude = data["latitude"]
        pressure = data["pressure"]
        days = data["days"]
        initial = data["initial"]
        day15 = data["day15"]
        u300 = data["u300"]
        jet_latitudes = data["jet_latitudes"]

        top_axis = top_axes[column]
        top_image = top_axis.contourf(
            latitude,
            pressure,
            day15 - initial,
            levels=DELTA_LEVELS,
            cmap="RdBu_r",
            extend="both",
        )
        initial_contours = top_axis.contour(
            latitude,
            pressure,
            initial,
            levels=[10, 20, 30, 40],
            colors="0.22",
            linewidths=0.65,
        )
        top_axis.clabel(initial_contours, inline=True, fontsize=5.8, fmt="%d")
        top_axis.set_xlim(20.0, 80.0)
        top_axis.set_ylim(1000.0, 100.0)
        top_axis.set_xticks(LATITUDE_TICKS)
        top_axis.set_yticks(PRESSURE_TICKS)
        top_axis.set_title(rf"$s={shift}^\circ$", fontsize=8.8, pad=3)
        top_axis.tick_params(labelbottom=False)
        if column == 0:
            top_axis.set_ylabel("Pressure (hPa)", fontsize=8.2)
        else:
            top_axis.tick_params(labelleft=False)
        style_axis(top_axis)
        add_panel_label(top_axis, f"({chr(97 + column)})")

        middle_axis = middle_axes[column]
        middle_image = middle_axis.contourf(
            latitude,
            days,
            u300 - u300[0][None, :],
            levels=DELTA_LEVELS,
            cmap="RdBu_r",
            extend="both",
        )
        absolute_contours = middle_axis.contour(
            latitude,
            days,
            u300,
            levels=[10, 20, 30, 40, 50],
            colors="0.28",
            linewidths=0.55,
        )
        middle_axis.clabel(absolute_contours, inline=True, fontsize=5.8, fmt="%d")
        middle_axis.plot(
            jet_latitudes,
            days,
            color="#F2C500",
            linewidth=1.5,
            zorder=4,
        )
        middle_axis.axvline(
            jet_latitudes[0],
            color="#F2C500",
            linestyle="--",
            linewidth=0.9,
            zorder=4,
        )
        middle_axis.set_xlim(20.0, 80.0)
        middle_axis.set_ylim(days.min(), days.max())
        middle_axis.set_xticks(LATITUDE_TICKS)
        middle_axis.set_xlabel("Latitude (°N)", fontsize=8.2, labelpad=2)
        if column == 0:
            middle_axis.set_ylabel("Time (days)", fontsize=8.2)
        else:
            middle_axis.tick_params(labelleft=False)
        style_axis(middle_axis)
        add_panel_label(middle_axis, f"({chr(100 + column)})")

    top_colorbar = figure.colorbar(
        top_image,
        cax=top_colorbar_axis,
        orientation="horizontal",
        ticks=DELTA_TICKS,
    )
    top_colorbar.set_label(
        r"Day 15 minus initial $[u]$ (m s$^{-1}$)",
        fontsize=7.5,
        labelpad=1,
    )
    top_colorbar.ax.xaxis.set_label_position("top")
    top_colorbar.ax.tick_params(labelsize=6.8, length=2, pad=1)

    middle_colorbar = figure.colorbar(
        middle_image,
        cax=middle_colorbar_axis,
        orientation="horizontal",
        ticks=DELTA_TICKS,
    )
    middle_colorbar.set_label(
        r"300-hPa $[u]$ anomaly (m s$^{-1}$)",
        fontsize=7.5,
        labelpad=1,
    )
    middle_colorbar.ax.tick_params(labelsize=6.8, length=2, pad=1)

    curve_legend = [
        Line2D([0], [0], color="#F2C500", linewidth=1.5, label="Instantaneous maximum"),
        Line2D(
            [0],
            [0],
            color="#F2C500",
            linewidth=0.9,
            linestyle="--",
            label="Initial latitude",
        ),
    ]
    middle_axes[0].legend(
        handles=curve_legend,
        loc="upper right",
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        fontsize=5.8,
        handlelength=2.0,
        borderpad=0.25,
        labelspacing=0.25,
    )

    plot_bottom_panel(
        bottom_axes[0],
        table,
        "day8_delta_u_max_300_ms",
        "day15_delta_u_max_300_ms",
        "Change in 300-hPa\nmaximum zonal-mean wind\n(m s$^{-1}$)",
        "(g)",
        (-1.35, 4.05),
    )
    plot_bottom_panel(
        bottom_axes[1],
        table,
        "day8_delta_jet_latitude_300_deg",
        "day15_delta_jet_latitude_300_deg",
        "Change in 300-hPa\njet latitude\n(degrees)",
        "(h)",
        (-1.45, 9.65),
    )
    plot_bottom_panel(
        bottom_axes[2],
        table,
        "day8_delta_max_abs_dtheta850_dy_K_per_1000km",
        "day15_delta_max_abs_dtheta850_dy_K_per_1000km",
        r"Change in maximum 850-hPa" "\n" r"$|\partial[\theta]/\partial y|$" "\n" r"(K per 1000 km)",
        "(i)",
        (-5.35, 2.75),
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=B_COLORS[1.0], markeredgecolor="none", markersize=5.0, label=r"$b=1$"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=B_COLORS[1.5], markeredgecolor="none", markersize=5.0, label=r"$b=1.5$"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=B_COLORS[2.0], markeredgecolor="none", markersize=5.0, label=r"$b=2$"),
        Line2D([0], [0], marker=N_MARKERS[1], linestyle="none", markerfacecolor="white", markeredgecolor="0.2", markersize=5.0, label=r"$n=1$"),
        Line2D([0], [0], marker=N_MARKERS[3], linestyle="none", markerfacecolor="white", markeredgecolor="0.2", markersize=5.0, label=r"$n=3$"),
        Line2D([0], [0], marker=N_MARKERS[6], linestyle="none", markerfacecolor="white", markeredgecolor="0.2", markersize=5.0, label=r"$n=6$"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="0.45", markersize=5.0, label="Day 8"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.45", markeredgecolor="0.45", markersize=5.0, label="Day 15"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        columnspacing=1.3,
        handletextpad=0.4,
    )

    figure.savefig(PNG_PATH, dpi=300, facecolor="white")
    figure.savefig(PDF_PATH, facecolor="white")
    plt.close(figure)


def format_group_table(frame: pd.DataFrame) -> str:
    output = frame.copy().round(3)
    output.index.name = "parameter"
    columns = ["parameter", *output.columns.tolist()]
    rows = [[index, *row.tolist()] for index, row in output.iterrows()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        body.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *body])


def write_report(statistics: dict[str, float | int | dict], methods_text: str) -> None:
    initial_by_b = statistics["initial_by_b"]
    grouped = statistics["grouped"]
    source_method_lines = [line for line in methods_text.splitlines() if line.startswith("-")]
    source_method_summary = "\n".join(source_method_lines[:6])

    caption = (
        "**Figure S15. Evolution of the Eulerian zonal-mean jet and low-level "
        "baroclinicity in the 45-case standard ensemble.** Panels (a–c) show "
        "Day-15-minus-initial zonal-mean zonal-wind cross sections for the "
        "representative `b2n3` cases with prescribed shifts of −10°, 0°, and "
        "+10°; gray contours denote the initial zonal-mean wind. Panels (d–f) "
        "show the corresponding 300-hPa zonal-mean wind anomalies, with absolute "
        "wind contours and the instantaneous (solid yellow) and initial (dashed "
        "yellow) latitudes of the 300-hPa maximum. Panels (g–i) show changes in "
        "the 300-hPa maximum zonal-mean wind, its latitude, and the maximum "
        "850-hPa potential-temperature gradient for all standard cases. Open "
        "and filled symbols indicate Days 8 and 15, respectively; colors denote "
        "`b`, and marker shapes denote `n`."
    )

    response = (
        "We added an ensemble-wide diagnosis of Eulerian zonal-mean-flow evolution. "
        f"At Day 8, the mean absolute changes in the 300-hPa maximum zonal-mean "
        f"wind and its latitude are only {statistics['day8_mean_abs_u']:.3f} m s−1 "
        f"and {statistics['day8_mean_abs_latitude']:.2f}°, respectively, with all "
        f"latitude changes between {statistics['day8_min_latitude']:.0f}° and "
        f"+{statistics['day8_max_latitude']:.0f}°. By Day 15, after mature nonlinear "
        f"wave–mean-flow adjustment, the mean signed wind and latitude changes reach "
        f"+{statistics['day15_mean_signed_u']:.3f} m s−1 and "
        f"+{statistics['day15_mean_latitude']:.2f}°, with individual wind changes "
        f"from {statistics['day15_min_signed_u']:.3f} to "
        f"+{statistics['day15_max_signed_u']:.3f} m s−1 and latitude changes up to "
        f"{statistics['day15_max_latitude']:.0f}°. The adjustment is therefore not "
        "uniformly small and is strongest in broad, large-`b`, and poleward-shifted "
        "configurations. Nevertheless, the prescribed ordering remains identifiable: "
        f"all {statistics['latitude_order_groups_retained']}/"
        f"{statistics['latitude_order_groups_total']} fixed-(`b`,`n`) groups retain "
        "strict Day-15 latitude ordering with `s`, and all "
        f"{statistics['wind_order_groups_retained']}/"
        f"{statistics['wind_order_groups_total']} fixed-(`n`,`s`) groups retain "
        "strict Day-15 wind-speed ordering with `b`. The maximum 850-hPa "
        f"potential-temperature gradient decreases by {abs(statistics['day15_mean_gradient']):.2f} "
        "K per 1000 km on average. These results characterize an initialized "
        "life-cycle sensitivity and are not interpreted as a statistically "
        "equilibrated storm-track response or a closed momentum budget."
    )

    text = rf"""# Revised Supplementary Figure S15: methods and verification

## Methods

The figure uses the 45 standard simulations only. All Day-8 and Day-15 values in panels (g–i) were independently recalculated from the hourly source table rather than copied from the case-summary table. The first available output at 1 h defines the initial state; Day 8 and Day 15 correspond to 192 and 360 h. At every hour, the 300-hPa maximum zonal-mean wind and its latitude are searched over 15–75°N. The low-level metric is

\[
G_{{850}}=\max_{{15^\circ\text{{–}}75^\circ\mathrm{{N}}}}
\left|\frac{{\partial[\theta]_{{850}}}}{{\partial y}}\right|.
\]

The upper six panels retain the representative-case analysis for `BCwave_b2n3s-10`, `BCwave_b2n3`, and `BCwave_b2n3s10`, calculated directly from their pressure-level NetCDF output. No EP-flux, TEM, PV, or closed momentum budget is inferred.

Source-method checks imported from `standard_jet_baroclinicity_methods.md`:

{source_method_summary}

## Independently verified numerical values

- **Day 8:** mean signed and mean absolute 300-hPa maximum-wind changes are {statistics['day8_mean_signed_u']:.6f} and {statistics['day8_mean_abs_u']:.6f} m s−1. Absolute changes range from {statistics['day8_min_abs_u']:.6f} to {statistics['day8_max_abs_u']:.6f} m s−1. The mean signed and mean absolute latitude changes are {statistics['day8_mean_signed_latitude']:.3f}° and {statistics['day8_mean_abs_latitude']:.3f}°, with a range of {statistics['day8_min_latitude']:.0f}° to +{statistics['day8_max_latitude']:.0f}°. The mean change in `G850` is {statistics['day8_mean_gradient']:.3f} K per 1000 km.
- **Day 15:** mean signed and mean absolute wind changes are +{statistics['day15_mean_signed_u']:.6f} and {statistics['day15_mean_abs_u']:.6f} m s−1. The signed range is {statistics['day15_min_signed_u']:.6f} to +{statistics['day15_max_signed_u']:.6f} m s−1. The mean and median latitude changes are +{statistics['day15_mean_latitude']:.3f}° and {statistics['day15_median_latitude']:.0f}°, with a range of {statistics['day15_min_latitude']:.0f}–{statistics['day15_max_latitude']:.0f}°. The mean `G850` change is {statistics['day15_mean_gradient']:.6f} K per 1000 km, with a range of {statistics['day15_min_gradient']:.3f} to +{statistics['day15_max_gradient']:.3f} K per 1000 km.
- **Full 1–360 h interval:** the median case maximum absolute wind-speed change is {statistics['interval_median_max_abs_u']:.3f} m s−1, with a range of {statistics['interval_min_max_abs_u']:.3f}–{statistics['interval_max_max_abs_u']:.3f} m s−1. The mean case maximum absolute latitude change is {statistics['interval_mean_max_abs_latitude']:.3f}°, with a range of {statistics['interval_min_max_abs_latitude']:.0f}–{statistics['interval_max_max_abs_latitude']:.0f}°.
- **Initial diagnostic distinction:** the initial 300-hPa maximum zonal-mean winds average {initial_by_b[1.0]:.3f}, {initial_by_b[1.5]:.3f}, and {initial_by_b[2.0]:.3f} m s−1 for `b=1`, `1.5`, and `2`, respectively. These are zonal-mean 300-hPa maxima, not generic three-dimensional jet-core maxima.

Mean adjustment by prescribed parameter:

### Grouped by `b`

{format_group_table(grouped['b'])}

### Grouped by `n`

{format_group_table(grouped['n'])}

### Grouped by `s`

{format_group_table(grouped['s'])}

## Ordering checks

- Day-15 jet latitude remains strictly ordered with prescribed `s` in **{statistics['latitude_order_groups_retained']}/{statistics['latitude_order_groups_total']}** fixed-(`b`,`n`) groups.
- Day-15 300-hPa maximum zonal-mean wind remains strictly ordered with `b` in **{statistics['wind_order_groups_retained']}/{statistics['wind_order_groups_total']}** fixed-(`n`,`s`) groups.

## Proposed caption

{caption}

## Proposed R1-10 response paragraph

{response}
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def write_manifest() -> None:
    deliverables = [PNG_PATH, PDF_PATH, Path(__file__).resolve(), TABLE_PATH, REPORT_PATH]
    lines = []
    for path in deliverables:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_PACKAGE.exists():
        raise FileNotFoundError(SOURCE_PACKAGE)

    with zipfile.ZipFile(SOURCE_PACKAGE) as archive:
        hourly = read_package_csv(archive, HOURLY_MEMBER)
        case_summary = read_package_csv(archive, CASE_SUMMARY_MEMBER)
        group_summary = read_package_csv(archive, GROUP_SUMMARY_MEMBER)
        methods_text = read_package_text(archive, METHODS_MEMBER)

    table = construct_plotted_table(hourly)
    verify_source_summaries(table, case_summary, group_summary, methods_text)
    statistics = descriptive_statistics(table)

    table.to_csv(TABLE_PATH, index=False)
    create_figure(table)
    write_report(statistics, methods_text)
    write_manifest()

    print(PNG_PATH)
    print(PDF_PATH)
    print(TABLE_PATH)
    print(REPORT_PATH)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
