#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = (
    ROOT
    / "publication_notebooks"
    / "outputs"
    / "Figure6_EKE_timeseries_area_mass_weighted.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent

B_VALUES = [2.0, 1.5, 1.0]
N_VALUES = [6, 3, 1]
S_VALUES = [0, 10, -10]
S_COLORS = {0: "blue", 10: "red", -10: "green"}
S_LINESTYLES = {0: "solid", 10: "dashed", -10: "dotted"}

PLOT_START_HOUR = 144.0
EKE_COLUMN = "eke_area_mass_weighted_m2_s-2"
EADY_COLUMN = "initial_eady_area_mass_weighted_30_70N_day-1"

A_EARTH = 6_371_000.0
OMEGA = 7.2921159e-5
GRAVITY = 9.80665
RD = 287.05
CP = 1004.0


def pressure_trapezoid_weights(pressure_hpa: np.ndarray) -> np.ndarray:
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    if pressure_hpa.ndim != 1 or pressure_hpa.size < 2:
        raise ValueError("Pressure coordinate must contain at least two levels")
    if not np.all(np.diff(pressure_hpa) > 0):
        raise ValueError("Pressure levels must increase monotonically")
    weights = np.empty_like(pressure_hpa)
    weights[0] = 0.5 * (pressure_hpa[1] - pressure_hpa[0])
    weights[-1] = 0.5 * (pressure_hpa[-1] - pressure_hpa[-2])
    weights[1:-1] = 0.5 * (pressure_hpa[2:] - pressure_hpa[:-2])
    return weights


def compute_global_eady(path: Path) -> dict[str, float | str]:
    with h5py.File(path, "r") as dataset:
        pressure_hpa = np.asarray(dataset["plev"][:], dtype=float)
        latitude = np.asarray(dataset["grid_yt"][:], dtype=float)
        zonal_wind = np.asarray(dataset["u_plev"][0, :, :, :], dtype=float)
        temperature = np.asarray(dataset["t_plev"][0, :, :, :], dtype=float)

    zonal_wind[zonal_wind <= -1.0e9] = np.nan
    temperature[temperature <= -1.0e9] = np.nan
    zonal_wind = np.nanmean(zonal_wind, axis=-1)
    temperature = np.nanmean(temperature, axis=-1)

    pressure_order = np.argsort(pressure_hpa)
    latitude_order = np.argsort(latitude)
    pressure_hpa = pressure_hpa[pressure_order]
    latitude = latitude[latitude_order]
    zonal_wind = zonal_wind[pressure_order][:, latitude_order]
    temperature = temperature[pressure_order][:, latitude_order]

    theta = temperature * (1000.0 / pressure_hpa[:, None]) ** (RD / CP)
    log_pressure = np.log(pressure_hpa)
    du_dlogp = np.gradient(zonal_wind, log_pressure, axis=0, edge_order=2)
    dtheta_dlogp = np.gradient(theta, log_pressure, axis=0, edge_order=2)
    du_dz = -GRAVITY / (RD * temperature) * du_dlogp
    dtheta_dz = -GRAVITY / (RD * temperature) * dtheta_dlogp
    n_squared = GRAVITY / theta * dtheta_dz
    buoyancy_frequency = np.sqrt(np.where(n_squared > 1.0e-8, n_squared, np.nan))
    coriolis = 2.0 * OMEGA * np.sin(np.deg2rad(latitude))
    eady = 0.31 * np.abs(coriolis)[None, :] * np.abs(du_dz) / buoyancy_frequency
    eady *= 86400.0

    latitude_mask = (latitude >= 30.0) & (latitude <= 70.0)
    selected_latitude = latitude[latitude_mask]
    selected_eady = eady[:, latitude_mask]
    area_weights = np.cos(np.deg2rad(selected_latitude))
    area_weights /= area_weights.sum()
    pressure_weights_hpa = pressure_trapezoid_weights(pressure_hpa)
    pressure_weights = pressure_weights_hpa / pressure_weights_hpa.sum()
    weights = pressure_weights[:, None] * area_weights[None, :]
    valid = np.isfinite(selected_eady)
    numerator = np.nansum(np.where(valid, selected_eady * weights, 0.0))
    denominator = np.sum(np.where(valid, weights, 0.0))
    if denominator <= 0.0:
        raise RuntimeError(f"No valid initial Eady values in {path}")

    return {
        "case": path.stem,
        EADY_COLUMN: float(numerator / denominator),
        "eady_valid_weight_fraction": float(denominator),
        "eady_domain_latitude_min_deg": float(selected_latitude.min()),
        "eady_domain_latitude_max_deg": float(selected_latitude.max()),
        "eady_pressure_top_hpa": float(pressure_hpa.min()),
        "eady_pressure_bottom_hpa": float(pressure_hpa.max()),
    }


def make_figure(timeseries: pd.DataFrame, eady_data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 14),
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    panel_index = 0
    for row, b_value in enumerate(B_VALUES):
        for column, n_value in enumerate(N_VALUES):
            axis = axes[row, column]
            for s_value in S_VALUES:
                selected = timeseries[
                    (timeseries["b"] == b_value)
                    & (timeseries["n"] == n_value)
                    & (timeseries["s"] == s_value)
                    & (timeseries["time_h"] >= PLOT_START_HOUR)
                ]
                axis.plot(
                    selected["time_h"],
                    selected[EKE_COLUMN],
                    color=S_COLORS[s_value],
                    linestyle=S_LINESTYLES[s_value],
                    linewidth=2,
                )

            panel_eady = eady_data[
                (eady_data["b"] == b_value) & (eady_data["n"] == n_value)
            ].set_index("s")
            if set(panel_eady.index) != set(S_VALUES):
                raise RuntimeError(
                    f"Missing Eady values for b={b_value:g}, n={n_value}: "
                    f"found s={sorted(panel_eady.index.tolist())}"
                )
            annotation_box = FancyBboxPatch(
                (0.025, 0.555),
                0.72,
                0.405,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                transform=axis.transAxes,
                facecolor="white",
                edgecolor="0.70",
                linewidth=0.7,
                alpha=0.94,
                clip_on=True,
                zorder=3,
            )
            axis.add_patch(annotation_box)
            axis.text(
                0.05,
                0.91,
                "Initial NH-mean Eady rate",
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=13.2,
                fontweight="semibold",
                color="black",
                zorder=4,
            )
            axis.text(
                0.05,
                0.855,
                "30–70°N; area–mass weighted (day⁻¹)",
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=10.8,
                color="0.25",
                zorder=4,
            )
            for y_position, s_value in zip((0.765, 0.675, 0.585), S_VALUES):
                eady_value = float(panel_eady.loc[s_value, EADY_COLUMN])
                axis.plot(
                    [0.05, 0.13],
                    [y_position, y_position],
                    transform=axis.transAxes,
                    color=S_COLORS[s_value],
                    linestyle=S_LINESTYLES[s_value],
                    linewidth=2.4,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=4,
                )
                s_label = rf"$s={s_value:+d}^\circ$" if s_value else r"$s=0^\circ$"
                axis.text(
                    0.15,
                    y_position,
                    rf"{s_label}   {eady_value:.3f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="center",
                    fontsize=12.8,
                    color=S_COLORS[s_value],
                    zorder=4,
                )

            letter = chr(ord("a") + panel_index)
            panel_index += 1
            axis.set_title(
                rf"({letter}) $b={b_value:g}$, $n={n_value}$",
                fontsize=20,
                weight="bold",
                pad=6,
            )
            axis.set_xlim(144, 360)
            axis.set_ylim(0, 45)
            axis.set_xticks([150, 200, 250, 300, 350])
            axis.set_yticks([0, 20, 40])
            axis.tick_params(labelsize=20)
            for hour in range(144, 361, 48):
                axis.axvline(hour, color="lightgray", linewidth=0.5, zorder=0)
            for value in range(0, 41, 10):
                axis.axhline(
                    value,
                    color="lightgray",
                    linewidth=0.5,
                    linestyle="--",
                    zorder=0,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=S_COLORS[s_value],
            linestyle=S_LINESTYLES[s_value],
            linewidth=3,
            label=rf"$s={s_value:+d}^\circ$" if s_value else r"$s=0^\circ$",
        )
        for s_value in S_VALUES
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.992),
        ncol=3,
        frameon=False,
        fontsize=20,
        handlelength=3.0,
        columnspacing=2.2,
        handletextpad=0.7,
    )

    figure.supxlabel("Time (h)", fontsize=24, y=0.04)
    figure.supylabel(
        r"Area- and mass-weighted EKE (m$^2$ s$^{-2}$)",
        fontsize=24,
        x=0.015,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.09,
        top=0.885,
        wspace=0.10,
        hspace=0.22,
    )

    png_path = OUTPUT_DIR / "Figure6_EKE_evolution_with_midlatitude_Eady_mean.png"
    pdf_path = OUTPUT_DIR / "Figure6_EKE_evolution_with_midlatitude_Eady_mean.pdf"
    figure.savefig(
        png_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)
    print(png_path)
    print(pdf_path)


def main() -> None:
    timeseries = pd.read_csv(INPUT_CSV)
    expected_columns = {"b", "n", "s", "time_h", EKE_COLUMN}
    missing = expected_columns.difference(timeseries.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    cases = timeseries[["case", "b", "n", "s"]].drop_duplicates()
    eady_rows = []
    for case in cases["case"]:
        path = ROOT / f"{case}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"computing initial 30–70°N Eady mean: {case}", flush=True)
        eady_rows.append(compute_global_eady(path))
    eady_data = pd.DataFrame(eady_rows)
    plotted_eady = cases.merge(eady_data, on="case", how="left", validate="one_to_one")
    if plotted_eady[EADY_COLUMN].isna().any():
        missing_cases = plotted_eady.loc[plotted_eady[EADY_COLUMN].isna(), "case"].tolist()
        raise RuntimeError(f"Missing global Eady values for cases: {missing_cases}")
    plotted_eady.sort_values(["b", "n", "s"]).to_csv(
        OUTPUT_DIR / "Figure6_initial_midlatitude_Eady_annotations.csv",
        index=False,
    )
    make_figure(timeseries, plotted_eady)


if __name__ == "__main__":
    main()
