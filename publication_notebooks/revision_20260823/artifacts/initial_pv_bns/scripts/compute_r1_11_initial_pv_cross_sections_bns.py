#!/usr/bin/env python3
"""Nine-panel initial pressure-coordinate Ertel-PV cross sections for R1-11."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from compute_r1_12_initial_pv_cross_sections import (
    deepest_crossing_pressure,
    pressure_coordinate_pv,
)


ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "paper_revision" / "supplemental_analysis" / "figures"
PNG_PATH = FIGDIR / "r1_11_initial_pv_cross_sections_bns.png"
PDF_PATH = FIGDIR / "r1_11_initial_pv_cross_sections_bns.pdf"

PANEL_ROWS = [
    {
        "row_label": r"Varying $s$",
        "panels": [
            ("BCwave_b2n3s-10", r"$s=-10^\circ$"),
            ("BCwave_b2n3", r"$s=0^\circ$"),
            ("BCwave_b2n3s10", r"$s=+10^\circ$"),
        ],
    },
    {
        "row_label": r"Varying $n$",
        "panels": [
            ("BCwave_b2n1", r"$n=1$"),
            ("BCwave_b2n3", r"$n=3$"),
            ("BCwave_b2n6", r"$n=6$"),
        ],
    },
    {
        "row_label": r"Varying $b$",
        "panels": [
            ("BCwave_b1n3", r"$b=1.0$"),
            ("BCwave_b15n3", r"$b=1.5$"),
            ("BCwave_b2n3", r"$b=2.0$"),
        ],
    },
]

PV_LEVELS = np.arange(0.0, 8.5, 0.5)
THETA_LEVELS = np.arange(270.0, 381.0, 10.0)
THETA_LABEL_LEVELS = np.array([280.0, 320.0, 360.0])
THETA_LABEL_LATITUDES = np.array([60.0, 73.0, 68.0])
LATITUDE_LIMITS = (15.0, 80.0)
PRESSURE_LIMITS = (1000.0, 100.0)
LATITUDE_TICKS = [20, 30, 40, 50, 60, 70, 80]
PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200, 100]
PV_CMAP = "YlOrRd"
TROPOPAUSE_COLOR = "#1261A0"


def requested_case_sequence() -> list[str]:
    return [case for row in PANEL_ROWS for case, _ in row["panels"]]


def validate_requested_layout() -> None:
    expected = [
        "BCwave_b2n3s-10",
        "BCwave_b2n3",
        "BCwave_b2n3s10",
        "BCwave_b2n1",
        "BCwave_b2n3",
        "BCwave_b2n6",
        "BCwave_b1n3",
        "BCwave_b15n3",
        "BCwave_b2n3",
    ]
    actual = requested_case_sequence()
    if actual != expected:
        raise RuntimeError(f"Panel configuration mismatch: {actual}")
    counts = Counter(actual)
    if counts["BCwave_b2n3"] != 3:
        raise RuntimeError("The reference BCwave_b2n3 case must appear exactly three times")
    if any(count != 1 for case, count in counts.items() if case != "BCwave_b2n3"):
        raise RuntimeError(f"Unexpected duplicate experiment: {counts}")
    for case in counts:
        path = ROOT / f"{case}.nc"
        if not path.exists():
            raise FileNotFoundError(path)


def load_initial_diagnostics(case: str) -> dict[str, np.ndarray]:
    source = ROOT / f"{case}.nc"
    with xr.open_dataset(source, decode_times=False) as dataset:
        latitude = dataset["grid_yt"].values.astype(float)
        pressure = dataset["plev"].values.astype(float)
        time_hour = float(dataset["time"].isel(time=0).values)
        zonal_mean_u = (
            dataset["u_plev"].isel(time=0).mean("grid_xt").load().values.astype(float)
        )
        zonal_mean_temperature = (
            dataset["t_plev"].isel(time=0).mean("grid_xt").load().values.astype(float)
        )

    latitude_order = np.argsort(latitude)
    latitude = latitude[latitude_order]
    zonal_mean_u = zonal_mean_u[:, latitude_order]
    zonal_mean_temperature = zonal_mean_temperature[:, latitude_order]
    theta, pv = pressure_coordinate_pv(
        zonal_mean_u,
        zonal_mean_temperature,
        pressure,
        latitude,
    )
    return {
        "source": np.array(str(source)),
        "time_hour": np.array(time_hour),
        "latitude": latitude,
        "pressure": pressure,
        "theta": theta,
        "pv": pv,
    }


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.018,
        0.974,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7},
        zorder=10,
    )


def add_theta_labels(
    axis: plt.Axes,
    latitude: np.ndarray,
    pressure: np.ndarray,
    theta: np.ndarray,
) -> None:
    for theta_level, label_latitude in zip(THETA_LABEL_LEVELS, THETA_LABEL_LATITUDES):
        latitude_index = int(np.nanargmin(np.abs(latitude - label_latitude)))
        label_pressure = deepest_crossing_pressure(
            pressure,
            theta[:, latitude_index],
            target=theta_level,
        )
        if np.isfinite(label_pressure):
            axis.text(
                latitude[latitude_index],
                label_pressure,
                f"{theta_level:.0f} K",
                color="0.27",
                fontsize=6.8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.54, "pad": 0.25},
                zorder=5,
            )


def add_2pvu_label(
    axis: plt.Axes,
    latitude: np.ndarray,
    pressure: np.ndarray,
    pv: np.ndarray,
) -> None:
    label_latitude = 55.0
    latitude_index = int(np.nanargmin(np.abs(latitude - label_latitude)))
    label_pressure = deepest_crossing_pressure(
        pressure,
        pv[:, latitude_index],
        target=2.0,
    )
    if np.isfinite(label_pressure):
        axis.text(
            latitude[latitude_index],
            label_pressure,
            "2 PVU",
            color=TROPOPAUSE_COLOR,
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.25},
            zorder=7,
        )


def validate_common_coordinates(cache: dict[str, dict[str, np.ndarray]]) -> None:
    reference = cache["BCwave_b2n3"]
    for case, data in cache.items():
        if not np.array_equal(data["latitude"], reference["latitude"]):
            raise RuntimeError(f"Latitude coordinates differ for {case}")
        if not np.array_equal(data["pressure"], reference["pressure"]):
            raise RuntimeError(f"Pressure coordinates differ for {case}")
        if not np.isclose(float(data["time_hour"]), float(reference["time_hour"])):
            raise RuntimeError(f"Initial output time differs for {case}")


def report_controlled_differences(cache: dict[str, dict[str, np.ndarray]]) -> None:
    for row in PANEL_ROWS:
        cases = [case for case, _ in row["panels"]]
        reference_pv = cache[cases[1]]["pv"]
        differences = [
            float(np.nanmax(np.abs(cache[case]["pv"] - reference_pv)))
            for case in (cases[0], cases[2])
        ]
        if not all(value > 0.0 for value in differences):
            raise RuntimeError(f"Unexpected duplicate scientific field in {row['row_label']}")
        print(
            f"{row['row_label']}: max |PV difference from center panel| = "
            f"{differences[0]:.6f}, {differences[1]:.6f} PVU"
        )


def main() -> None:
    validate_requested_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    unique_cases = list(dict.fromkeys(requested_case_sequence()))
    diagnostics = {case: load_initial_diagnostics(case) for case in unique_cases}
    validate_common_coordinates(diagnostics)
    report_controlled_differences(diagnostics)

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(7.2, 8.45),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    figure.subplots_adjust(
        left=0.125,
        right=0.987,
        top=0.925,
        bottom=0.145,
        wspace=0.075,
        hspace=0.17,
    )

    image = None
    panel_number = 0
    for row_index, row in enumerate(PANEL_ROWS):
        for column_index, (case, parameter_title) in enumerate(row["panels"]):
            axis = axes[row_index, column_index]
            data = diagnostics[case]
            latitude = data["latitude"]
            pressure = data["pressure"]
            theta = data["theta"]
            pv = data["pv"]

            image = axis.contourf(
                latitude,
                pressure,
                pv,
                levels=PV_LEVELS,
                cmap=PV_CMAP,
                extend="max",
            )
            axis.contour(
                latitude,
                pressure,
                theta,
                levels=THETA_LEVELS,
                colors="0.28",
                linewidths=0.62,
            )
            axis.contour(
                latitude,
                pressure,
                pv,
                levels=[2.0],
                colors=TROPOPAUSE_COLOR,
                linewidths=1.9,
            )
            add_theta_labels(axis, latitude, pressure, theta)
            add_2pvu_label(axis, latitude, pressure, pv)

            axis.set_xlim(*LATITUDE_LIMITS)
            axis.set_ylim(*PRESSURE_LIMITS)
            axis.set_xticks(LATITUDE_TICKS)
            axis.set_yticks(PRESSURE_TICKS)
            axis.set_title(parameter_title, fontsize=9.6, pad=3.0)
            axis.grid(True, color="0.86", linewidth=0.42, alpha=0.62)
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=7.5,
                width=0.7,
                length=3.0,
                labelleft=column_index == 0,
                labelbottom=row_index == 2,
                bottom=True,
                left=True,
            )
            if row_index == 2:
                axis.set_xlabel("Latitude (°N)", fontsize=8.4, labelpad=2)
            add_panel_label(axis, f"({chr(97 + panel_number)})")
            panel_number += 1

    figure.suptitle(
        "Idealized initial PV and dynamical-tropopause structure",
        fontsize=12.2,
        y=0.979,
    )
    figure.text(
        0.069,
        0.535,
        "Pressure (hPa)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.0,
    )
    for row_index, row in enumerate(PANEL_ROWS):
        row_position = axes[row_index, 0].get_position()
        figure.text(
            0.018,
            0.5 * (row_position.y0 + row_position.y1),
            row["row_label"],
            rotation=90,
            va="center",
            ha="center",
            fontsize=9.4,
            fontweight="bold",
        )

    colorbar_axis = figure.add_axes([0.245, 0.055, 0.55, 0.022])
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=np.arange(0.0, 9.0, 1.0),
    )
    colorbar.set_label("Initial zonal-mean Ertel PV (PVU)", fontsize=8.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=7.5, width=0.7, length=2.5, pad=1.5)

    figure.savefig(PNG_PATH, dpi=300, facecolor="white")
    figure.savefig(PDF_PATH, facecolor="white")
    plt.close(figure)

    print("Panel experiments:")
    for panel_index, case in enumerate(requested_case_sequence()):
        print(f"  ({chr(97 + panel_index)}) {case}: {ROOT / (case + '.nc')}")
    print(f"Common PV levels: {PV_LEVELS.tolist()} PVU")
    print(f"Common theta levels: {THETA_LEVELS.tolist()} K")
    print(f"Common latitude limits: {LATITUDE_LIMITS}")
    print(f"Common pressure limits: {PRESSURE_LIMITS}")
    print(f"Initial time: {float(diagnostics['BCwave_b2n3']['time_hour']):g} h")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
