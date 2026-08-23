#!/usr/bin/env python3
"""Replot the two eddy-SLP heatmap figures using JAMES/AGU figure styling."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper_revision" / "count_duration_heatmaps_left_b_axis_replot_20260819"
OUTPUT = ROOT / "paper_revision" / "count_duration_heatmaps_james_qc_20260819"
FIGURES = OUTPUT / "figures"

B_VALUES = [1.0, 1.5, 2.0]
N_VALUES = [1, 3, 6]
S_VALUES = [-10, -5, 0, 5, 10]
ENSEMBLES = ["constant u0", "constant Umax"]


def heatmap_array(
    data: pd.DataFrame,
    ensemble: str,
    shift: int,
    value_column: str,
) -> np.ndarray:
    subset = data[(data["ensemble"] == ensemble) & (data["s"] == shift)]
    values = (
        subset.pivot(index="b", columns="n", values=value_column)
        .reindex(index=B_VALUES, columns=N_VALUES)
        .to_numpy(dtype=float)
    )
    if values.shape != (3, 3) or np.isnan(values).any():
        raise RuntimeError(
            f"Incomplete heatmap for ensemble={ensemble}, s={shift}, column={value_column}"
        )
    return values


def plot_heatmap_figure(
    data: pd.DataFrame,
    value_column: str,
    color_max: float,
    colorbar_ticks: list[float],
    colorbar_label: str,
    output_stem: str,
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        f"{output_stem}_white_red",
        ["#ffffff", "#ff0000"],
        N=256,
    )
    figure = plt.figure(figsize=(7.25, 11.8), facecolor="white")
    grid = figure.add_gridspec(
        5,
        3,
        width_ratios=[1.0, 1.0, 0.12],
        left=0.15,
        right=0.91,
        top=0.955,
        bottom=0.065,
        wspace=0.18,
        hspace=0.34,
    )

    image = None
    panel_index = 0
    for row, shift in enumerate(S_VALUES):
        for column, ensemble in enumerate(ENSEMBLES):
            axis = figure.add_subplot(grid[row, column])
            values = heatmap_array(data, ensemble, shift, value_column)
            image = axis.imshow(
                values,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=color_max,
                aspect="equal",
                interpolation="nearest",
            )

            for row_index in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    value = values[row_index, column_index]
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=11.5,
                        color="white" if value / color_max > 0.52 else "black",
                    )

            panel_letter = chr(ord("a") + panel_index)
            panel_index += 1
            axis.text(
                -0.045,
                1.018,
                f"({panel_letter})",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=12.0,
                fontweight="bold",
                clip_on=False,
                zorder=5,
            )

            axis.set_xticks(range(3), [str(value) for value in N_VALUES])
            axis.set_yticks(range(3), [f"{value:g}" for value in B_VALUES])
            if column == 0:
                axis.text(
                    -0.30,
                    0.50,
                    r"$b$",
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False,
                )
            axis.tick_params(axis="both", labelsize=10.2, length=3.0, width=0.8)

            if row == 0:
                title = (
                    r"constant $u_0$"
                    if ensemble == "constant u0"
                    else r"constant $U_{\max}$"
                )
                axis.set_title(title, fontsize=13.5, pad=24, weight="semibold")
            if row == len(S_VALUES) - 1:
                axis.set_xlabel(r"$n$", fontsize=12.3, labelpad=3)
            else:
                axis.tick_params(axis="x", labelbottom=False)
            if column == 0:
                axis.set_ylabel(rf"$s={shift:+g}^\circ$", fontsize=12.2, labelpad=18)
            else:
                axis.tick_params(axis="y", labelleft=False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.9)

    if image is None:
        raise RuntimeError("No heatmap image was generated")
    colorbar_axis = figure.add_subplot(grid[:, 2])
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=colorbar_ticks,
    )
    colorbar.set_label(colorbar_label, fontsize=11.6, labelpad=9)
    colorbar.ax.tick_params(labelsize=9.5, length=3)

    for suffix in ("png", "pdf"):
        path = FIGURES / f"{output_stem}.{suffix}"
        figure.savefig(
            path,
            dpi=320 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.06,
        )
        print(path)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    count_data = pd.read_csv(
        SOURCE / "tables" / "Figure6_s5_consistent_plotted_case_counts.csv"
    )
    duration_data = pd.read_csv(
        SOURCE / "tables" / "Figure13_contrack_case_audit_all90.csv"
    )
    if len(count_data) != 90 or len(duration_data) != 90:
        raise RuntimeError(
            f"Expected 90 rows per table; found {len(count_data)} and {len(duration_data)}"
        )

    plot_heatmap_figure(
        data=count_data,
        value_column="plotted_eddy_minus10_count",
        color_max=75.0,
        colorbar_ticks=[0, 15, 30, 45, 60, 75],
        colorbar_label=r"Closed eddy-SLP $-10$-hPa events",
        output_stem="Figure6_eddy_slp_coalescence_heatmaps",
    )
    plot_heatmap_figure(
        data=duration_data,
        value_column="eddy_duration_h",
        color_max=260.0,
        colorbar_ticks=[0, 50, 100, 150, 200, 250],
        colorbar_label="Maximum eddy-SLP persistence (h)",
        output_stem="Figure13_eddy_anticyclone_overlap_duration",
    )


if __name__ == "__main__":
    main()
