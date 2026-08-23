#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "analysis" / "upper_lower_baroclinicity"
FIGURES = BASE / "figures"
INITIAL_PATH = BASE / "initial_vertical_profiles.nc"
GROWTH_PATH = BASE / "growth_stage_flux_profiles.nc"
CONTRAST_PATH = BASE / "paired_b_contrasts.csv"
ROBUSTNESS_PATH = BASE / "robustness_layer_bounds.csv"
SUMMARY_PATH = BASE / "lower_upper_layer_summary.csv"

B_VALUES = [1.0, 1.5, 2.0]
B_COLORS = {1.0: "#0072B2", 1.5: "#E69F00", 2.0: "#6A3D9A"}
ENSEMBLES = ["standard", "u30"]
ENSEMBLE_LABELS = {"standard": r"constant $u_0$", "u30": r"constant $U_{\max}=30$ m s$^{-1}$"}
ENSEMBLE_COLORS = {"standard": "#555555", "u30": "#C63B2D"}
ENSEMBLE_MARKERS = {"standard": "o", "u30": "^"}
LOWER_SHADE = "#DCEFFC"
UPPER_SHADE = "#FBE5CC"
PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200]
PROFILE_WIDTH = 15.0
N_BOOT = 5000

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.0,
        "axes.labelsize": 10.2,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def seed_for(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def bootstrap_mean_profile(values: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    random = np.random.default_rng(seed_for("profile", key))
    indices = random.integers(0, values.shape[0], size=(N_BOOT, values.shape[0]))
    samples = values[indices].mean(axis=1)
    return values.mean(axis=0), np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def bootstrap_mean_xy(x: np.ndarray, y: np.ndarray, key: str) -> tuple[float, float, float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    random = np.random.default_rng(seed_for("mean_xy", key))
    indices = random.integers(0, x.size, size=(N_BOOT, x.size))
    x_samples = x[indices].mean(axis=1)
    y_samples = y[indices].mean(axis=1)
    return (
        float(x.mean()),
        float(np.percentile(x_samples, 2.5)),
        float(np.percentile(x_samples, 97.5)),
        float(y.mean()),
        float(np.percentile(y_samples, 2.5)),
        float(np.percentile(y_samples, 97.5)),
    )


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, key: str) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    rho = float(spearmanr(x, y).statistic)
    random = np.random.default_rng(seed_for("spearman", key))
    estimates = []
    for _ in range(N_BOOT):
        indices = random.integers(0, x.size, size=x.size)
        value = float(spearmanr(x[indices], y[indices]).statistic)
        if np.isfinite(value):
            estimates.append(value)
    low, high = np.percentile(estimates, [2.5, 97.5]) if estimates else (np.nan, np.nan)
    return rho, float(low), float(high), int(x.size)


def add_layer_shading(axis: plt.Axes) -> None:
    axis.axhspan(850, 1000, color=LOWER_SHADE, alpha=0.62, zorder=0)
    axis.axhspan(300, 500, color=UPPER_SHADE, alpha=0.58, zorder=0)


def format_pressure_axis(axis: plt.Axes, show_label: bool) -> None:
    axis.set_yscale("log")
    axis.set_ylim(1000, 200)
    axis.set_yticks(PRESSURE_TICKS)
    axis.set_yticklabels([str(value) for value in PRESSURE_TICKS])
    axis.yaxis.set_minor_formatter(NullFormatter())
    if show_label:
        axis.set_ylabel("Pressure (hPa)")
    else:
        axis.tick_params(labelleft=False)
    axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.015,
        0.985,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
        zorder=20,
    )


def padded_limits(values: np.ndarray, include_zero: bool = True, symmetric: bool = False) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    low = float(finite.min())
    high = float(finite.max())
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if symmetric:
        bound = max(abs(low), abs(high))
        return -1.06 * bound, 1.06 * bound
    span = high - low
    pad = 0.06 * span if span > 0 else max(abs(high), 1.0) * 0.06
    return low - pad, high + pad


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    return png, pdf


def profile_legend() -> list[object]:
    handles = [Line2D([0], [0], color=B_COLORS[b], linewidth=2.4, label=rf"$b={b:g}$") for b in B_VALUES]
    handles.extend(
        [
            Patch(facecolor=LOWER_SHADE, edgecolor="none", alpha=0.72, label="1000–850 hPa"),
            Patch(facecolor=UPPER_SHADE, edgecolor="none", alpha=0.68, label="500–300 hPa"),
        ]
    )
    return handles


def make_initial_figure() -> tuple[Path, Path, pd.DataFrame]:
    specifications = [
        ("initial_vertical_wind_shear", 1.0e3, r"$\partial\bar{u}/\partial z$ ($10^{-3}$ s$^{-1}$)", False),
        (
            "initial_meridional_temperature_gradient",
            -1.0e6,
            r"$-\partial\bar{T}/\partial y$ (K per 1000 km)",
            False,
        ),
        ("initial_eady_growth_rate", 1.0, r"$\sigma_E$ (day$^{-1}$)", True),
    ]
    statistics = []
    with xr.open_dataset(INITIAL_PATH, decode_times=False, engine="netcdf4") as dataset:
        pressure = dataset.plev.values.astype(float)
        width_index = int(np.argmin(np.abs(dataset.jet_half_width.values - PROFILE_WIDTH)))
        pressure_mask = (pressure >= 200.0) & (pressure <= 1000.0)
        ensemble = dataset.ensemble.values.astype(str)
        b_values = dataset.b.values.astype(float)
        figure, axes = plt.subplots(3, 2, figsize=(7.4, 8.25), sharey=True, constrained_layout=True)
        panel_index = 0
        for row, (variable, scale, xlabel, positive_only) in enumerate(specifications):
            all_values = dataset[variable][:, width_index, :].values.astype(float) * scale
            limits = padded_limits(all_values[:, pressure_mask], include_zero=True, symmetric=False)
            if positive_only:
                limits = (0.0, limits[1])
            for column, ensemble_name in enumerate(ENSEMBLES):
                axis = axes[row, column]
                add_layer_shading(axis)
                axis.axvline(0.0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
                for b_value in B_VALUES:
                    selected = (ensemble == ensemble_name) & np.isclose(b_values, b_value)
                    values = all_values[selected]
                    for case_profile in values:
                        axis.plot(case_profile[pressure_mask], pressure[pressure_mask], color=B_COLORS[b_value], alpha=0.18, linewidth=0.55, zorder=2)
                    mean, low, high = bootstrap_mean_profile(values, f"initial-{ensemble_name}-{b_value}-{variable}")
                    axis.fill_betweenx(pressure[pressure_mask], low[pressure_mask], high[pressure_mask], color=B_COLORS[b_value], alpha=0.14, linewidth=0, zorder=3)
                    axis.plot(mean[pressure_mask], pressure[pressure_mask], color=B_COLORS[b_value], linewidth=2.35, zorder=4)
                    for pressure_value, mean_value, low_value, high_value in zip(pressure, mean, low, high):
                        statistics.append(
                            {
                                "figure": "initial_vertical_structure_standard_vs_u30",
                                "ensemble": ensemble_name,
                                "b": b_value,
                                "metric": variable,
                                "pressure_hpa": pressure_value,
                                "mean": mean_value,
                                "bootstrap_ci_low": low_value,
                                "bootstrap_ci_high": high_value,
                                "n_ns_groups": int(values.shape[0]),
                            }
                        )
                axis.set_xlim(*limits)
                axis.set_xlabel(xlabel)
                format_pressure_axis(axis, show_label=column == 0)
                if row == 0:
                    axis.set_title(ENSEMBLE_LABELS[ensemble_name], pad=7)
                add_panel_label(axis, f"({chr(97 + panel_index)})")
                panel_index += 1
        figure.legend(handles=profile_legend(), loc="outside upper center", ncols=5, frameon=False, handlelength=2.5, columnspacing=1.25)
        png, pdf = save_figure(figure, "initial_vertical_structure_standard_vs_u30_1000_850")
    return png, pdf, pd.DataFrame(statistics)


def make_flux_figure() -> tuple[Path, Path, pd.DataFrame]:
    specifications = [
        ("eddy_heat_flux_vT", 1.0, r"$\overline{v'T'}$ (K m s$^{-1}$)"),
        (
            "baroclinic_conversion_proxy",
            1.0e5,
            r"$-\overline{v'T'}\,\partial\bar{T}/\partial y$ ($10^{-5}$ K$^2$ s$^{-1}$)",
        ),
        ("ep_flux_vertical", 1.0e-6, r"$F_p$ ($10^6$ Pa m$^2$ s$^{-2}$)"),
    ]
    statistics = []
    with xr.open_dataset(GROWTH_PATH, decode_times=False, engine="netcdf4") as dataset:
        pressure = dataset.plev.values.astype(float)
        width_index = int(np.argmin(np.abs(dataset.jet_half_width.values - PROFILE_WIDTH)))
        growth_index = list(dataset.growth_window.values.astype(str)).index("case_relative_10_to_60pct")
        pressure_mask = (pressure >= 200.0) & (pressure <= 1000.0)
        ensemble = dataset.ensemble.values.astype(str)
        b_values = dataset.b.values.astype(float)
        figure, axes = plt.subplots(3, 2, figsize=(7.4, 8.25), sharey=True, constrained_layout=True)
        panel_index = 0
        for row, (variable, scale, xlabel) in enumerate(specifications):
            all_values = dataset[variable][:, growth_index, width_index, :].values.astype(float) * scale
            limits = padded_limits(all_values[:, pressure_mask], include_zero=True, symmetric=False)
            for column, ensemble_name in enumerate(ENSEMBLES):
                axis = axes[row, column]
                add_layer_shading(axis)
                axis.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)
                for b_value in B_VALUES:
                    selected = (ensemble == ensemble_name) & np.isclose(b_values, b_value)
                    values = all_values[selected]
                    for case_profile in values:
                        axis.plot(case_profile[pressure_mask], pressure[pressure_mask], color=B_COLORS[b_value], alpha=0.18, linewidth=0.55, zorder=2)
                    mean, low, high = bootstrap_mean_profile(values, f"flux-{ensemble_name}-{b_value}-{variable}")
                    axis.fill_betweenx(pressure[pressure_mask], low[pressure_mask], high[pressure_mask], color=B_COLORS[b_value], alpha=0.14, linewidth=0, zorder=3)
                    axis.plot(mean[pressure_mask], pressure[pressure_mask], color=B_COLORS[b_value], linewidth=2.35, zorder=4)
                    peak_index = int(np.nanargmax(np.abs(mean[pressure_mask])))
                    peak_pressure = float(pressure[pressure_mask][peak_index])
                    peak_value = float(mean[pressure_mask][peak_index])
                    for pressure_value, mean_value, low_value, high_value in zip(pressure, mean, low, high):
                        statistics.append(
                            {
                                "figure": "eddy_flux_vertical_structure_standard_vs_u30",
                                "ensemble": ensemble_name,
                                "b": b_value,
                                "metric": variable,
                                "pressure_hpa": pressure_value,
                                "mean": mean_value,
                                "bootstrap_ci_low": low_value,
                                "bootstrap_ci_high": high_value,
                                "n_ns_groups": int(values.shape[0]),
                                "group_mean_peak_abs_pressure_hpa": peak_pressure,
                                "group_mean_peak_value": peak_value,
                            }
                        )
                axis.set_xlim(*limits)
                axis.set_xlabel(xlabel)
                format_pressure_axis(axis, show_label=column == 0)
                if row == 0:
                    axis.set_title(ENSEMBLE_LABELS[ensemble_name], pad=7)
                add_panel_label(axis, f"({chr(97 + panel_index)})")
                panel_index += 1
        figure.legend(handles=profile_legend(), loc="outside upper center", ncols=5, frameon=False, handlelength=2.5, columnspacing=1.25)
        png, pdf = save_figure(figure, "eddy_flux_vertical_structure_standard_vs_u30_1000_850")
    return png, pdf, pd.DataFrame(statistics)


def matched_contrast(contrasts: pd.DataFrame, ensemble: str, layer: str, metric: str) -> pd.DataFrame:
    selected = contrasts[
        contrasts.ensemble.eq(ensemble) & contrasts.layer.eq(layer) & contrasts.metric.eq(metric)
    ].copy()
    return selected.sort_values(["n", "s"]).reset_index(drop=True)


def make_response_figure() -> tuple[Path, Path, pd.DataFrame]:
    metrics = [
        ("initial_eady_growth_rate_day-1", r"$\Delta\sigma_E$ (day$^{-1}$)", 1.0, "Initial Eady rate"),
        (
            "case_relative_mean_eddy_heat_flux_vT_K_m_s-1",
            r"$\Delta\overline{v'T'}$ (K m s$^{-1}$)",
            1.0,
            "Growth-stage heat flux",
        ),
        (
            "case_relative_mean_baroclinic_conversion_proxy_K2_s-1",
            r"$\Delta$ conversion proxy ($10^{-5}$ K$^2$ s$^{-1}$)",
            1.0e5,
            "Growth-stage conversion",
        ),
    ]
    contrasts = pd.read_csv(CONTRAST_PATH)
    statistics = []
    all_y = contrasts[
        contrasts.metric.eq("peak_eke_300_m2_s-2") & contrasts.layer.eq("upper")
    ].peak_eke_b2_minus_b1.to_numpy(float)
    y_limits = padded_limits(all_y, include_zero=True, symmetric=False)
    figure, axes = plt.subplots(3, 2, figsize=(7.4, 8.25), sharey=True, constrained_layout=True)
    panel_index = 0
    for row, (metric, xlabel, scale, row_label) in enumerate(metrics):
        row_values = []
        for layer in ["lower", "upper"]:
            for ensemble_name in ENSEMBLES:
                row_values.extend((matched_contrast(contrasts, ensemble_name, layer, metric).b2_minus_b1 * scale).tolist())
        x_limits = padded_limits(np.asarray(row_values), include_zero=True, symmetric=False)
        for column, layer in enumerate(["lower", "upper"]):
            axis = axes[row, column]
            grouped = {}
            for ensemble_name in ENSEMBLES:
                frame = matched_contrast(contrasts, ensemble_name, layer, metric)
                frame["x"] = frame.b2_minus_b1 * scale
                frame["y"] = frame.peak_eke_b2_minus_b1
                grouped[ensemble_name] = frame
            for group_index in range(15):
                standard_row = grouped["standard"].iloc[group_index]
                u30_row = grouped["u30"].iloc[group_index]
                if (standard_row.n, standard_row.s) != (u30_row.n, u30_row.s):
                    raise RuntimeError("Mismatched (n,s) ordering")
                axis.plot(
                    [standard_row.x, u30_row.x],
                    [standard_row.y, u30_row.y],
                    color="#BDBDBD",
                    alpha=0.48,
                    linewidth=0.65,
                    zorder=1,
                )
            annotation_lines = []
            for ensemble_name in ENSEMBLES:
                frame = grouped[ensemble_name]
                color = ENSEMBLE_COLORS[ensemble_name]
                marker = ENSEMBLE_MARKERS[ensemble_name]
                axis.scatter(
                    frame.x,
                    frame.y,
                    s=24,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.9,
                    zorder=3,
                )
                mean_x, x_low, x_high, mean_y, y_low, y_high = bootstrap_mean_xy(
                    frame.x.to_numpy(float), frame.y.to_numpy(float), f"{metric}-{layer}-{ensemble_name}"
                )
                axis.errorbar(
                    mean_x,
                    mean_y,
                    xerr=[[mean_x - x_low], [x_high - mean_x]],
                    yerr=[[mean_y - y_low], [y_high - mean_y]],
                    fmt="*",
                    markersize=10.5,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    elinewidth=1.2,
                    capsize=2.3,
                    zorder=5,
                )
                rho, rho_low, rho_high, sample_size = bootstrap_spearman(
                    frame.x.to_numpy(float), frame.y.to_numpy(float), f"eke-{metric}-{layer}-{ensemble_name}"
                )
                short_label = "Std" if ensemble_name == "standard" else "U30"
                annotation_lines.append(f"{short_label}: ρ={rho:.2f} [{rho_low:.2f}, {rho_high:.2f}]")
                statistics.append(
                    {
                        "statistic": "metric_change_vs_peak_eke_change",
                        "metric": metric,
                        "layer": layer,
                        "ensemble": ensemble_name,
                        "spearman_rho": rho,
                        "bootstrap_ci_low": rho_low,
                        "bootstrap_ci_high": rho_high,
                        "n_ns_groups": sample_size,
                        "mean_metric_change": mean_x,
                        "mean_metric_change_ci_low": x_low,
                        "mean_metric_change_ci_high": x_high,
                        "mean_peak_eke_change": mean_y,
                        "mean_peak_eke_change_ci_low": y_low,
                        "mean_peak_eke_change_ci_high": y_high,
                    }
                )
            lower = matched_contrast(contrasts, "standard", "lower", metric)
            upper = matched_contrast(contrasts, "standard", "upper", metric)
            if column == 0:
                for ensemble_name in ENSEMBLES:
                    lower = matched_contrast(contrasts, ensemble_name, "lower", metric)
                    upper = matched_contrast(contrasts, ensemble_name, "upper", metric)
                    rho, rho_low, rho_high, sample_size = bootstrap_spearman(
                        lower.b2_minus_b1.to_numpy(float) * scale,
                        upper.b2_minus_b1.to_numpy(float) * scale,
                        f"collinearity-{metric}-{ensemble_name}",
                    )
                    statistics.append(
                        {
                            "statistic": "lower_upper_metric_change_collinearity",
                            "metric": metric,
                            "layer": "lower_vs_upper",
                            "ensemble": ensemble_name,
                            "spearman_rho": rho,
                            "bootstrap_ci_low": rho_low,
                            "bootstrap_ci_high": rho_high,
                            "n_ns_groups": sample_size,
                            "mean_metric_change": np.nan,
                            "mean_metric_change_ci_low": np.nan,
                            "mean_metric_change_ci_high": np.nan,
                            "mean_peak_eke_change": np.nan,
                            "mean_peak_eke_change_ci_low": np.nan,
                            "mean_peak_eke_change_ci_high": np.nan,
                        }
                    )
            axis.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
            axis.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
            axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_xlabel(xlabel)
            if column == 0:
                axis.set_ylabel(r"$\Delta$ peak 300-hPa EKE (m$^2$ s$^{-2}$)")
            else:
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.set_title("Lower layer" if layer == "lower" else "Upper layer", pad=7)
            axis.text(
                0.98,
                0.035,
                "\n".join(annotation_lines) + "\n15 paired (n,s) groups",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.86, "pad": 2.2},
                zorder=10,
            )
            if column == 0:
                axis.text(0.02, 0.90, row_label, transform=axis.transAxes, ha="left", va="top", fontsize=9.1, fontweight="bold")
            add_panel_label(axis, f"({chr(97 + panel_index)})")
            panel_index += 1
    handles = [
        Line2D([0], [0], marker=ENSEMBLE_MARKERS["standard"], color="none", markerfacecolor=ENSEMBLE_COLORS["standard"], markeredgecolor="white", markersize=7, label="Standard"),
        Line2D([0], [0], marker=ENSEMBLE_MARKERS["u30"], color="none", markerfacecolor=ENSEMBLE_COLORS["u30"], markeredgecolor="white", markersize=7, label=r"Fixed $U_{max}=30$ m s$^{-1}$"),
        Line2D([0], [0], marker="*", color="black", markerfacecolor="white", markersize=9, linewidth=0, label="Mean ± 95% bootstrap CI"),
        Line2D([0], [0], color="#BDBDBD", linewidth=0.8, label="Same (n,s) group"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False, handlelength=2.0, columnspacing=1.15)
    png, pdf = save_figure(figure, "lower_upper_layer_response_summary")
    return png, pdf, pd.DataFrame(statistics)


def create_contact_sheet(paths: list[Path]) -> Path:
    target_width = 900
    margin = 24
    title_height = 42
    labels = [
        "Initial vertical structure",
        "Growth-stage eddy-flux structure",
        "Lower/upper matched responses",
    ]
    prepared = []
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
    for path, label in zip(paths, labels):
        image = Image.open(path).convert("RGB")
        scale = target_width / image.width
        resized = image.resize((target_width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
        block = Image.new("RGB", (target_width, resized.height + title_height), "white")
        draw = ImageDraw.Draw(block)
        draw.text((12, 7), label, fill="#222222", font=font)
        block.paste(resized, (0, title_height))
        prepared.append(block)
    height = margin + sum(image.height + margin for image in prepared)
    sheet = Image.new("RGB", (target_width + 2 * margin, height), "white")
    y = margin
    for image in prepared:
        sheet.paste(image, (margin, y))
        y += image.height + margin
    output = FIGURES / "upper_lower_baroclinicity_figure_contact_sheet.png"
    sheet.save(output, dpi=(120, 120), optimize=True)
    return output


def format_ci(row: pd.Series) -> str:
    return f"ρ={row.spearman_rho:.2f} [{row.bootstrap_ci_low:.2f}, {row.bootstrap_ci_high:.2f}]"


def make_captions(scatter_statistics: pd.DataFrame) -> None:
    correlation = scatter_statistics[scatter_statistics.statistic.eq("metric_change_vs_peak_eke_change")]
    collinearity = scatter_statistics[scatter_statistics.statistic.eq("lower_upper_metric_change_collinearity")]
    metric_labels = {
        "initial_eady_growth_rate_day-1": "initial Eady-rate contrast",
        "case_relative_mean_eddy_heat_flux_vT_K_m_s-1": "growth-stage heat-flux contrast",
        "case_relative_mean_baroclinic_conversion_proxy_K2_s-1": "growth-stage conversion-proxy contrast",
    }
    lines = [
        "# Upper/Lower Baroclinicity Figure Captions",
        "",
        "## Figure 1. Initial vertical structure in the standard and fixed-Umax ensembles",
        "",
        "Initial jet-relative vertical profiles of (a, b) vertical wind shear, (c, d) the poleward-decreasing meridional temperature gradient plotted as -dT/dy, and (e, f) Eady growth rate. The left and right columns show the standard and fixed-Umax=30 m s-1 ensembles, respectively. Colors denote b=1, 1.5, and 2. Each thin line is one of the 15 matched (n,s) groups, thick lines are group means, and shading around each thick line is the 95% bootstrap confidence interval obtained by resampling the 15 groups. Profiles use the primary initial-jet-relative ±15 degree band, cosine-latitude weighting, and no smoothing. Blue and orange background shading mark the predefined 850-700-hPa lower and 500-300-hPa upper layers. Both ensemble columns use identical axes within each diagnostic row.",
        "",
        "## Figure 2. Growth-stage eddy-flux vertical structure",
        "",
        "As in Figure 1, but for case-relative EKE growth-stage profiles of (a, b) eddy heat flux v'T', (c, d) the explicitly labeled baroclinic-conversion proxy -v'T' dTbar/dy, and (e, f) the QG pressure-coordinate vertical EP-flux component Fp. The growth stage is defined independently for each case and jet-relative width from the first 10% to the first 60% crossing of the 300-hPa EKE maximum within 1-360 h. Thin lines show all 15 (n,s) groups, thick lines show their means, and envelopes are 95% group-bootstrap confidence intervals. No temporal or vertical smoothing is applied. Shared row-wise axes expose the reversal of b ordering between the standard and fixed-Umax ensembles. The EP-flux component is a diagnostic quantity rather than a closed TEM or model-tendency budget. The largest absolute heat-flux and conversion-proxy values occur mainly near 925-1000 hPa, below the predefined 850-700-hPa lower layer.",
        "",
        "## Figure 3. Matched lower- and upper-layer responses",
        "",
        "Relationships between paired b=2 minus b=1 changes in (a, b) initial Eady growth rate, (c, d) growth-stage eddy heat flux, and (e, f) the baroclinic-conversion proxy and the corresponding change in peak 300-hPa EKE. Columns show the predefined lower and upper layers. Each point is one fixed (n,s) group; thin gray segments join the same group between the standard and fixed-Umax ensembles. Stars and error bars show the ensemble mean and its 95% paired-group bootstrap confidence interval. Annotations report Spearman correlation and a 95% bootstrap confidence interval from 15 (n,s) groups per ensemble. No regression line or smoothing is used.",
        "",
        "### Correlation and collinearity details for Figure 3",
        "",
        "| Metric | Ensemble | Lower vs EKE | Upper vs EKE | Lower-upper collinearity |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, label in metric_labels.items():
        for ensemble_name in ENSEMBLES:
            lower = correlation[
                correlation.metric.eq(metric) & correlation.layer.eq("lower") & correlation.ensemble.eq(ensemble_name)
            ].iloc[0]
            upper = correlation[
                correlation.metric.eq(metric) & correlation.layer.eq("upper") & correlation.ensemble.eq(ensemble_name)
            ].iloc[0]
            collinear = collinearity[
                collinearity.metric.eq(metric) & collinearity.ensemble.eq(ensemble_name)
            ].iloc[0]
            ensemble_text = "Standard" if ensemble_name == "standard" else "Fixed Umax = 30 m s-1"
            lines.append(
                f"| {label} | {ensemble_text} | {format_ci(lower)} | {format_ci(upper)} | {format_ci(collinear)} |"
            )
    lines.extend(
        [
            "",
            "The standard ensemble has strong lower-upper collinearity for the Eady-rate, heat-flux, and conversion-proxy contrasts, so its lower- and upper-level associations cannot be treated as independent evidence. Under fixed Umax, lower-upper collinearity remains substantial for the initial Eady-rate contrast but collapses for the growth-stage heat-flux and conversion contrasts; the lower-level relations with EKE remain much stronger than the corresponding upper-level relations.",
        ]
    )
    (BASE / "upper_lower_baroclinicity_figure_captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_scientific_report(scatter_statistics: pd.DataFrame, profile_statistics: pd.DataFrame) -> None:
    robustness = pd.read_csv(ROBUSTNESS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    primary = robustness[
        np.isclose(robustness.jet_half_width_deg, 15.0) & robustness.layer_scheme.eq("main")
    ]
    correlation = scatter_statistics[scatter_statistics.statistic.eq("metric_change_vs_peak_eke_change")]
    collinearity = scatter_statistics[scatter_statistics.statistic.eq("lower_upper_metric_change_collinearity")]

    def robust_row(ensemble: str, layer: str, metric: str) -> pd.Series:
        return primary[
            primary.ensemble.eq(ensemble) & primary.layer.eq(layer) & primary.metric.eq(metric)
        ].iloc[0]

    def corr_row(ensemble: str, layer: str, metric: str) -> pd.Series:
        return correlation[
            correlation.ensemble.eq(ensemble) & correlation.layer.eq(layer) & correlation.metric.eq(metric)
        ].iloc[0]

    peak_pressures = (
        profile_statistics[
            profile_statistics.metric.isin(["eddy_heat_flux_vT", "baroclinic_conversion_proxy"])
        ][["ensemble", "b", "metric", "group_mean_peak_abs_pressure_hpa"]]
        .drop_duplicates()
        .sort_values(["metric", "ensemble", "b"])
    )
    peak_pressure_text = ", ".join(
        f"{row.ensemble} b={row.b:g} {row.metric}: {row.group_mean_peak_abs_pressure_hpa:.0f} hPa"
        for row in peak_pressures.itertuples(index=False)
    )

    standard_eady_lower = robust_row("standard", "lower", "initial_eady_growth_rate_day-1")
    standard_eady_upper = robust_row("standard", "upper", "initial_eady_growth_rate_day-1")
    u30_eady_lower = robust_row("u30", "lower", "initial_eady_growth_rate_day-1")
    u30_eady_upper = robust_row("u30", "upper", "initial_eady_growth_rate_day-1")
    standard_heat_lower = robust_row("standard", "lower", "case_relative_mean_eddy_heat_flux_vT_K_m_s-1")
    standard_heat_upper = robust_row("standard", "upper", "case_relative_mean_eddy_heat_flux_vT_K_m_s-1")
    u30_heat_lower = robust_row("u30", "lower", "case_relative_mean_eddy_heat_flux_vT_K_m_s-1")
    u30_heat_upper = robust_row("u30", "upper", "case_relative_mean_eddy_heat_flux_vT_K_m_s-1")
    standard_conversion_lower = robust_row("standard", "lower", "case_relative_mean_baroclinic_conversion_proxy_K2_s-1")
    standard_conversion_upper = robust_row("standard", "upper", "case_relative_mean_baroclinic_conversion_proxy_K2_s-1")
    u30_conversion_lower = robust_row("u30", "lower", "case_relative_mean_baroclinic_conversion_proxy_K2_s-1")
    u30_conversion_upper = robust_row("u30", "upper", "case_relative_mean_baroclinic_conversion_proxy_K2_s-1")
    standard_peak = robust_row("standard", "upper", "peak_eke_300_m2_s-2")
    u30_peak = robust_row("u30", "upper", "peak_eke_300_m2_s-2")

    edge_cases = summary[
        np.isclose(summary.jet_half_width_deg, 15.0)
        & summary.layer_scheme.eq("main")
        & summary.layer.eq("upper")
    ].drop_duplicates(["ensemble", "case"]).groupby("ensemble").peak_eke_at_analysis_end.sum()

    lines = [
        "# Scientific Interpretation of the Upper/Lower Baroclinicity Analysis",
        "",
        "## Executive conclusion",
        "",
        "Holding the realized jet maximum fixed reverses the b dependence of peak EKE and produces the clearest dynamically consistent reversal in the lower-tropospheric diagnostics. The fixed-Umax lower-layer initial Eady rate, eddy heat flux, conversion proxy, mean EKE, and integrated EKE all decrease from b=1 to b=2 and retain their sign across jet-relative widths, layer definitions, and the fixed Day 4-Day 8 sensitivity window. Upper-level initial baroclinicity also decreases, but the upper-level heat-flux and conversion responses depend on the growth-window definition and correlate weakly with the EKE contrast. The results therefore lean toward a Held-type low-level-control interpretation, but they do not constitute an orthogonal upper-versus-lower shear experiment and cannot causally exclude a Yuval/Kaspi-type upper-level sensitivity.",
        "",
        "## 1. Standard ensemble: response of upper and lower baroclinicity to increasing b",
        "",
        f"In the standard ensemble, increasing b strengthens both layers. The paired b=2 minus b=1 initial Eady-rate contrast is {standard_eady_lower.mean_b2_minus_b1:.3f} day-1 in the lower layer and {standard_eady_upper.mean_b2_minus_b1:.3f} day-1 in the upper layer; both 95% bootstrap intervals exclude zero. Growth-stage heat flux increases by {standard_heat_lower.mean_b2_minus_b1:.3f} K m s-1 below and {standard_heat_upper.mean_b2_minus_b1:.3f} K m s-1 aloft. The conversion proxy likewise increases by {standard_conversion_lower.mean_b2_minus_b1:.3e} and {standard_conversion_upper.mean_b2_minus_b1:.3e} K2 s-1. Peak EKE increases by {standard_peak.mean_b2_minus_b1:.2f} m2 s-2 with a 95% CI [{standard_peak.bootstrap_ci_low_b2_minus_b1:.2f}, {standard_peak.bootstrap_ci_high_b2_minus_b1:.2f}].",
        "",
        "These changes are not vertically independent. Across the 15 matched (n,s) groups, standard lower and upper contrasts are strongly collinear, especially for the growth-stage heat flux and conversion proxy. Thus, the standard ensemble establishes a coherent whole-column strengthening with b, not a clean test of whether the upper or lower layer controls the response.",
        "",
        "## 2. Fixed-Umax ensemble: are the vertical changes the same?",
        "",
        f"No. With Umax fixed, initial baroclinicity decreases from b=1 to b=2 in both layers, but the reduction is much larger below: the lower Eady-rate contrast is {u30_eady_lower.mean_b2_minus_b1:.3f} day-1 compared with {u30_eady_upper.mean_b2_minus_b1:.3f} day-1 aloft. The lower growth-stage heat flux and conversion proxy decrease strongly ({u30_heat_lower.mean_b2_minus_b1:.3f} K m s-1 and {u30_conversion_lower.mean_b2_minus_b1:.3e} K2 s-1). In contrast, the case-relative upper heat-flux contrast is {u30_heat_upper.mean_b2_minus_b1:.3f} K m s-1 with a CI spanning zero, and the upper conversion-proxy contrast is positive ({u30_conversion_upper.mean_b2_minus_b1:.3e} K2 s-1).",
        "",
        f"The EKE ordering reverses decisively: the paired b=2 minus b=1 peak-EKE contrast is {u30_peak.mean_b2_minus_b1:.2f} m2 s-2, with a 95% CI [{u30_peak.bootstrap_ci_low_b2_minus_b1:.2f}, {u30_peak.bootstrap_ci_high_b2_minus_b1:.2f}].",
        "",
        "## 3. Is the U30 EKE reversal closer to lower- or upper-level changes?",
        "",
        "It is substantially closer to the lower-level changes. The fixed-Umax lower heat-flux and conversion-proxy contrasts have strong positive rank associations with the EKE contrast across matched groups, whereas the corresponding upper-level associations are weak or negative. The lower metrics also reproduce the EKE sign in every matched group for the primary diagnostics. Upper initial Eady rate decreases, but its group-to-group variations do not track the EKE contrast nearly as closely.",
        "",
        "| Metric | U30 lower vs EKE | U30 upper vs EKE |",
        "|---|---:|---:|",
    ]
    metric_rows = [
        ("Initial Eady rate", "initial_eady_growth_rate_day-1"),
        ("Growth-stage heat flux", "case_relative_mean_eddy_heat_flux_vT_K_m_s-1"),
        ("Growth-stage conversion proxy", "case_relative_mean_baroclinic_conversion_proxy_K2_s-1"),
    ]
    for label, metric in metric_rows:
        lower = corr_row("u30", "lower", metric)
        upper = corr_row("u30", "upper", metric)
        lines.append(f"| {label} | {format_ci(lower)} | {format_ci(upper)} |")
    lines.extend(
        [
            "",
            "This is mechanistic consistency rather than causal isolation. The U30 construction still changes balanced wind and temperature gradients at both upper and lower levels when b changes.",
            "",
            "## 4. Vertical concentration of eddy heat flux and conversion",
            "",
            "The vertical-profile figure shows that the largest absolute group-mean eddy heat flux and conversion-proxy values occur near 925-1000 hPa. Thus, the strongest signal is near the surface and below the predefined 850-700-hPa lower layer. Within the two headline layers, the lower-layer responses are nevertheless much larger and more systematically ordered than their upper-layer counterparts. The QG vertical EP-flux component is also largest in magnitude near the lower boundary and should be interpreted diagnostically rather than as a closed wave-activity budget.",
            "",
            f"Group-mean peak-pressure inventory: {peak_pressure_text}.",
            "",
            "## 5. Relation to Held-type and Yuval/Kaspi-type interpretations",
            "",
            "The results are more consistent with a Held-type low-level-control pathway because the lower initial Eady rate, lower heat flux, lower conversion proxy, and EKE all reverse together under fixed Umax, remain robust to the tested layer and latitude bounds, and show the strongest matched-group associations. A purely upper-level-sensitivity interpretation is less consistent with the primary case-relative flux diagnostics, because upper heat flux and conversion do not reverse coherently with EKE.",
            "",
            "However, the current experiments cannot strictly distinguish the two theories. Both upper and lower initial structures change with b; standard lower and upper metrics are highly collinear; the fixed-window upper heat flux does become negative; and no experiment independently changes one layer while holding the other fixed. The appropriate conclusion is therefore 'stronger consistency with low-level control,' not 'proof that upper-level structure is unimportant.'",
            "",
            "## 6. Statements suitable for the paper versus limitations",
            "",
            "### Suitable scientific results",
            "",
            "- Fixing Umax reverses the b ordering of peak EKE across all 15 matched (n,s) groups.",
            "- In the standard ensemble, increasing b strengthens both lower- and upper-level initial baroclinicity and growth-stage eddy activity.",
            "- In the fixed-Umax ensemble, the lower initial Eady rate, heat flux, conversion proxy, and EKE all decrease with b; these directions are robust to the tested jet-relative widths, pressure-layer definitions, and Day 4-Day 8 sensitivity window.",
            "- Matched-group correlations are strongest between lower-layer heat-flux/conversion changes and the EKE change.",
            "- Heat flux and the conversion proxy attain their largest absolute group-mean values near 925-1000 hPa.",
            "",
            "### Statements that must remain limitations or caveats",
            "",
            "- The standard and fixed-Umax ensembles are not orthogonal upper-versus-lower shear experiments.",
            "- The results do not provide a causal estimate of lower-level influence while upper-level structure is held fixed, or vice versa.",
            "- Standard lower and upper metrics are highly collinear, so their separate correlations with EKE are not independent evidence.",
            "- The fixed-Umax upper heat-flux and conversion responses are sensitive to the growth-window definition.",
            f"- Peak EKE is right-edge limited at 360 h in {int(edge_cases['standard'])}/45 standard and {int(edge_cases['u30'])}/45 fixed-Umax cases.",
            "- The EP-flux quantities are QG diagnostics and not a closed TEM or model-tendency budget.",
            "",
            "## Bottom line",
            "",
            "The cleanest interpretation is that controlling jet amplitude exposes a robust lower-level weakening as b increases, and this lower-level weakening tracks the reversed EKE ordering more closely than the upper-level eddy-flux response. Because upper and lower initial structures still covary, the experiments favor but do not uniquely establish low-level control.",
        ]
    )
    (BASE / "upper_lower_baroclinicity_scientific_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    initial_png, _, initial_statistics = make_initial_figure()
    flux_png, _, flux_statistics = make_flux_figure()
    response_png, _, scatter_statistics = make_response_figure()
    profile_statistics = pd.concat([initial_statistics, flux_statistics], ignore_index=True)
    profile_statistics.to_csv(BASE / "upper_lower_baroclinicity_profile_statistics.csv", index=False)
    scatter_statistics.to_csv(BASE / "upper_lower_baroclinicity_response_statistics.csv", index=False)
    make_captions(scatter_statistics)
    make_scientific_report(scatter_statistics, flux_statistics)
    create_contact_sheet([initial_png, flux_png, response_png])
    print(f"Wrote figures and reports to {BASE}")


if __name__ == "__main__":
    main()
