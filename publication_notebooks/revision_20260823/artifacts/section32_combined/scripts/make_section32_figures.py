#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "paper_revision" / "section32_vertical_profiles_eke_fluxes_20260820"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
DATA = OUTPUT / "data"
CHECKPOINTS = DATA / "case_checkpoints"

ANALYSIS = ROOT / "analysis" / "upper_lower_baroclinicity"
CASE_RESULTS = ANALYSIS / "case_results"
MATCHED = ANALYSIS / "matched_case_inventory.csv"
STANDARD_ROOT = ROOT
U30_ROOT = ROOT / "priority_revision_analysis_20260720" / "simulations" / "umax30_all_bns"

B_VALUES = [1.0, 1.5, 2.0]
N_VALUES = [1, 3, 6]
S_VALUES = [-10, -5, 0, 5, 10]
B_COLORS = {1.0: "#0072B2", 1.5: "#E69F00", 2.0: "#6A3D9A"}
ENSEMBLE_ORDER = ["standard", "u30"]
ENSEMBLE_LABELS = {"standard": r"Constant $u_0$", "u30": r"Constant $U_{\max}=30$ m s$^{-1}$"}
N_LINESTYLES = {1: "-", 3: "--", 6: ":"}
LOWER_SHADE = "#DCEFFC"
UPPER_SHADE = "#FBE5CC"
PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200]
PRESSURE_MIN = 200.0
PRESSURE_MAX = 1000.0
R_EARTH = 6_371_000.0
OMEGA = 7.2921159e-5
GRAVITY = 9.80665
RD = 287.05
CP = 1004.0
KAPPA = RD / CP
N_BOOT = 5000
TIME_BLOCK = 6
FLUX_WIDTH = 15.0
FLUX_VARIABLES = [
    ("eddy_heat_flux_vT", 1.0, r"$\overline{v'T'}$ (K m s$^{-1}$)"),
    ("baroclinic_conversion_proxy", 1.0e5, r"$-\overline{v'T'}\,\partial\overline{T}/\partial y$ ($10^{-5}$ K$^2$ s$^{-1}$)"),
    ("ep_flux_vertical", 1.0e-6, r"$F_p$ ($10^6$ Pa m$^2$ s$^{-2}$)"),
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.6,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def fill_float(values: object) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        return np.ma.filled(values, np.nan).astype(float, copy=False)
    array = np.asarray(values, dtype=float)
    array[array <= -1.0e9] = np.nan
    return array


def pressure_weights(pressure_hpa: np.ndarray) -> np.ndarray:
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    weights = np.empty_like(pressure_hpa)
    weights[0] = 0.5 * (pressure_hpa[1] - pressure_hpa[0])
    weights[-1] = 0.5 * (pressure_hpa[-1] - pressure_hpa[-2])
    weights[1:-1] = 0.5 * (pressure_hpa[2:] - pressure_hpa[:-2])
    return weights / weights.sum()


def parse_s(case: str) -> int:
    if "s-10" in case:
        return -10
    if "s-5" in case:
        return -5
    if "s10" in case:
        return 10
    if "s5" in case:
        return 5
    return 0


def case_metadata() -> pd.DataFrame:
    matched = pd.read_csv(MATCHED)
    rows = []
    for row in matched.itertuples(index=False):
        rows.append({
            "ensemble": "standard",
            "case": row.standard_case,
            "paired_case": row.u30_case,
            "b": float(row.b),
            "n": int(row.n),
            "s": int(row.s),
            "source_path": str(STANDARD_ROOT / f"{row.standard_case}.nc"),
        })
        rows.append({
            "ensemble": "u30",
            "case": row.u30_case,
            "paired_case": row.standard_case,
            "b": float(row.b),
            "n": int(row.n),
            "s": int(row.s),
            "source_path": str(U30_ROOT / row.u30_case / "atmos_4x_hourly.nc"),
        })
    metadata = pd.DataFrame(rows).sort_values(["ensemble", "b", "n", "s"]).reset_index(drop=True)
    if metadata.groupby("ensemble").size().to_dict() != {"standard": 45, "u30": 45}:
        raise RuntimeError("Expected 45 cases in each ensemble")
    for path in metadata.source_path:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    return metadata


def source_hour_indices(dataset: netCDF4.Dataset, ensemble: str) -> tuple[np.ndarray, np.ndarray]:
    source_time = fill_float(dataset.variables["time"][:])
    desired_time = np.arange(1.0, 361.0)
    if ensemble == "standard":
        indices = np.arange(desired_time.size, dtype=int)
        if not np.allclose(source_time[:360], desired_time):
            indices = np.array([int(np.argmin(np.abs(source_time - value))) for value in desired_time])
    else:
        indices = np.array([int(np.argmin(np.abs(source_time - value))) for value in desired_time])
    selected = source_time[indices]
    if not np.allclose(selected, desired_time, atol=1e-5):
        raise RuntimeError(f"Could not identify hourly outputs: {selected[:4]} ... {selected[-4:]}")
    return indices, desired_time


def read_time_block(variable: netCDF4.Variable, source_indices: np.ndarray, start: int, stop: int, latitude_slice: slice) -> np.ndarray:
    selected = source_indices[start:stop]
    if selected.size == 0:
        return np.empty((0,))
    differences = np.diff(selected)
    if selected.size == 1 or np.all(differences == differences[0]):
        step = 1 if selected.size == 1 else int(differences[0])
        source_start = int(selected[0])
        source_stop = int(selected[-1] + step)
        values = variable[source_start:source_stop:step, :, latitude_slice, :]
    else:
        values = variable[selected.tolist(), :, latitude_slice, :]
    return fill_float(values)


def initial_structure(u_initial: np.ndarray, temperature_initial: np.ndarray, pressure: np.ndarray, latitude: np.ndarray) -> dict[str, object]:
    zonal_u = np.nanmean(u_initial, axis=-1)
    zonal_temperature = np.nanmean(temperature_initial, axis=-1)
    k300 = int(np.argmin(np.abs(pressure - 300.0)))
    latitude_mask = (latitude >= 15.0) & (latitude <= 75.0)
    candidate_indices = np.flatnonzero(latitude_mask)
    core_index = int(candidate_indices[np.nanargmax(zonal_u[k300, latitude_mask])])
    core_latitude = float(latitude[core_index])
    pressure_order = np.argsort(pressure)
    pressure_sorted = pressure[pressure_order]
    u_sorted = zonal_u[pressure_order]
    temperature_sorted = zonal_temperature[pressure_order]
    core_u = u_sorted[:, core_index]
    core_temperature = temperature_sorted[:, core_index]
    theta = core_temperature * (1000.0 / pressure_sorted) ** KAPPA
    log_pressure = np.log(pressure_sorted)
    du_dlogp = np.gradient(core_u, log_pressure, edge_order=2)
    dtheta_dlogp = np.gradient(theta, log_pressure, edge_order=2)
    du_dz = -GRAVITY / (RD * core_temperature) * du_dlogp
    dtheta_dz = -GRAVITY / (RD * core_temperature) * dtheta_dlogp
    n_squared = GRAVITY / theta * dtheta_dz
    buoyancy_frequency = np.sqrt(np.where(n_squared > 1.0e-8, n_squared, np.nan))
    coriolis = 2.0 * OMEGA * np.sin(np.deg2rad(core_latitude))
    eady = 0.31 * abs(coriolis) * np.abs(du_dz) / buoyancy_frequency * 86400.0
    return {
        "pressure": pressure_sorted,
        "core_latitude": core_latitude,
        "zonal_wind": core_u,
        "eady": eady,
    }


def calculate_case(metadata_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensemble = str(metadata_row.ensemble)
    case = str(metadata_row.case)
    source = Path(str(metadata_row.source_path))
    with netCDF4.Dataset(source) as dataset:
        pressure = fill_float(dataset.variables["plev"][:])
        latitude = fill_float(dataset.variables["grid_yt"][:])
        source_indices, hourly_time = source_hour_indices(dataset, ensemble)
        north_indices = np.flatnonzero(latitude >= 0.0)
        if north_indices.size < 2:
            raise RuntimeError(f"No NH latitude points in {source}")
        north_slice = slice(int(north_indices[0]), int(north_indices[-1]) + 1)
        north_latitude = latitude[north_indices]
        area_weights = np.cos(np.deg2rad(north_latitude))
        area_weights = area_weights / area_weights.sum()
        mass_weights = pressure_weights(pressure)
        u_variable = dataset.variables["u_plev"]
        v_variable = dataset.variables["v_plev"]
        t_variable = dataset.variables["t_plev"]
        first_u = read_time_block(u_variable, source_indices, 0, 1, north_slice)[0]
        first_t = read_time_block(t_variable, source_indices, 0, 1, north_slice)[0]
        initial = initial_structure(first_u, first_t, pressure, north_latitude)
        eke_values = np.full(hourly_time.size, np.nan, dtype=float)
        for start in range(0, hourly_time.size, TIME_BLOCK):
            stop = min(start + TIME_BLOCK, hourly_time.size)
            u_values = read_time_block(u_variable, source_indices, start, stop, north_slice)
            v_values = read_time_block(v_variable, source_indices, start, stop, north_slice)
            u_prime = u_values - np.nanmean(u_values, axis=-1, keepdims=True)
            v_prime = v_values - np.nanmean(v_values, axis=-1, keepdims=True)
            eke = 0.5 * (u_prime ** 2 + v_prime ** 2)
            level_area_mean = np.nansum(np.nanmean(eke, axis=-1) * area_weights[None, None, :], axis=-1)
            eke_values[start:stop] = np.nansum(level_area_mean * mass_weights[None, :], axis=-1)
            print(f"    {ensemble:8s} {case:24s} EKE {stop:3d}/360", end="\r", flush=True)
        print("", flush=True)
    peak_index = int(np.nanargmax(eke_values))
    peak_value = float(eke_values[peak_index])
    rising_values = eke_values[: peak_index + 1]
    start_candidates = np.flatnonzero(rising_values >= 0.50 * peak_value)
    start_index = int(start_candidates[0]) if start_candidates.size else 0
    end_candidates = np.flatnonzero(rising_values[start_index:] >= 0.80 * peak_value)
    end_index = int(start_index + end_candidates[0]) if end_candidates.size else peak_index
    window_indices = np.arange(start_index, end_index + 1, dtype=int)
    result_path = CASE_RESULTS / f"{ensemble}__{case}.nc"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with netCDF4.Dataset(result_path) as result:
        result_time = fill_float(result.variables["time"][:])
        width = fill_float(result.variables["jet_half_width"][:])
        width_index = int(np.argmin(np.abs(width - FLUX_WIDTH)))
        flux_rows = []
        for variable_name, _, _ in FLUX_VARIABLES:
            values = fill_float(result.variables[variable_name][window_indices, width_index, :])
            mean_values = np.nanmean(values, axis=0)
            for pressure_value, mean_value in zip(result.variables["plev"][:], mean_values):
                flux_rows.append({
                    "ensemble": ensemble,
                    "case": case,
                    "b": float(metadata_row.b),
                    "n": int(metadata_row.n),
                    "s": int(metadata_row.s),
                    "variable": variable_name,
                    "pressure_hpa": float(pressure_value),
                    "value": float(mean_value),
                    "window_start_h": float(hourly_time[start_index]),
                    "window_end_h": float(hourly_time[end_index]),
                    "window_count": int(window_indices.size),
                })
    eke_rows = pd.DataFrame({
        "ensemble": ensemble,
        "case": case,
        "b": float(metadata_row.b),
        "n": int(metadata_row.n),
        "s": int(metadata_row.s),
        "time_h": hourly_time,
        "eke_nh_area_mass_weighted_m2_s-2": eke_values,
    })
    initial_rows = pd.DataFrame({
        "ensemble": ensemble,
        "case": case,
        "b": float(metadata_row.b),
        "n": int(metadata_row.n),
        "s": int(metadata_row.s),
        "pressure_hpa": initial["pressure"],
        "initial_zonal_wind_at_jet_core_m_s-1": initial["zonal_wind"],
        "initial_eady_growth_rate_day-1": initial["eady"],
        "initial_jet_core_latitude_deg": initial["core_latitude"],
    })
    summary = pd.DataFrame([{
        "ensemble": ensemble,
        "case": case,
        "b": float(metadata_row.b),
        "n": int(metadata_row.n),
        "s": int(metadata_row.s),
        "initial_jet_core_latitude_deg": float(initial["core_latitude"]),
        "eke_peak_nh_area_mass_weighted_m2_s-2": peak_value,
        "eke_peak_time_h": float(hourly_time[peak_index]),
        "eke_50pct_window_start_h": float(hourly_time[start_index]),
        "eke_50pct_window_end_h": float(hourly_time[end_index]),
        "eke_50pct_window_count": int(window_indices.size),
    }])
    return initial_rows, eke_rows, pd.DataFrame(flux_rows), summary


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def checkpoint_paths(ensemble: str, case: str) -> dict[str, Path]:
    checkpoint = CHECKPOINTS / f"{ensemble}__{case}"
    return {
        "initial": checkpoint.with_name(checkpoint.name + "__initial.csv"),
        "eke": checkpoint.with_name(checkpoint.name + "__eke.csv"),
        "flux": checkpoint.with_name(checkpoint.name + "__flux.csv"),
        "summary": checkpoint.with_name(checkpoint.name + "__summary.csv"),
        "done": checkpoint.with_name(checkpoint.name + ".done"),
    }


def save_case_checkpoint(initial: pd.DataFrame, eke: pd.DataFrame, flux: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensemble = str(summary.iloc[0]["ensemble"])
    case = str(summary.iloc[0]["case"])
    paths = checkpoint_paths(ensemble, case)
    for key, frame in [("initial", initial), ("eke", eke), ("flux", flux), ("summary", summary)]:
        atomic_write_csv(frame, paths[key])
    paths["done"].write_text("complete\n")


def load_case_checkpoint(ensemble: str, case: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    paths = checkpoint_paths(ensemble, case)
    required = [paths["initial"], paths["eke"], paths["flux"], paths["summary"], paths["done"]]
    if not all(path.exists() for path in required):
        return None
    return tuple(pd.read_csv(paths[key]) for key in ["initial", "eke", "flux", "summary"])


def compute_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial_path = DATA / "initial_profiles_at_jet_core.csv"
    eke_path = DATA / "eke_nh_area_mass_weighted_timeseries_all90.csv"
    flux_path = DATA / "eddy_flux_profiles_50_80pct_peak_eke_all90.csv"
    summary_path = TABLES / "case_summary_all90.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not force and all(path.exists() for path in [initial_path, eke_path, flux_path, summary_path]):
        return pd.read_csv(initial_path), pd.read_csv(eke_path), pd.read_csv(flux_path), pd.read_csv(summary_path)
    metadata = case_metadata()
    initial_frames, eke_frames, flux_frames, summary_frames = [], [], [], []
    completed = 0
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        checkpoint = None if force else load_case_checkpoint(row.ensemble, row.case)
        if checkpoint is not None:
            initial, eke, flux, summary = checkpoint
            completed += 1
            print(f"[{index:02d}/90] {row.ensemble} {row.case} -- resumed from checkpoint", flush=True)
        else:
            print(f"[{index:02d}/90] {row.ensemble} {row.case} -- calculating", flush=True)
            initial, eke, flux, summary = calculate_case(pd.Series(row._asdict()))
            save_case_checkpoint(initial, eke, flux, summary)
            completed += 1
            print(f"[{index:02d}/90] {row.ensemble} {row.case} -- checkpoint saved ({completed}/90)", flush=True)
        initial_frames.append(initial)
        eke_frames.append(eke)
        flux_frames.append(flux)
        summary_frames.append(summary)
    initial = pd.concat(initial_frames, ignore_index=True)
    eke = pd.concat(eke_frames, ignore_index=True)
    flux = pd.concat(flux_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    atomic_write_csv(initial, initial_path)
    atomic_write_csv(eke, eke_path)
    atomic_write_csv(flux, flux_path)
    atomic_write_csv(summary, summary_path)
    return initial, eke, flux, summary


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def bootstrap_profile(values: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(deterministic_seed("profile", key))
    indices = rng.integers(0, values.shape[0], size=(N_BOOT, values.shape[0]))
    means = values[indices].mean(axis=1)
    return values.mean(axis=0), np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(0.018, 0.98, f"({label})", transform=axis.transAxes, ha="left", va="top", fontsize=11.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8}, zorder=30)


def add_layer_shading(axis: plt.Axes) -> None:
    axis.axhspan(850, 1000, color=LOWER_SHADE, alpha=0.58, zorder=0)
    axis.axhspan(300, 500, color=UPPER_SHADE, alpha=0.54, zorder=0)


def format_pressure_axis(axis: plt.Axes, show_ylabel: bool = True) -> None:
    axis.set_yscale("log")
    axis.set_ylim(PRESSURE_MAX, PRESSURE_MIN)
    axis.set_yticks(PRESSURE_TICKS)
    axis.set_yticklabels([str(value) for value in PRESSURE_TICKS])
    axis.yaxis.set_minor_formatter(NullFormatter())
    if show_ylabel:
        axis.set_ylabel("Pressure (hPa)")
    else:
        axis.tick_params(labelleft=False)
    axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(figure: plt.Figure, stem: str) -> None:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    figure.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def b_legend() -> list[Line2D]:
    return [Line2D([0], [0], color=B_COLORS[b], lw=2.4, label=rf"$b={b:g}$") for b in B_VALUES]


def draw_profile_group(axis: plt.Axes, frame: pd.DataFrame, ensemble: str, value_column: str, scale: float = 1.0, x_limits: tuple[float, float] | None = None, top_wind_axis: bool = False, x_label: str = "") -> None:
    subset = frame[frame.ensemble.eq(ensemble)].copy()
    pressure = np.sort(subset.pressure_hpa.unique())
    for b_value in B_VALUES:
        group = subset[np.isclose(subset.b, b_value)]
        case_pivot = group.pivot(index="case", columns="pressure_hpa", values=value_column).reindex(columns=pressure)
        values = case_pivot.to_numpy(float) * scale
        for row_values in values:
            axis.plot(row_values, pressure, color=B_COLORS[b_value], alpha=0.16, lw=0.5, zorder=2)
        mean, low, high = bootstrap_profile(values, f"{ensemble}-{value_column}-{b_value}")
        axis.fill_betweenx(pressure, low, high, color=B_COLORS[b_value], alpha=0.13, lw=0, zorder=3)
        axis.plot(mean, pressure, color=B_COLORS[b_value], lw=2.3, zorder=4)
    add_layer_shading(axis)
    axis.axvline(0, color="#777777", lw=0.65, ls="--", zorder=1)
    format_pressure_axis(axis, show_ylabel=True)
    if x_limits is not None:
        axis.set_xlim(*x_limits)
    axis.set_xlabel(x_label)
    if top_wind_axis:
        top = axis.twiny()
        wind_subset = subset.pivot(index="case", columns="pressure_hpa", values="initial_zonal_wind_at_jet_core_m_s-1").reindex(columns=pressure)
        for b_value in B_VALUES:
            values = wind_subset.loc[subset[subset.b.eq(b_value)].case.unique()].to_numpy(float)
            mean, _, _ = bootstrap_profile(values, f"wind-{ensemble}-{b_value}")
            top.plot(mean, pressure, color=B_COLORS[b_value], lw=1.35, alpha=0.85, zorder=5)
        top.set_xlim(0, 40)
        top.set_xlabel(r"Initial zonal wind at jet core (m s$^{-1}$)", labelpad=3, fontsize=9.5)
        top.tick_params(axis="x", labelsize=8.0, colors="#444444", pad=1)
        top.spines["top"].set_visible(True)
        top.spines["right"].set_visible(False)
        top.spines["left"].set_visible(False)


def make_initial_wind_figure(initial: pd.DataFrame) -> None:
    pressure_mask = (initial.pressure_hpa >= PRESSURE_MIN) & (initial.pressure_hpa <= PRESSURE_MAX)
    frame = initial[pressure_mask].copy()
    all_values = frame["initial_zonal_wind_at_jet_core_m_s-1"].to_numpy(float)
    x_limits = (min(-2.0, np.nanmin(all_values) - 1), np.nanmax(all_values) + 2)
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 4.25), sharey=True, constrained_layout=False)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.79, wspace=0.20)
    for index, ensemble in enumerate(ENSEMBLE_ORDER):
        axis = axes[index]
        draw_profile_group(axis, frame, ensemble, "initial_zonal_wind_at_jet_core_m_s-1", x_limits=x_limits, x_label=r"Zonal wind (m s$^{-1}$)")
        axis.set_title(ENSEMBLE_LABELS[ensemble], pad=4, fontsize=10.5)
        add_panel_label(axis, chr(97 + index))
        if index == 1:
            axis.tick_params(labelleft=False)
    handles = b_legend() + [Patch(facecolor=LOWER_SHADE, edgecolor="none", alpha=0.58, label="1000–850 hPa"), Patch(facecolor=UPPER_SHADE, edgecolor="none", alpha=0.54, label="500–300 hPa")]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=False, handlelength=2.4, columnspacing=1.0)
    save_figure(figure, "part1_initial_zonal_wind_profiles_45cases")


def make_eady_figure(initial: pd.DataFrame) -> None:
    pressure_mask = (initial.pressure_hpa >= PRESSURE_MIN) & (initial.pressure_hpa <= PRESSURE_MAX)
    frame = initial[pressure_mask].copy()
    all_values = frame["initial_eady_growth_rate_day-1"].to_numpy(float)
    x_limits = (0.0, np.nanmax(all_values) * 1.08)
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 4.25), sharey=True, constrained_layout=False)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.72, wspace=0.20)
    for index, ensemble in enumerate(ENSEMBLE_ORDER):
        axis = axes[index]
        draw_profile_group(axis, frame, ensemble, "initial_eady_growth_rate_day-1", x_limits=x_limits, top_wind_axis=True, x_label=r"Initial Eady growth rate (day$^{-1}$)")
        axis.set_title("")
        add_panel_label(axis, chr(97 + index))
        if index == 1:
            axis.tick_params(labelleft=False)
    handles = b_legend() + [Patch(facecolor=LOWER_SHADE, edgecolor="none", alpha=0.58, label="1000–850 hPa"), Patch(facecolor=UPPER_SHADE, edgecolor="none", alpha=0.54, label="500–300 hPa")]
    figure.text(0.29, 0.815, ENSEMBLE_LABELS["standard"], ha="center", va="center", fontsize=10.5)
    figure.text(0.78, 0.815, ENSEMBLE_LABELS["u30"], ha="center", va="center", fontsize=10.5)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, frameon=False, handlelength=2.4, columnspacing=1.0)
    save_figure(figure, "part2_initial_eady_growth_rate_45cases")


def make_eke_figure(eke: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 4.25), sharex=True, sharey=True, constrained_layout=False)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.79, wspace=0.20)
    for index, ensemble in enumerate(ENSEMBLE_ORDER):
        axis = axes[index]
        subset = eke[(eke.ensemble.eq(ensemble)) & (eke.time_h >= 144)]
        for b_value in B_VALUES:
            b_subset = subset[np.isclose(subset.b, b_value)]
            for case, case_data in b_subset.groupby("case", sort=False):
                n_value = int(case_data.n.iloc[0])
                axis.plot(case_data.time_h, case_data["eke_nh_area_mass_weighted_m2_s-2"], color=B_COLORS[b_value], ls=N_LINESTYLES[n_value], lw=0.45, alpha=0.22, zorder=2)
            mean_series = b_subset.groupby("time_h", as_index=False)["eke_nh_area_mass_weighted_m2_s-2"].mean()
            axis.plot(mean_series.time_h, mean_series["eke_nh_area_mass_weighted_m2_s-2"], color=B_COLORS[b_value], lw=2.2, zorder=4)
        axis.set_title(ENSEMBLE_LABELS[ensemble], pad=4, fontsize=10.5)
        add_panel_label(axis, chr(97 + index))
        axis.set_xlim(144, 360)
        axis.set_xticks([150, 200, 250, 300, 350])
        axis.grid(True, color="#D9D9D9", lw=0.5, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8.5)
        if index == 0:
            axis.set_ylabel(r"NH area- and mass-weighted EKE (m$^2$ s$^{-2}$)")
        else:
            axis.tick_params(labelleft=False)
        axis.set_xlabel("Time (h)")
    ymax = float(eke["eke_nh_area_mass_weighted_m2_s-2"].max())
    for axis in axes:
        axis.set_ylim(0, 100)
        for hour in range(144, 361, 48):
            axis.axvline(hour, color="#BEBEBE", lw=0.6, ls="--", zorder=0)
    handles = b_legend() + [Line2D([0], [0], color="#444444", lw=1.2, ls=N_LINESTYLES[n], label=rf"$n={n}$") for n in N_VALUES]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False, handlelength=2.1, columnspacing=0.9)
    save_figure(figure, "part3_nh_eke_evolution_45cases_per_ensemble")


def make_flux_figure(flux: pd.DataFrame, initial: pd.DataFrame) -> None:
    pressure_mask = (flux.pressure_hpa >= PRESSURE_MIN) & (flux.pressure_hpa <= PRESSURE_MAX)
    flux = flux[pressure_mask].copy()
    wind = initial[(initial.pressure_hpa >= PRESSURE_MIN) & (initial.pressure_hpa <= PRESSURE_MAX)].copy()
    figure, axes = plt.subplots(3, 2, figsize=(7.55, 8.0), sharey=True, constrained_layout=True)
    panel_index = 0
    for row, (variable, scale, xlabel) in enumerate(FLUX_VARIABLES):
        values_for_limits = flux.loc[flux.variable.eq(variable), "value"].to_numpy(float) * scale
        finite = values_for_limits[np.isfinite(values_for_limits)]
        low = min(float(finite.min()), 0.0)
        high = max(float(finite.max()), 0.0)
        span = high - low
        x_limits = (low - 0.07 * span, high + 0.07 * span) if span > 0 else (-1, 1)
        for column, ensemble in enumerate(ENSEMBLE_ORDER):
            axis = axes[row, column]
            subset = flux[(flux.ensemble.eq(ensemble)) & flux.variable.eq(variable)]
            draw_profile_group(axis, subset.merge(wind[["ensemble", "case", "pressure_hpa", "initial_zonal_wind_at_jet_core_m_s-1"]], on=["ensemble", "case", "pressure_hpa"], how="left"), ensemble, "value", scale=scale, x_limits=x_limits, top_wind_axis=True, x_label=xlabel)
            axis.set_title(ENSEMBLE_LABELS[ensemble], pad=18, fontsize=10.2) if row == 0 else axis.set_title("")
            add_panel_label(axis, chr(97 + panel_index))
            panel_index += 1
            if column == 1:
                axis.tick_params(labelleft=False)
    handles = b_legend() + [Patch(facecolor=LOWER_SHADE, edgecolor="none", alpha=0.58, label="1000–850 hPa"), Patch(facecolor=UPPER_SHADE, edgecolor="none", alpha=0.54, label="500–300 hPa")]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=5, frameon=False, handlelength=2.4, columnspacing=1.0)
    save_figure(figure, "part4_eddy_fluxes_50_80pct_peak_eke_45cases")


def make_ep_flux_figure(flux: pd.DataFrame, initial: pd.DataFrame) -> None:
    """Render the compact two-panel eddy-heat-flux part for the four-part figure."""
    pressure_mask = (flux.pressure_hpa >= PRESSURE_MIN) & (flux.pressure_hpa <= PRESSURE_MAX)
    flux_subset = flux[pressure_mask & flux.variable.eq("ep_flux_vertical")].copy()
    wind = initial[(initial.pressure_hpa >= PRESSURE_MIN) & (initial.pressure_hpa <= PRESSURE_MAX)].copy()
    flux_scale = 1.0e-6
    values = flux_subset.value.to_numpy(float) * flux_scale
    finite = values[np.isfinite(values)]
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    span = high - low
    x_limits = (low - 0.07 * span, high + 0.07 * span)
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 4.25), sharey=True, constrained_layout=False)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.72, wspace=0.20)
    for index, ensemble in enumerate(ENSEMBLE_ORDER):
        axis = axes[index]
        subset = flux_subset[flux_subset.ensemble.eq(ensemble)]
        merged = subset.merge(
            wind[["ensemble", "case", "pressure_hpa", "initial_zonal_wind_at_jet_core_m_s-1"]],
            on=["ensemble", "case", "pressure_hpa"], how="left"
        )
        draw_profile_group(
            axis, merged, ensemble, "value", scale=flux_scale, x_limits=x_limits,
            top_wind_axis=False, x_label=r"$F_p$ ($10^6$ Pa m$^2$ s$^{-2}$)"
        )
        add_panel_label(axis, chr(97 + index))
        if index == 1:
            axis.tick_params(labelleft=False)
    figure.text(0.29, 0.815, ENSEMBLE_LABELS["standard"], ha="center", va="center", fontsize=10.5)
    figure.text(0.78, 0.815, ENSEMBLE_LABELS["u30"], ha="center", va="center", fontsize=10.5)
    handles = b_legend() + [
        Patch(facecolor=LOWER_SHADE, edgecolor="none", alpha=0.58, label="1000–850 hPa"),
        Patch(facecolor=UPPER_SHADE, edgecolor="none", alpha=0.54, label="500–300 hPa"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, frameon=False, handlelength=2.4, columnspacing=1.0)
    save_figure(figure, "part4_ep_flux_vertical_50_80pct_peak_eke_45cases")


def make_combined_eight_panel_figure(initial: pd.DataFrame, eke: pd.DataFrame, flux: pd.DataFrame) -> None:
    """Create the requested four-part, two-column composite with continuous labels (a)–(h)."""
    pressure_mask = (initial.pressure_hpa >= PRESSURE_MIN) & (initial.pressure_hpa <= PRESSURE_MAX)
    initial_frame = initial[pressure_mask].copy()
    all_wind = initial_frame["initial_zonal_wind_at_jet_core_m_s-1"].to_numpy(float)
    wind_limits = (min(-2.0, np.nanmin(all_wind) - 1), np.nanmax(all_wind) + 2)
    all_eady = initial_frame["initial_eady_growth_rate_day-1"].to_numpy(float)
    eady_limits = (0.0, np.nanmax(all_eady) * 1.08)
    flux_mask = (flux.pressure_hpa >= PRESSURE_MIN) & (flux.pressure_hpa <= PRESSURE_MAX) & flux.variable.eq("ep_flux_vertical")
    flux_scale = 1.0e-6
    ep_flux = flux[flux_mask].copy()
    wind = initial_frame[["ensemble", "case", "pressure_hpa", "initial_zonal_wind_at_jet_core_m_s-1"]].copy()
    ep_flux = ep_flux.merge(wind, on=["ensemble", "case", "pressure_hpa"], how="left")
    heat_values = ep_flux.value.to_numpy(float) * flux_scale
    finite = heat_values[np.isfinite(heat_values)]
    heat_low = min(float(finite.min()), 0.0)
    heat_high = max(float(finite.max()), 0.0)
    heat_span = heat_high - heat_low
    heat_limits = (heat_low - 0.07 * heat_span, heat_high + 0.07 * heat_span)

    figure, axes = plt.subplots(4, 2, figsize=(7.35, 13.2), constrained_layout=False)
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.055, top=0.895, hspace=0.72, wspace=0.20)
    labels = list("abcdefgh")

    for column, ensemble in enumerate(ENSEMBLE_ORDER):
        axis = axes[0, column]
        draw_profile_group(axis, initial_frame, ensemble, "initial_zonal_wind_at_jet_core_m_s-1", x_limits=wind_limits, x_label=r"Zonal wind (m s$^{-1}$)")
        add_panel_label(axis, labels[column])
        axis.set_title(ENSEMBLE_LABELS[ensemble], pad=4, fontsize=10.5)

        axis = axes[1, column]
        draw_profile_group(axis, initial_frame, ensemble, "initial_eady_growth_rate_day-1", x_limits=eady_limits, top_wind_axis=True, x_label=r"Initial Eady growth rate (day$^{-1}$)")
        add_panel_label(axis, labels[2 + column])

        axis = axes[2, column]
        subset = eke[eke.ensemble.eq(ensemble)]
        for b_value in B_VALUES:
            b_subset = subset[np.isclose(subset.b, b_value)]
            for case, case_data in b_subset.groupby("case", sort=False):
                n_value = int(case_data.n.iloc[0])
                axis.plot(case_data.time_h, case_data["eke_nh_area_mass_weighted_m2_s-2"], color=B_COLORS[b_value], ls=N_LINESTYLES[n_value], lw=0.45, alpha=0.22, zorder=2)
            mean_series = b_subset.groupby("time_h", as_index=False)["eke_nh_area_mass_weighted_m2_s-2"].mean()
            axis.plot(mean_series.time_h, mean_series["eke_nh_area_mass_weighted_m2_s-2"], color=B_COLORS[b_value], lw=2.2, zorder=4)
        axis.set_xlim(144, 360)
        axis.set_xticks([150, 200, 250, 300, 350])
        axis.grid(True, color="#D9D9D9", lw=0.5, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8.5)
        axis.set_xlabel("Time (h)")
        add_panel_label(axis, labels[4 + column])

        axis = axes[3, column]
        subset = ep_flux[ep_flux.ensemble.eq(ensemble)]
        draw_profile_group(axis, subset, ensemble, "value", scale=flux_scale, x_limits=heat_limits, top_wind_axis=False, x_label=r"$F_p$ ($10^6$ Pa m$^2$ s$^{-2}$)")
        add_panel_label(axis, labels[6 + column])

        if column == 1:
            axes[0, column].tick_params(labelleft=False)
            axes[1, column].tick_params(labelleft=False)
            axes[2, column].tick_params(labelleft=False)
            axes[3, column].tick_params(labelleft=False)

    ymax = float(eke["eke_nh_area_mass_weighted_m2_s-2"].max())
    for axis in axes[2, :]:
        axis.set_ylim(0, 100)
        for hour in range(144, 361, 48):
            axis.axvline(hour, color="#BEBEBE", lw=0.6, ls="--", zorder=0)
    axes[2, 0].set_ylabel(r"NH area- and mass-weighted EKE (m$^2$ s$^{-2}$)")

    handles = b_legend() + [
        Patch(facecolor=LOWER_SHADE, edgecolor="none", alpha=0.58, label="1000–850 hPa"),
        Patch(facecolor=UPPER_SHADE, edgecolor="none", alpha=0.54, label="500–300 hPa"),
        Line2D([0], [0], color="#444444", lw=1.2, ls=N_LINESTYLES[1], label=r"$n=1$"),
        Line2D([0], [0], color="#444444", lw=1.2, ls=N_LINESTYLES[3], label=r"$n=3$"),
        Line2D([0], [0], color="#444444", lw=1.2, ls=N_LINESTYLES[6], label=r"$n=6$"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.945), ncol=8, frameon=False, handlelength=1.8, columnspacing=0.75, fontsize=8.3)
    figure.savefig(FIGURES / "section32_four_part_combined_8panels.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(FIGURES / "section32_four_part_combined_8panels.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def make_contact_sheet() -> None:
    from PIL import Image, ImageDraw, ImageFont
    paths = [FIGURES / name for name in [
        "part1_initial_zonal_wind_profiles_45cases.png",
        "part2_initial_eady_growth_rate_45cases.png",
        "part3_nh_eke_evolution_45cases_per_ensemble.png",
        "part4_ep_flux_vertical_50_80pct_peak_eke_45cases.png",
    ]]
    labels = ["Part 1: Initial zonal wind", "Part 2: Initial Eady growth rate", "Part 3: NH EKE evolution", "Part 4: Vertical EP flux, 50–80% peak EKE"]
    target_width = 900
    margin = 18
    title_height = 38
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    blocks = []
    for path, label in zip(paths, labels):
        image = Image.open(path).convert("RGB")
        scale = target_width / image.width
        resized = image.resize((target_width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
        block = Image.new("RGB", (target_width, resized.height + title_height), "white")
        draw = ImageDraw.Draw(block)
        draw.text((10, 7), label, fill="#222222", font=font)
        block.paste(resized, (0, title_height))
        blocks.append(block)
    total_height = margin + sum(block.height + margin for block in blocks)
    sheet = Image.new("RGB", (target_width + 2 * margin, total_height), "white")
    y = margin
    for block in blocks:
        sheet.paste(block, (margin, y))
        y += block.height + margin
    sheet.save(FIGURES / "section32_four_part_contact_sheet.png", dpi=(120, 120))


def make_combined_figure() -> None:
    """Create one portrait composite containing all four diagnostic parts."""
    from PIL import Image, ImageChops, ImageDraw, ImageFont

    source_names = [
        ("Part 1 — Initial zonal-wind profiles", "part1_initial_zonal_wind_profiles_45cases.png"),
        ("Part 2 — Initial Eady growth rate", "part2_initial_eady_growth_rate_45cases.png"),
        ("Part 3 — Northern Hemisphere EKE evolution", "part3_nh_eke_evolution_45cases_per_ensemble.png"),
        ("Part 4 — Eddy fluxes during the 50–80% EKE-growth interval", "part4_eddy_fluxes_50_80pct_peak_eke_45cases.png"),
    ]
    target_width = 2400
    section_header = 92
    section_gap = 28
    outer_margin = 36
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    header_font = ImageFont.truetype(str(font_path), 42) if font_path.exists() else ImageFont.load_default()

    blocks = []
    for label, filename in source_names:
        image = Image.open(FIGURES / filename).convert("RGB")
        background = Image.new("RGB", image.size, "white")
        difference = ImageChops.difference(image, background).convert("L")
        bbox = difference.point(lambda value: 255 if value > 8 else 0).getbbox()
        if bbox is not None:
            left, top, right, bottom = bbox
            padding = 24
            image = image.crop((max(0, left - padding), max(0, top - padding), min(image.width, right + padding), min(image.height, bottom + padding)))
        scale = target_width / image.width
        image = image.resize((target_width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
        block = Image.new("RGB", (target_width, section_header + image.height), "white")
        draw = ImageDraw.Draw(block)
        draw.text((14, 20), label, fill="#222222", font=header_font)
        block.paste(image, (0, section_header))
        blocks.append(block)

    canvas_height = 2 * outer_margin + sum(block.height for block in blocks) + section_gap * (len(blocks) - 1)
    canvas = Image.new("RGB", (target_width + 2 * outer_margin, canvas_height), "white")
    y = outer_margin
    for block in blocks:
        canvas.paste(block, (outer_margin, y))
        y += block.height + section_gap

    png_path = FIGURES / "section32_four_part_combined.png"
    canvas.save(png_path, dpi=(300, 300))

    figure = plt.figure(figsize=((target_width + 2 * outer_margin) / 300, canvas_height / 300), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(canvas)
    axis.axis("off")
    figure.savefig(FIGURES / "section32_four_part_combined.pdf", facecolor="white", dpi=300)
    plt.close(figure)


def write_methods_and_captions(summary: pd.DataFrame) -> None:
    lines = [
        "# Section 3.2 four-part diagnostic figure package",
        "",
        "## Data scope",
        "Each ensemble contains all 45 combinations of b = 1, 1.5, 2; n = 1, 3, 6; and s = -10, -5, 0, 5, 10 degrees. The two ensemble columns therefore represent 45 cases each and 90 simulations in total.",
        "",
        "## Part 1",
        "Initial zonal-mean zonal-wind profiles are sampled at each case's initial 300-hPa zonal-mean jet-core latitude. Individual cases are shown as faint lines and b-group means as thick lines.",
        "",
        "## Part 2",
        "Initial Eady growth rate is calculated at the same fixed initial jet-core latitude using pressure-coordinate log-pressure derivatives. Pressure decreases upward on a logarithmic axis. The upper horizontal axis shows the corresponding initial zonal-wind profile.",
        "",
        "## Part 3",
        "EKE is the Northern Hemisphere domain-averaged, cosine-latitude area-weighted and pressure-trapezoid mass-weighted eddy kinetic energy. At each time, primes are departures from the instantaneous zonal mean at each pressure and latitude. All 45 cases per ensemble are plotted.",
        "",
        "## Part 4",
        "Eddy-flux profiles use the existing jet-relative +/-15-degree spatial averaging and eddy definitions. For each case, the EKE time series is used to identify the first rising-phase interval from 50% to 80% of that case's maximum NH EKE. Fluxes are averaged only over that case-specific interval. The plotted diagnostics are v-prime T-prime, the baroclinic-conversion proxy, and the pressure-coordinate vertical EP-flux component F_p.",
        "",
        "## Presentation QC",
        "The figures use white backgrounds, no overall suptitle, panel labels, shared pressure axes, consistent b colors, readable two-column layouts, and vector PDF output. No manuscript, Supporting Information, response letter, or Word file was modified.",
        "## Compact four-part composite",
        "The compact composite section32_four_part_combined_8panels.png/.pdf contains four equal two-column parts with continuous labels (a)–(h). The EKE row is plotted only over 144–360 h, matching the earlier Figure 6 convention and excluding the initial low-amplitude initialization interval. The fourth row uses the pressure-coordinate vertical EP flux F_p, averaged over the case-specific 50–80% peak-EKE interval. The complete three-diagnostic eddy-flux figure remains available separately.",
        "",
        "## Proposed captions",
        "### Part 1",
        "Initial zonal-mean zonal-wind profiles sampled at the latitude of each case's initial 300-hPa jet core. Left and right columns show the constant-u0 and constant-Umax ensembles, respectively. Thin lines show all 45 cases in each ensemble; colors denote b, and thick lines show b-group means. Shading marks the 850–700-hPa and 500–300-hPa layers.",
        "",
        "### Part 2",
        "Initial Eady growth-rate profiles at the latitude of the initial 300-hPa jet core. The upper horizontal axis shows the corresponding initial zonal-wind profile. Thin lines show all 45 cases per ensemble and thick colored lines show b-group means; the two ensemble columns use identical pressure and Eady-rate scales.",
        "",
        "### Part 3",
        "Northern Hemisphere domain-averaged EKE evolution for all 45 cases in the constant-u0 and constant-Umax ensembles. EKE is cosine-latitude area-weighted and pressure-trapezoid mass-weighted. Thin lines show individual cases, colors denote b, line styles denote n, and thick lines show b-group means.",
        "",
        "### Part 4",
        "Eddy-flux vertical structure averaged over the first rising-phase interval in which Northern Hemisphere EKE increases from 50% to 80% of its case-specific maximum. Panels show v-prime T-prime, the baroclinic-conversion proxy, and the pressure-coordinate vertical EP-flux component. Thin lines show all 45 cases per ensemble, thick lines show b-group means, and the upper horizontal axis shows the initial jet-core zonal wind.",
        "",
        "## Window summary",
        f"The mean 50–80% window length is {summary.eke_50pct_window_count.mean():.1f} hourly samples; the minimum and maximum are {summary.eke_50pct_window_count.min()} and {summary.eke_50pct_window_count.max()} samples.",
    ]
    (OUTPUT / "methods_and_captions.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    for directory in [FIGURES, TABLES, DATA]:
        directory.mkdir(parents=True, exist_ok=True)
    initial, eke, flux, summary = compute_data(force=arguments.force)
    make_initial_wind_figure(initial)
    make_eady_figure(initial)
    make_eke_figure(eke)
    make_flux_figure(flux, initial)
    make_ep_flux_figure(flux, initial)
    make_combined_eight_panel_figure(initial, eke, flux)
    make_contact_sheet()
    make_combined_figure()
    write_methods_and_captions(summary)
    print("Figures written to", FIGURES)
    print("Data written to", DATA)
    print("Summary written to", TABLES)


if __name__ == "__main__":
    main()
