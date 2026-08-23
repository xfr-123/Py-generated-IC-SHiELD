#!/usr/bin/env python3
"""Extend Figure S14 to the complete constant-u0 and constant-Umax ensembles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr


ROOT = Path("/data/keeling/a/mingfei5/a/data/original")
CURRENT = ROOT / "eddy" / "initial_state_analysis_20260722"
UMAX = ROOT / "eddy" / "controlled_umax30_bns_analysis_20260722"
SIM_ROOT = ROOT / "priority_revision_analysis_20260720" / "simulations" / "umax30_all_bns"
MANIFEST = SIM_ROOT / "umax30_cases.csv"
RESPONSES = UMAX / "tables" / "umax30_all45_headline_responses.csv"
COARSE = UMAX / "coarse5avg"
OUT = ROOT / "paper_revision" / "figure_S14_two_ensembles_20260818"

A_EARTH = 6_371_000.0
OMEGA = 7.2921159e-5
G = 9.80665
RD = 287.05
CP = 1004.0
KAPPA = RD / CP

PREDICTORS = [
    ("low_level_eady_max_day-1", "Low-level Eady\ngrowth"),
    ("u_max_initial_ms", "Initial Umax"),
    ("rossby_penetration_theory_km", "Rossby penetration\nH_R = fL/N"),
    ("positive_shear_layer_depth_hpa", "Positive-shear-\nlayer depth"),
    ("initial_flank_shear_s-1", "Initial flank\nshear"),
    ("barotropic_conversion_positive_ms-2", "Positive BT\nconversion"),
]

STANDARD_RESPONSES = [
    ("peak_eke_m2s-2", "Peak 300-hPa EKE"),
    ("post_onset_awb_fraction", "Post-onset AWB fraction"),
    ("eddy_minus10_coalescence_count", "−10-hPa eddy-SLP\ncoalescence count"),
    ("max_anticyclone_overlap_lifetime_h", "Maximum anticyclone-\noverlap duration"),
]

UMAX_RESPONSES = [
    ("peak_eke_m2s-2", "Peak 300-hPa EKE"),
    ("post_onset_awb_fraction", "Post-onset AWB fraction"),
    ("eddy_minus10_coalescence_count", "−10-hPa eddy-SLP\ncoalescence count"),
    ("anticyclone_overlap_duration_hours", "Maximum anticyclone-\noverlap duration"),
]


def initial_profile(case_dir: Path):
    """Reproduce the original 5-degree, first-hour area-weighted remapping."""
    datasets = [
        xr.open_dataset(
            case_dir / f"atmos_4x_hourly.tile{tile}.nc",
            decode_times=False,
            engine="scipy",
        )
        for tile in range(1, 7)
    ]
    try:
        lat_parts, lon_parts, area_parts = [], [], []
        for tile in range(1, 7):
            with xr.open_dataset(
                case_dir / f"grid_spec.tile{tile}.nc",
                decode_times=False,
                engine="scipy",
            ) as grid_tile:
                lat_parts.append(grid_tile.grid_latt.values.ravel())
                lon_parts.append(np.mod(grid_tile.grid_lont.values, 360.0).ravel())
                area_parts.append(grid_tile.area.values.ravel())

        lat = np.arange(-87.5, 88.0, 5.0)
        lon = np.arange(2.5, 360.0, 5.0)
        native_lat = np.concatenate(lat_parts)
        native_lon = np.concatenate(lon_parts)
        native_area = np.concatenate(area_parts).astype(float)
        lat_index = np.floor((native_lat + 90.0) / 5.0).astype(int)
        lon_index = np.floor(native_lon / 5.0).astype(int)
        valid = (
            (lat_index >= 0)
            & (lat_index < lat.size)
            & (lon_index >= 0)
            & (lon_index < lon.size)
        )
        target = lat_index * lon.size + lon_index
        denominator = np.bincount(
            target[valid],
            weights=native_area[valid],
            minlength=lat.size * lon.size,
        )

        output = {}
        for name in ["u_plev", "t_plev", "h_plev"]:
            tile_values = [
                dataset[name].isel(time=slice(0, 4)).mean("time").values
                for dataset in datasets
            ]
            combined = np.concatenate(
                [value.reshape(value.shape[0], -1) for value in tile_values],
                axis=1,
            )
            mapped_levels = []
            for level in range(combined.shape[0]):
                numerator = np.bincount(
                    target[valid],
                    weights=combined[level, valid] * native_area[valid],
                    minlength=lat.size * lon.size,
                )
                mapped_levels.append(
                    np.divide(
                        numerator,
                        denominator,
                        out=np.full_like(numerator, np.nan),
                        where=denominator > 0,
                    )
                )
            output[name] = np.stack(mapped_levels).reshape(
                combined.shape[0], lat.size, lon.size
            )
        plev = datasets[0].plev.values.astype(float)
        return plev, lat, lon, output
    finally:
        for dataset in datasets:
            dataset.close()


def half_max_width_km(lat: np.ndarray, profile: np.ndarray):
    valid = (lat >= 15.0) & (lat <= 75.0) & np.isfinite(profile)
    indices = np.where(valid)[0]
    peak_index = indices[np.nanargmax(profile[valid])]
    peak = float(profile[peak_index])
    half = 0.5 * peak

    left = float(lat[indices[0]])
    for index in range(peak_index - 1, indices[0] - 1, -1):
        if profile[index] < half <= profile[index + 1]:
            fraction = (half - profile[index]) / (profile[index + 1] - profile[index])
            left = float(lat[index] + fraction * (lat[index + 1] - lat[index]))
            break

    right = float(lat[indices[-1]])
    for index in range(peak_index, indices[-1]):
        if profile[index] >= half > profile[index + 1]:
            fraction = (half - profile[index]) / (profile[index + 1] - profile[index])
            right = float(lat[index] + fraction * (lat[index + 1] - lat[index]))
            break

    half_width = 0.5 * (right - left) * np.pi / 180.0 * A_EARTH / 1000.0
    return half_width, float(lat[peak_index])


def initial_metrics(case: str, b_value: float):
    plev, lat, _, initial = initial_profile(SIM_ROOT / case)
    nlat = (lat >= 15.0) & (lat <= 75.0)
    k850, k700, k300 = [int(np.argmin(abs(plev - level))) for level in (850, 700, 300)]

    zonal_u = initial["u_plev"].mean(axis=-1)
    zonal_t = initial["t_plev"].mean(axis=-1)
    zonal_h = initial["h_plev"].mean(axis=-1)
    u300 = zonal_u[k300]
    jet_index = np.where(nlat)[0][np.nanargmax(u300[nlat])]
    u_max_initial = float(np.nanmax(zonal_u[:, nlat]))
    dudy = np.gradient(u300, A_EARTH * np.deg2rad(lat), edge_order=2)
    flank_shear = float(np.nanmax(np.abs(dudy[nlat])))

    pressure = plev[:, None]
    theta = zonal_t * (1000.0 / pressure) ** KAPPA
    dz = np.maximum(zonal_h[k700] - zonal_h[k850], 100.0)
    dtheta = theta[k700] - theta[k850]
    buoyancy_frequency = np.sqrt(
        np.maximum(G / theta[k850] * dtheta / dz, 1.0e-10)
    )
    coriolis = 2.0 * OMEGA * np.sin(np.deg2rad(lat))
    eady = (
        0.31
        * np.abs(coriolis)
        * np.abs(zonal_u[k700] - zonal_u[k850])
        / dz
        / buoyancy_frequency
        * 86400.0
    )
    low_level_eady = float(np.nanmax(eady[nlat]))

    log_pressure = np.log(plev)
    du_dlogp = np.gradient(zonal_u, log_pressure, axis=0, edge_order=1)
    dtheta_dlogp = np.gradient(theta, log_pressure, axis=0, edge_order=1)
    dudz = -G / (RD * zonal_t) * du_dlogp
    dtheta_dz = -G / (RD * zonal_t) * dtheta_dlogp
    n_squared = G / theta * dtheta_dz
    full_n = np.sqrt(np.where(n_squared > 1.0e-6, n_squared, np.nan))
    half_width_km, jet_lat = half_max_width_km(lat, u300)
    jet_index_rossby = int(np.argmin(np.abs(lat - jet_lat)))
    troposphere = (plev >= 300.0) & (plev <= 850.0)
    mean_n = float(np.nanmean(full_n[troposphere, jet_index_rossby]))
    rossby_depth = (
        abs(coriolis[jet_index_rossby]) * half_width_km * 1000.0 / mean_n / 1000.0
    )

    return {
        "low_level_eady_max_day-1": low_level_eady,
        "u_max_initial_ms": u_max_initial,
        "rossby_penetration_theory_km": float(rossby_depth),
        "positive_shear_layer_depth_hpa": float(
            1000.0 * (1.0 - np.exp(-float(b_value) / np.sqrt(2.0)))
        ),
        "initial_flank_shear_s-1": flank_shear,
    }


def positive_bt_conversion(case: str, onset: int, peak: int) -> float:
    path = COARSE / f"{case}.nc"
    with h5py.File(path, "r") as dataset:
        plev = dataset["plev"][:].astype(float)
        lat = dataset["latitude"][:].astype(float)
        k300 = int(np.argmin(np.abs(plev - 300.0)))
        nlat = (lat >= 15.0) & (lat <= 75.0)
        u = dataset["u_plev"][:, k300, nlat, :].astype(float)
        v = dataset["v_plev"][:, k300, nlat, :].astype(float)

    u_mean = u.mean(axis=-1, keepdims=True)
    v_mean = v.mean(axis=-1, keepdims=True)
    uv_flux = ((u - u_mean) * (v - v_mean)).mean(axis=-1)
    y = A_EARTH * np.deg2rad(lat[nlat])
    mean_shear = np.gradient(u_mean[..., 0], y, axis=-1, edge_order=2)
    conversion = -uv_flux * mean_shear
    start = max(0, int(onset))
    stop = min(conversion.shape[0], max(start + 1, int(peak) + 1))
    return float(np.nanmean(np.maximum(conversion[start:stop], 0.0)))


def process_umax_record(record: dict) -> dict:
    metrics = initial_metrics(str(record["case"]), float(record["b"]))
    metrics["barotropic_conversion_positive_ms-2"] = positive_bt_conversion(
        str(record["case"]),
        int(record["growth_onset_index"]),
        int(record["peak_eke_time_index"]),
    )
    return {**record, **metrics}


def build_umax_table() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    response = pd.read_csv(RESPONSES)
    merged = manifest.merge(response, on=["case", "b", "n", "s"], how="inner", validate="one_to_one")
    if len(merged) != 45:
        raise RuntimeError(f"Expected 45 matched Umax cases, found {len(merged)}")

    records = merged.to_dict("records")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_umax_record, record): record["case"] for record in records}
        for number, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"{number:02d}/45 {row['case']}", flush=True)
    return pd.DataFrame(rows).sort_values(["b", "n", "s"]).reset_index(drop=True)


def correlations(data: pd.DataFrame, responses, ensemble: str) -> pd.DataFrame:
    rows = []
    for predictor, predictor_label in PREDICTORS:
        for response, response_label in responses:
            x = data[predictor].to_numpy(dtype=float)
            y = data[response].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 4 and np.unique(x[valid]).size > 1 and np.unique(y[valid]).size > 1:
                rho, p_value = spearmanr(x[valid], y[valid])
            else:
                rho, p_value = np.nan, np.nan
            rows.append(
                {
                    "ensemble": ensemble,
                    "predictor": predictor,
                    "predictor_label": predictor_label.replace("\n", " "),
                    "response": response,
                    "response_label": response_label.replace("\n", " "),
                    "n": int(valid.sum()),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def draw_heatmap(standard_stats: pd.DataFrame, umax_stats: pd.DataFrame):
    predictor_order = [name for name, _ in PREDICTORS]
    panels = [
        (standard_stats, STANDARD_RESPONSES, "(a) Constant u₀ ensemble"),
        (umax_stats, UMAX_RESPONSES, "(b) Constant Umax = 30 m s⁻¹ ensemble"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.15), sharey=True)
    image = None
    for axis, (stats, responses, title) in zip(axes, panels):
        response_order = [name for name, _ in responses]
        matrix = stats.pivot(index="response", columns="predictor", values="spearman_rho").loc[
            response_order, predictor_order
        ]
        image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                color = "white" if np.isfinite(value) and abs(value) >= 0.58 else "black"
                axis.text(
                    column,
                    row,
                    "NA" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )
        axis.set_xticks(
            np.arange(len(PREDICTORS)),
            [label for _, label in PREDICTORS],
            rotation=32,
            ha="right",
            rotation_mode="anchor",
            fontsize=10,
        )
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
        axis.set_xticks(np.arange(-0.5, len(PREDICTORS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(responses), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)

    axes[0].set_yticks(
        np.arange(len(STANDARD_RESPONSES)),
        [label for _, label in STANDARD_RESPONSES],
        fontsize=10.5,
    )
    axes[0].set_ylabel("Selected diagnostic response", fontsize=11)
    axes[1].tick_params(labelleft=False)

    color_axis = figure.add_axes([0.25, 0.065, 0.52, 0.035])
    color_bar = figure.colorbar(image, cax=color_axis, orientation="horizontal")
    color_bar.set_label("Spearman correlation (ρ)", fontsize=10.5)
    color_bar.set_ticks([-1, -0.5, 0, 0.5, 1])

    figure.subplots_adjust(left=0.235, right=0.985, bottom=0.29, top=0.91, wspace=0.10)
    figure.savefig(OUT / "Figure_S14_constant_u0_vs_constant_Umax.png", dpi=600, facecolor="white")
    figure.savefig(OUT / "Figure_S14_constant_u0_vs_constant_Umax.pdf", facecolor="white")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    standard = pd.read_csv(CURRENT / "tables" / "archive_initial_predictors_and_responses.csv")
    umax = build_umax_table()
    standard_stats = correlations(standard, STANDARD_RESPONSES, "constant_u0")
    umax_stats = correlations(umax, UMAX_RESPONSES, "constant_Umax_30_ms")
    combined_stats = pd.concat([standard_stats, umax_stats], ignore_index=True)

    umax.to_csv(OUT / "constant_Umax_45case_predictors_and_responses.csv", index=False)
    combined_stats.to_csv(OUT / "Figure_S14_two_ensemble_spearman_long.csv", index=False)
    for name, stats in [("constant_u0", standard_stats), ("constant_Umax_30_ms", umax_stats)]:
        stats.pivot(index="response_label", columns="predictor_label", values="spearman_rho").to_csv(
            OUT / f"Figure_S14_{name}_correlation_matrix.csv"
        )
    draw_heatmap(standard_stats, umax_stats)

    standard_reference = pd.read_csv(CURRENT / "tables" / "predictor_response_spearman.csv")
    standard_reference = standard_reference[standard_reference["dataset"].eq("original_45_case_ensemble")]
    check = standard_stats.merge(
        standard_reference[["predictor", "response", "spearman_rho"]],
        on=["predictor", "response"],
        suffixes=("_new", "_reference"),
    )
    check["absolute_difference"] = abs(check["spearman_rho_new"] - check["spearman_rho_reference"])
    check.to_csv(OUT / "standard_panel_reproduction_check.csv", index=False)
    print(f"maximum standard-panel reproduction difference = {check.absolute_difference.max():.3e}")
    print(umax_stats[["predictor_label", "response_label", "n", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
