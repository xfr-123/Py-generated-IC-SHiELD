#!/usr/bin/env python3
"""Thorncroft-style EP-flux sections for Reviewer 1 comment 6."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "paper_revision" / "supplemental_analysis"
FIGDIR = OUTDIR / "figures"
CASES = [
    ("BCwave_b2n3s-10", -10),
    ("BCwave_b2n3", 0),
    ("BCwave_b2n3s10", 10),
]
TARGET_DAYS = (10.0, 12.0, 14.0)
WINDOW_HOURS = 24.0
LATITUDE_LOAD_BOUNDS = (15.0, 85.0)
LATITUDE_PLOT_BOUNDS = (20.0, 80.0)
PRESSURE_PLOT_BOUNDS = (100.0, 1000.0)
PRESSURE_TICKS = (1000, 850, 700, 500, 300, 200, 100)

A_EARTH = 6_371_000.0
OMEGA = 7.2921159e-5
GRAVITY = 9.80665
RD = 287.05
CP = 1004.0
KAPPA = RD / CP
P0_HPA = 1000.0
SECONDS_PER_DAY = 86_400.0

WIND_CONTOUR_LEVELS = (-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
QUIVER_LATITUDE_STEP = 8
QUIVER_PRESSURE_RANGE_HPA = (100.0, 925.0)
QUIVER_PRESSURE_ROW_COUNT = 8
QUIVER_REFERENCE_CUBIC_METERS = 1.0e15
QUIVER_SCALE_CUBIC_METERS_PER_INCH = 9.0e15
DIVERGENCE_COLOR_LIMIT = 12.0


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - target)))


def log_uniform_pressure_positions(pressure_hpa: np.ndarray) -> np.ndarray:
    """Select native pressure rows at near-uniform spacing on a log axis."""
    pressure = np.asarray(pressure_hpa, dtype=float)
    targets = np.geomspace(
        QUIVER_PRESSURE_RANGE_HPA[0],
        QUIVER_PRESSURE_RANGE_HPA[1],
        QUIVER_PRESSURE_ROW_COUNT,
    )
    positions = np.array(
        [nearest_index(pressure, target) for target in targets], dtype=int
    )
    return np.unique(positions)


def time_window_indices(time_hours: np.ndarray, target_day: float) -> np.ndarray:
    target_hour = 24.0 * target_day
    half_width = WINDOW_HOURS / 2.0
    return np.flatnonzero((time_hours >= target_hour - half_width) & (time_hours < target_hour + half_width))


def compute_ep_section(
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    pressure_hpa: np.ndarray,
    latitude_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return 24-h means of instantaneous pressure-coordinate EP diagnostics.

    The Cartesian EP-vector components follow Edmon et al. (1980) and the
    implementation documented by Jucker (2021):

      F_phi* = -[u'v']
      F_p*   = f [v'theta']/d[theta]/dp

    The zonal eddy covariances, static stability, EP vectors, and divergence
    are evaluated at every hourly output before the 24-h mean is formed.
    Here p is in hPa, so F_p* is in hPa m s-2. EP-flux divergence divided by
    a cos(phi) is reported as a zonal acceleration in m s-1 day-1.
    """
    pressure = np.asarray(pressure_hpa, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    potential_temperature = temperature * (
        P0_HPA / pressure[None, :, None, None]
    ) ** KAPPA

    u_mean = np.nanmean(u, axis=-1)
    v_mean = np.nanmean(v, axis=-1)
    theta_mean = np.nanmean(potential_temperature, axis=-1)

    u_prime = u - u_mean[..., None]
    v_prime = v - v_mean[..., None]
    theta_prime = potential_temperature - theta_mean[..., None]
    uv = np.nanmean(u_prime * v_prime, axis=-1)
    vtheta = np.nanmean(v_prime * theta_prime, axis=-1)

    theta_pressure_gradient = np.gradient(
        theta_mean, pressure, axis=1, edge_order=2
    )
    stable = np.abs(theta_pressure_gradient) > 1.0e-8
    heat_thickness_flux = np.divide(
        vtheta,
        theta_pressure_gradient,
        out=np.full_like(vtheta, np.nan, dtype=float),
        where=stable,
    )

    phi = np.deg2rad(latitude)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    coriolis = 2.0 * OMEGA * sinphi

    ep_meridional_hourly = -uv
    ep_vertical_hourly = coriolis[None, None, :] * heat_thickness_flux

    horizontal_divergence_hourly = -np.gradient(
        uv * cosphi[None, None, :] ** 2,
        phi,
        axis=2,
        edge_order=2,
    ) / (A_EARTH * cosphi[None, None, :] ** 2)
    vertical_divergence_hourly = np.gradient(
        ep_vertical_hourly,
        pressure,
        axis=1,
        edge_order=2,
    )

    horizontal_divergence = np.nanmean(horizontal_divergence_hourly, axis=0)
    vertical_divergence = np.nanmean(vertical_divergence_hourly, axis=0)
    return {
        "zonal_mean_u": np.nanmean(u_mean, axis=0),
        "ep_flux_meridional_cartesian": np.nanmean(
            ep_meridional_hourly, axis=0
        ),
        "ep_flux_vertical_cartesian": np.nanmean(ep_vertical_hourly, axis=0),
        "ep_flux_divergence_acceleration": (
            horizontal_divergence + vertical_divergence
        )
        * SECONDS_PER_DAY,
        "ep_flux_horizontal_divergence_acceleration": horizontal_divergence
        * SECONDS_PER_DAY,
        "ep_flux_vertical_divergence_acceleration": vertical_divergence
        * SECONDS_PER_DAY,
        "eddy_momentum_flux_uv": np.nanmean(uv, axis=0),
        "eddy_heat_flux_vtheta": np.nanmean(vtheta, axis=0),
        "zonal_mean_potential_temperature": np.nanmean(theta_mean, axis=0),
    }


def analyze_case(case: str, shift: int) -> xr.Dataset:
    with xr.open_dataset(ROOT / f"{case}.nc", decode_times=False) as dataset:
        time_hours = dataset["time"].values.astype(float)
        pressure_hpa = dataset["plev"].values.astype(float)
        latitude_all = dataset["grid_yt"].values.astype(float)
        latitude_indices = np.flatnonzero(
            (latitude_all >= LATITUDE_LOAD_BOUNDS[0])
            & (latitude_all <= LATITUDE_LOAD_BOUNDS[1])
        )
        latitude_deg = latitude_all[latitude_indices]

        results: dict[str, list[np.ndarray]] = {}
        window_start = []
        window_end = []
        sample_count = []
        for target_day in TARGET_DAYS:
            indices = time_window_indices(time_hours, target_day)
            if indices.size != int(WINDOW_HOURS):
                raise ValueError(
                    f"{case} day {target_day:g}: expected {WINDOW_HOURS:g} hourly samples, "
                    f"found {indices.size}"
                )
            fields = {
                name: dataset[name]
                .isel(time=indices, grid_yt=latitude_indices)
                .load()
                .values.astype(float)
                for name in ("u_plev", "v_plev", "t_plev")
            }
            section = compute_ep_section(
                fields["u_plev"],
                fields["v_plev"],
                fields["t_plev"],
                pressure_hpa,
                latitude_deg,
            )
            for name, values in section.items():
                results.setdefault(name, []).append(values)
            window_start.append(float(time_hours[indices[0]] / 24.0))
            window_end.append(float(time_hours[indices[-1]] / 24.0))
            sample_count.append(int(indices.size))

    data_vars = {
        name: (("target_day", "plev", "latitude"), np.stack(values, axis=0))
        for name, values in results.items()
    }
    output = xr.Dataset(
        data_vars=data_vars,
        coords={
            "target_day": np.asarray(TARGET_DAYS, dtype=float),
            "plev": pressure_hpa,
            "latitude": latitude_deg,
            "window_start_day": ("target_day", np.asarray(window_start)),
            "window_end_day": ("target_day", np.asarray(window_end)),
            "sample_count": ("target_day", np.asarray(sample_count, dtype=int)),
        },
        attrs={
            "case": case,
            "s_deg": shift,
            "window_definition": "24-hour mean: target day minus 12 h inclusive to target day plus 12 h exclusive",
            "ep_flux_definition": "Edmon et al. (1980) pressure-coordinate quasigeostrophic EP flux with total zonal eddies",
            "ep_flux_vector_scaling": "Cartesian components; plotted with Edmon et al. geometric scaling following Jucker (2021)",
        },
    )
    output["zonal_mean_u"].attrs["units"] = "m s-1"
    output["ep_flux_meridional_cartesian"].attrs["units"] = "m2 s-2"
    output["ep_flux_vertical_cartesian"].attrs["units"] = "hPa m s-2"
    for name in (
        "ep_flux_divergence_acceleration",
        "ep_flux_horizontal_divergence_acceleration",
        "ep_flux_vertical_divergence_acceleration",
    ):
        output[name].attrs["units"] = "m s-1 day-1"
    output["eddy_momentum_flux_uv"].attrs["units"] = "m2 s-2"
    output["eddy_heat_flux_vtheta"].attrs["units"] = "K m s-1"
    output["zonal_mean_potential_temperature"].attrs["units"] = "K"
    return output


def arrow_components(
    latitude_deg: np.ndarray,
    pressure_hpa: np.ndarray,
    ep_meridional_cartesian: np.ndarray,
    ep_vertical_cartesian: np.ndarray,
    axes_width_inches: float,
    axes_height_inches: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale EP vectors for a latitude-log-pressure plot.

    This implements the Edmon et al. (1980) area-preserving vector scaling
    used in Jucker (2021), adapted to a logarithmic pressure axis.
    """
    latitude = np.asarray(latitude_deg, dtype=float)
    pressure = np.asarray(pressure_hpa, dtype=float)
    cosphi = np.cos(np.deg2rad(latitude))[None, :]

    physical_meridional = (
        2.0
        * np.pi
        / GRAVITY
        * cosphi**2
        * A_EARTH**2
        * ep_meridional_cartesian
    )
    physical_vertical = (
        2.0
        * np.pi
        / GRAVITY
        * cosphi**2
        * A_EARTH**3
        * ep_vertical_cartesian
    )

    latitude_span = LATITUDE_PLOT_BOUNDS[1] - LATITUDE_PLOT_BOUNDS[0]
    log_pressure_span = np.log(
        PRESSURE_PLOT_BOUNDS[1] / PRESSURE_PLOT_BOUNDS[0]
    )
    latitude_distance = axes_width_inches / latitude_span * 180.0 / np.pi
    pressure_distance = (
        -axes_height_inches / pressure[:, None] / log_pressure_span
    )
    return (
        physical_meridional * latitude_distance,
        physical_vertical * pressure_distance,
    )


def plot_figure(datasets: dict[str, xr.Dataset]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        len(TARGET_DAYS),
        len(CASES),
        figsize=(7.45, 8.55),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    norm = TwoSlopeNorm(
        vmin=-DIVERGENCE_COLOR_LIMIT,
        vcenter=0.0,
        vmax=DIVERGENCE_COLOR_LIMIT,
    )
    levels = np.linspace(-DIVERGENCE_COLOR_LIMIT, DIVERGENCE_COLOR_LIMIT, 21)
    panel_labels = iter([f"({chr(97 + index)})" for index in range(9)])
    contour_image = None
    quivers = []

    fig.canvas.draw()
    for row, target_day in enumerate(TARGET_DAYS):
        for column, (case, shift) in enumerate(CASES):
            dataset = datasets[case]
            pressure = dataset["plev"].values.astype(float)
            latitude = dataset["latitude"].values.astype(float)
            pressure_mask = (
                (pressure >= PRESSURE_PLOT_BOUNDS[0])
                & (pressure <= PRESSURE_PLOT_BOUNDS[1])
            )
            latitude_mask = (
                (latitude >= LATITUDE_PLOT_BOUNDS[0])
                & (latitude <= LATITUDE_PLOT_BOUNDS[1])
            )
            p = pressure[pressure_mask]
            lat = latitude[latitude_mask]
            day_selection = dataset.sel(target_day=target_day)

            def select(name: str) -> np.ndarray:
                return day_selection[name].values[np.ix_(pressure_mask, latitude_mask)]

            divergence = select("ep_flux_divergence_acceleration")
            divergence = np.where(p[:, None] >= 975.0, np.nan, divergence)
            zonal_mean_u = select("zonal_mean_u")
            ep_meridional = select("ep_flux_meridional_cartesian")
            ep_vertical = select("ep_flux_vertical_cartesian")

            ax = axes[row, column]
            contour_image = ax.contourf(
                lat,
                p,
                divergence,
                levels=levels,
                cmap="RdBu_r",
                norm=norm,
                extend="both",
            )
            wind_contours = ax.contour(
                lat,
                p,
                zonal_mean_u,
                levels=WIND_CONTOUR_LEVELS,
                colors="0.23",
                linewidths=0.68,
            )
            ax.clabel(
                wind_contours,
                levels=[0.0, 20.0, 40.0],
                inline=True,
                fontsize=6.5,
                fmt="%g",
            )

            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            arrow_u, arrow_v = arrow_components(
                lat,
                p,
                ep_meridional,
                ep_vertical,
                bbox.width,
                bbox.height,
            )
            pressure_positions = log_uniform_pressure_positions(p)
            latitude_positions = np.arange(1, lat.size, QUIVER_LATITUDE_STEP)
            row_indices, column_indices = np.ix_(pressure_positions, latitude_positions)
            quiver = ax.quiver(
                lat[latitude_positions],
                p[pressure_positions],
                arrow_u[row_indices, column_indices],
                arrow_v[row_indices, column_indices],
                angles="uv",
                scale_units="inches",
                scale=QUIVER_SCALE_CUBIC_METERS_PER_INCH,
                color="black",
                width=0.0028,
                headwidth=4.0,
                headlength=5.0,
                headaxislength=4.3,
                pivot="tail",
                zorder=4,
            )
            quivers.append(quiver)

            ax.set_yscale("log")
            ax.set_ylim(PRESSURE_PLOT_BOUNDS[1], PRESSURE_PLOT_BOUNDS[0])
            ax.set_xlim(*LATITUDE_PLOT_BOUNDS)
            ax.yaxis.set_major_locator(FixedLocator(PRESSURE_TICKS))
            ax.yaxis.set_major_formatter(
                FixedFormatter([str(value) for value in PRESSURE_TICKS])
            )
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
            ax.grid(color="0.88", linewidth=0.45, alpha=0.65)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.015,
                0.975,
                next(panel_labels),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11.0,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
                zorder=5,
            )
            if row == 0:
                ax.set_title(rf"$s={shift}^\circ$")
            if column == 0:
                ax.set_ylabel(f"Day {target_day:g}\nPressure (hPa)")
            if row == len(TARGET_DAYS) - 1:
                ax.set_xlabel("Latitude (°N)")

    colorbar = fig.colorbar(
        contour_image,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.035,
        aspect=36,
    )
    colorbar.set_label(
        r"EP-flux divergence acceleration (m s$^{-1}$ day$^{-1}$)"
    )
    axes[0, 0].quiverkey(
        quivers[0],
        0.56,
        0.91,
        QUIVER_REFERENCE_CUBIC_METERS,
        label=rf"{QUIVER_REFERENCE_CUBIC_METERS:.0e} m$^3$",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 8.0},
    )

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGDIR / "r1_6_ep_flux_evolution.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def finite_difference_u_tendency(
    case: str,
    target_day: float,
    pressure_hpa: np.ndarray,
    latitude_deg: np.ndarray,
) -> np.ndarray:
    with xr.open_dataset(ROOT / f"{case}.nc", decode_times=False) as dataset:
        time_hours = dataset["time"].values.astype(float)
        start_index = nearest_index(time_hours, target_day * 24.0 - 12.0)
        end_index = nearest_index(time_hours, target_day * 24.0 + 12.0)
        start = (
            dataset["u_plev"]
            .isel(time=start_index)
            .mean("grid_xt")
            .interp(plev=pressure_hpa, grid_yt=latitude_deg)
            .load()
            .values.astype(float)
        )
        end = (
            dataset["u_plev"]
            .isel(time=end_index)
            .mean("grid_xt")
            .interp(plev=pressure_hpa, grid_yt=latitude_deg)
            .load()
            .values.astype(float)
        )
    elapsed_days = (time_hours[end_index] - time_hours[start_index]) / 24.0
    return (end - start) / elapsed_days


def panel_metrics(datasets: dict[str, xr.Dataset]) -> pd.DataFrame:
    rows = []
    for case, shift in CASES:
        dataset = datasets[case]
        pressure = dataset["plev"].values.astype(float)
        latitude = dataset["latitude"].values.astype(float)
        mask_pressure = (
            (pressure >= PRESSURE_PLOT_BOUNDS[0])
            & (pressure <= 925.0)
        )
        mask_latitude = (
            (latitude >= LATITUDE_PLOT_BOUNDS[0])
            & (latitude <= LATITUDE_PLOT_BOUNDS[1])
        )
        p = pressure[mask_pressure]
        lat = latitude[mask_latitude]
        for target_day in TARGET_DAYS:
            selection = dataset.sel(target_day=target_day)
            acceleration = selection["ep_flux_divergence_acceleration"].values[
                np.ix_(mask_pressure, mask_latitude)
            ]
            vertical_flux = selection["ep_flux_vertical_cartesian"].values[
                np.ix_(mask_pressure, mask_latitude)
            ]
            wind_tendency = finite_difference_u_tendency(case, target_day, p, lat)
            finite = np.isfinite(acceleration) & np.isfinite(wind_tendency)
            correlation = (
                float(np.corrcoef(acceleration[finite], wind_tendency[finite])[0, 1])
                if np.count_nonzero(finite) >= 3
                else np.nan
            )
            convergence_index = np.unravel_index(
                np.nanargmin(acceleration), acceleration.shape
            )
            divergence_index = np.unravel_index(
                np.nanargmax(acceleration), acceleration.shape
            )
            upward_index = np.unravel_index(
                np.nanargmin(vertical_flux), vertical_flux.shape
            )
            rows.append(
                {
                    "case": case,
                    "s_deg": shift,
                    "target_day": target_day,
                    "window_start_day": float(
                        selection["window_start_day"].values
                    ),
                    "window_end_day": float(selection["window_end_day"].values),
                    "sample_count": int(selection["sample_count"].values),
                    "minimum_ep_acceleration_ms_per_day": float(
                        acceleration[convergence_index]
                    ),
                    "minimum_ep_acceleration_latitude_deg": float(
                        lat[convergence_index[1]]
                    ),
                    "minimum_ep_acceleration_pressure_hpa": float(
                        p[convergence_index[0]]
                    ),
                    "maximum_ep_acceleration_ms_per_day": float(
                        acceleration[divergence_index]
                    ),
                    "maximum_ep_acceleration_latitude_deg": float(
                        lat[divergence_index[1]]
                    ),
                    "maximum_ep_acceleration_pressure_hpa": float(
                        p[divergence_index[0]]
                    ),
                    "strongest_upward_ep_flux_hpa_m_s2": float(
                        vertical_flux[upward_index]
                    ),
                    "strongest_upward_ep_flux_latitude_deg": float(
                        lat[upward_index[1]]
                    ),
                    "strongest_upward_ep_flux_pressure_hpa": float(
                        p[upward_index[0]]
                    ),
                    "ep_acceleration_vs_zonal_wind_tendency_correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def write_method_note(metrics: pd.DataFrame, png_path: Path, pdf_path: Path) -> Path:
    note_path = OUTDIR / "r1_6_ep_flux_evolution_method_and_interpretation.md"
    with note_path.open("w", encoding="utf-8") as stream:
        stream.write("# Thorncroft-style EP-flux evolution diagnostic\n\n")
        stream.write("## Purpose\n\n")
        stream.write(
            "The previous zonal-mean-jet figure documented that the Eulerian zonal mean evolves, "
            "but it did not identify how transient eddies propagate or where they accelerate and "
            "decelerate the zonal flow. The new latitude-pressure sections add EP-flux vectors and "
            "EP-flux-divergence acceleration, providing a direct wave-activity and wave-mean-flow "
            "diagnostic in the style of Edmon et al. (1980) and Thorncroft et al. (1993).\n\n"
        )
        stream.write("## Definition\n\n")
        stream.write(
            "The calculation uses hourly total zonal eddies and 24-hour means centered on Days 10, "
            "12, and 14. The hourly pressure-coordinate diagnostics are calculated first and then averaged. The Cartesian vector components are\n\n"
        )
        stream.write(
            "$$F_\\phi^*=-[u'v'],\\qquad "
            "F_p^*=f\\frac{[v'\\theta']}{\\partial[\\theta]/\\partial p}.$$\n\n"
        )
        stream.write(
            "The 975- and 1000-hPa divergence values are not shaded or included in the panel-extrema table because the vertical derivative is one-sided at the lower pressure boundary; zonal-mean wind contours remain shown to 1000 hPa, while the lowest displayed EP-vector row is 925 hPa.\n\n"
        )
        stream.write(
            "The shaded field is $\\nabla\\cdot\\mathbf{F}/(a\\cos\\phi)$, expressed as a zonal-mean "
            "acceleration in m s$^{-1}$ day$^{-1}$. Positive shading denotes westerly acceleration; "
            "negative shading denotes westerly deceleration (EP-flux convergence and wave-activity "
            "absorption). Arrow geometry uses the Edmon et al. area-preserving scaling as described "
            "by Jucker (2021), with one common vector scale across all panels. The shared color scale is fixed at +/-12 m s$^{-1}$ day$^{-1}$; colorbar extensions denote stronger values, while the unclipped values are retained in the NetCDF and metrics table.\n\n"
        )
        stream.write(
            "For readability, vectors are subsampled at approximately uniform display spacing in "
            "log pressure from 100 to 925 hPa and at approximately 4-degree latitude intervals. "
            "This changes only vector density, not the underlying EP-flux calculation or common "
            "vector scale.\n\n"
        )
        stream.write("## Why this addresses the reviewer comment\n\n")
        stream.write(
            "- Upward arrows identify propagation of wave activity away from the lower-tropospheric baroclinic source.\n"
        )
        stream.write(
            "- Meridional arrow turning distinguishes poleward and equatorward redistribution of transient wave activity.\n"
        )
        stream.write(
            "- EP-flux convergence and divergence show where the eddies deposit or remove zonal momentum and therefore modify the contemporaneous zonal-mean jet.\n"
        )
        stream.write(
            "- Zonal-mean wind contours place the propagation and forcing directly relative to the evolving jet and waveguide.\n\n"
        )
        stream.write("## Interpretation and limitation\n\n")
        stream.write(
            "Across the selected cases, the forcing generally strengthens from Day 10 into the nonlinear stage, while the latitude-pressure pattern changes strongly with the imposed meridional shift. The $s=10^\circ$ case reaches the strongest convergence on Day 12 and then reorganizes and weakens by Day 14; the $s=0^\circ$ case continues strengthening through Day 14. This timing is consistent with the earlier nonlinear development of the positive-shift case. "
            "This supports describing the surface-pressure and jet changes as waves superposed on, "
            "and interacting with, an evolving zonal mean rather than as isolated vortices obeying a "
            "binary angular-momentum argument.\n\n"
        )
        stream.write(
            "This is a quasigeostrophic pressure-coordinate diagnostic within an Eulerian zonal-mean "
            "framework. It does not remove the definition dependence of the mean/eddy partition at "
            "large amplitude, and EP-flux divergence is not expected to close the full zonal-wind "
            "tendency because residual circulation, friction, and non-QG terms are omitted.\n\n"
        )
        stream.write("## Panel validation metrics\n\n")
        stream.write("```text\n")
        stream.write(metrics.round(3).to_string(index=False))
        stream.write("\n```\n\n")
        stream.write("## References\n\n")
        stream.write(
            "- Edmon, H. J., Hoskins, B. J., & McIntyre, M. E. (1980). Eliassen-Palm cross sections for the troposphere. *Journal of the Atmospheric Sciences*, 37, 2600–2616.\n"
        )
        stream.write(
            "- Thorncroft, C. D., Hoskins, B. J., & McIntyre, M. E. (1993). Two paradigms of baroclinic-wave life-cycle behaviour. *Quarterly Journal of the Royal Meteorological Society*, 119, 17–55. https://doi.org/10.1002/qj.49711950903\n"
        )
        stream.write(
            "- Jucker, M. (2021). Scaling of Eliassen-Palm flux vectors. *Atmospheric Science Letters*, 22, e1020. https://doi.org/10.1002/asl.1020\n\n"
        )
        stream.write(f"PNG: `{png_path.relative_to(OUTDIR)}`\n\n")
        stream.write(f"PDF: `{pdf_path.relative_to(OUTDIR)}`\n")
    return note_path


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, xr.Dataset] = {}
    for case, shift in CASES:
        print(f"Processing {case}", flush=True)
        datasets[case] = analyze_case(case, shift)

    combined = xr.concat(
        [datasets[case].expand_dims(case=[case]) for case, _ in CASES],
        dim="case",
    )
    combined.attrs["reference_1"] = (
        "Edmon, Hoskins, and McIntyre (1980), Eliassen-Palm cross sections for the troposphere"
    )
    combined.attrs["reference_2"] = (
        "Thorncroft, Hoskins, and McIntyre (1993), Two paradigms of baroclinic-wave life-cycle behaviour"
    )
    combined.attrs["reference_3"] = (
        "Jucker (2021), Scaling of Eliassen-Palm flux vectors"
    )
    diagnostics_path = OUTDIR / "r1_6_ep_flux_evolution_diagnostics.nc"
    combined.to_netcdf(diagnostics_path)

    print("Plotting", flush=True)
    png_path, pdf_path = plot_figure(datasets)
    print("Validating", flush=True)
    metrics = panel_metrics(datasets)
    metrics_path = OUTDIR / "r1_6_ep_flux_evolution_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    note_path = write_method_note(metrics, png_path, pdf_path)
    print(diagnostics_path)
    print(metrics_path)
    print(png_path)
    print(pdf_path)
    print(note_path)


if __name__ == "__main__":
    main()
