#!/usr/bin/env python3
"""Recompute SI Figures S20 and S21 from eddy surface pressure.

The calculation is restricted to the 45-member constant-u0/standard ensemble.
Eddy surface pressure is PRESsfc minus the instantaneous zonal mean at each
model time and latitude. Existing source figures and scripts are never changed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage, stats
import xarray as xr
from contrack import contrack as ContrackClass

ROOT = Path("/data/keeling/a/mingfei5/a/data/original")
DEFAULT_OUTDIR = ROOT / "paper_revision" / "Figures_S20_S21_eddy_slp_recomputed_20260821"
CURRENT_MAIN_CATALOG = (
    ROOT
    / "paper_revision"
    / "anticyclone_eddyfield_replot_20260819_v2"
    / "tables"
    / "Figure13_eddy_persistence_case_level.csv"
)
OLD_S20_PRIMARY = (
    ROOT
    / "paper_revision"
    / "supplemental_analysis"
    / "anticyclone_stationarity_primary_summary_all45.csv"
)
OLD_S21_AUDIT = (
    ROOT
    / "paper_revision"
    / "R2_2_anticyclone_definition_sensitivity_20260817"
    / "R2_2_definition_audit.csv"
)

EARTH_RADIUS_KM = 6371.0
EDDY_THRESHOLDS_HPA = (5, 10, 15)
OVERLAP_THRESHOLDS = (0.6, 0.8, 0.9)
MOBILITY_THRESHOLDS = (0.20, 0.30, 0.40)
MIN_AREA_KM2 = 100_000.0
MIN_DURATION_H = 72
S20_OVERLAP = 0.30
B_VALUES = (1.0, 1.5, 2.0)
N_VALUES = (1, 3, 6)
N_PLOT_ORDER = (6, 3, 1)
S_VALUES = (-10, -5, 0, 5, 10)
BASELINE_ID = "O_plus10_o0.8"


@dataclass(frozen=True)
class PressureObject:
    object_id: int
    pixels: np.ndarray
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    peak_eddy_slp_hpa: float


@dataclass
class TrackPoint:
    time_index: int
    obj: PressureObject
    overlap_from_previous: float = math.nan
    displacement_km_from_previous: float = math.nan
    normalized_displacement_from_previous: float = math.nan


def patched_resolution(self, name: str, force: bool = False) -> float:
    values = np.asarray(self.ds[name].values)
    if values.size < 2:
        return np.nan
    if np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(values.dtype, np.timedelta64):
        return float((values[1] - values[0]).astype("timedelta64[s]").astype("int64") / 3600)
    return float(abs(values[1] - values[0]))


ContrackClass._get_resolution = patched_resolution
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("contrack").setLevel(logging.ERROR)


def parse_case_name(name: str) -> dict[str, object]:
    match = re.fullmatch(r"BCwave_b(15|1|2)n([136])(?:s(-?10|-?5))?", name)
    if match is None:
        raise ValueError(f"Unexpected standard case name: {name}")
    return {
        "case": name,
        "b": 1.5 if match.group(1) == "15" else float(match.group(1)),
        "n": int(match.group(2)),
        "s": int(match.group(3) or 0),
    }


def standard_case_paths() -> list[Path]:
    parsed: list[tuple[dict[str, object], Path]] = []
    for path in ROOT.glob("BCwave_*.nc"):
        try:
            parsed.append((parse_case_name(path.stem), path))
        except ValueError:
            continue
    parsed.sort(key=lambda item: (float(item[0]["b"]), int(item[0]["n"]), int(item[0]["s"])))
    paths = [path for _, path in parsed]
    if len(paths) != 45:
        raise RuntimeError(f"Expected 45 standard cases, found {len(paths)}")
    return paths


def spherical_cell_areas_km2(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    latitude_edges = np.empty(latitude.size + 1, dtype=float)
    latitude_edges[1:-1] = 0.5 * (latitude[:-1] + latitude[1:])
    latitude_edges[0] = max(-90.0, latitude[0] - 0.5 * (latitude[1] - latitude[0]))
    latitude_edges[-1] = min(90.0, latitude[-1] + 0.5 * (latitude[-1] - latitude[-2]))
    longitude_radians = np.unwrap(np.deg2rad(longitude))
    longitude_step = float(np.median(np.diff(longitude_radians)))
    strip = EARTH_RADIUS_KM**2 * longitude_step * (
        np.sin(np.deg2rad(latitude_edges[1:])) - np.sin(np.deg2rad(latitude_edges[:-1]))
    )
    return np.broadcast_to(strip[:, None], (latitude.size, longitude.size)).copy()


def weighted_spherical_centroid(
    pixels: np.ndarray,
    latitude_flat: np.ndarray,
    longitude_flat: np.ndarray,
    area_flat: np.ndarray,
) -> tuple[float, float]:
    weights = area_flat[pixels]
    latitude_radians = np.deg2rad(latitude_flat[pixels])
    longitude_radians = np.deg2rad(longitude_flat[pixels])
    x_value = np.sum(weights * np.cos(latitude_radians) * np.cos(longitude_radians))
    y_value = np.sum(weights * np.cos(latitude_radians) * np.sin(longitude_radians))
    z_value = np.sum(weights * np.sin(latitude_radians))
    norm = math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value)
    if norm == 0.0 or not np.isfinite(norm):
        raise RuntimeError("Undefined spherical centroid")
    x_value /= norm
    y_value /= norm
    z_value /= norm
    return (
        math.degrees(math.atan2(z_value, math.sqrt(x_value * x_value + y_value * y_value))),
        math.degrees(math.atan2(y_value, x_value)) % 360.0,
    )


def great_circle_km(latitude1: float, longitude1: float, latitude2: float, longitude2: float) -> float:
    phi1 = math.radians(latitude1)
    phi2 = math.radians(latitude2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(((longitude2 - longitude1 + 540.0) % 360.0) - 180.0)
    haversine = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(max(0.0, haversine))))


def _find(parent: np.ndarray, value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = int(parent[value])
    return value


def _union(parent: np.ndarray, first: int, second: int) -> None:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root


def cyclic_connected_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Eight-neighbor components with the same-latitude cyclic seam used by ConTrack."""
    labels, nlabels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if nlabels == 0:
        return labels.astype(np.int32, copy=False), 0
    parent = np.arange(nlabels + 1, dtype=np.int32)
    for row_index in range(labels.shape[0]):
        first = int(labels[row_index, 0])
        second = int(labels[row_index, -1])
        if first > 0 and second > 0:
            _union(parent, first, second)
    root_map = np.arange(nlabels + 1, dtype=np.int32)
    for label in range(1, nlabels + 1):
        root_map[label] = _find(parent, label)
    rooted = root_map[labels]
    roots = np.unique(rooted[rooted > 0])
    compact = np.zeros(nlabels + 1, dtype=np.int32)
    for new_label, root in enumerate(roots, start=1):
        compact[int(root)] = new_label
    return compact[rooted], len(roots)


def objects_at_time(
    eddy_pressure_pa: np.ndarray,
    threshold_hpa: float,
    area_flat: np.ndarray,
    latitude_flat: np.ndarray,
    longitude_flat: np.ndarray,
) -> tuple[list[PressureObject], np.ndarray, dict[str, int]]:
    threshold_pa = threshold_hpa * 100.0
    mask = np.isfinite(eddy_pressure_pa) & (eddy_pressure_pa >= threshold_pa)
    labels, nlabels = cyclic_connected_labels(mask)
    labels_flat = labels.ravel()
    field_flat = eddy_pressure_pa.ravel()
    retained_mask = np.zeros(mask.size, dtype=bool)
    objects: list[PressureObject] = []
    rejected = 0
    dateline_crossing = 0
    boundary_touching = 0
    for label in range(1, nlabels + 1):
        pixels = np.flatnonzero(labels_flat == label).astype(np.int32)
        area_km2 = float(np.sum(area_flat[pixels]))
        if area_km2 < MIN_AREA_KM2:
            rejected += 1
            continue
        columns = pixels % mask.shape[1]
        if np.any(columns == 0) and np.any(columns == mask.shape[1] - 1):
            dateline_crossing += 1
        rows = pixels // mask.shape[1]
        if np.any(rows == 0) or np.any(rows == mask.shape[0] - 1):
            boundary_touching += 1
        centroid_lat, centroid_lon = weighted_spherical_centroid(
            pixels, latitude_flat, longitude_flat, area_flat
        )
        objects.append(
            PressureObject(
                object_id=label,
                pixels=pixels,
                area_km2=area_km2,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                peak_eddy_slp_hpa=float(np.nanmax(field_flat[pixels]) / 100.0),
            )
        )
        retained_mask[pixels] = True
    return objects, retained_mask.reshape(mask.shape), {
        "raw_components": int(nlabels),
        "retained_components": int(len(objects)),
        "area_rejected_components": int(rejected),
        "dateline_crossing_retained_components": int(dateline_crossing),
        "domain_boundary_touching_retained_components": int(boundary_touching),
    }


def pixel_overlap_min(first: PressureObject, second: PressureObject) -> float:
    common = np.intersect1d(first.pixels, second.pixels, assume_unique=True)
    if common.size == 0:
        return 0.0
    return float(common.size) / float(min(first.pixels.size, second.pixels.size))


def area_overlap_min(first: PressureObject, second: PressureObject, area_flat: np.ndarray) -> float:
    common = np.intersect1d(first.pixels, second.pixels, assume_unique=True)
    if common.size == 0:
        return 0.0
    return float(np.sum(area_flat[common])) / min(first.area_km2, second.area_km2)


def link_overlap_tracks(
    objects_by_time: list[list[PressureObject]], overlap_threshold: float
) -> dict[int, list[TrackPoint]]:
    """S20 identity linking, preserving the original intersection/min-pixel overlap metric."""
    active: dict[int, PressureObject] = {}
    tracks: dict[int, list[TrackPoint]] = {}
    next_track_id = 1
    for time_index, objects in enumerate(objects_by_time):
        candidates: list[tuple[float, int, int]] = []
        for track_id, previous in active.items():
            for object_index, current in enumerate(objects):
                overlap = pixel_overlap_min(previous, current)
                if overlap >= overlap_threshold:
                    candidates.append((overlap, track_id, object_index))
        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_objects: set[int] = set()
        assignments: dict[int, tuple[int, float]] = {}
        for overlap, track_id, object_index in candidates:
            if track_id in used_tracks or object_index in used_objects:
                continue
            used_tracks.add(track_id)
            used_objects.add(object_index)
            assignments[object_index] = (track_id, overlap)
        new_active: dict[int, PressureObject] = {}
        for object_index, current in enumerate(objects):
            if object_index in assignments:
                track_id, overlap = assignments[object_index]
            else:
                track_id = next_track_id
                next_track_id += 1
                overlap = math.nan
            point = TrackPoint(time_index=time_index, obj=current, overlap_from_previous=overlap)
            if tracks.get(track_id):
                previous_point = tracks[track_id][-1]
                point.displacement_km_from_previous = great_circle_km(
                    previous_point.obj.centroid_lat,
                    previous_point.obj.centroid_lon,
                    current.centroid_lat,
                    current.centroid_lon,
                )
                mean_radius = 0.5 * (
                    math.sqrt(previous_point.obj.area_km2 / math.pi)
                    + math.sqrt(current.area_km2 / math.pi)
                )
                point.normalized_displacement_from_previous = (
                    point.displacement_km_from_previous / mean_radius if mean_radius > 0.0 else math.nan
                )
            tracks.setdefault(track_id, []).append(point)
            new_active[track_id] = current
        active = new_active
    return tracks


def link_permissive_identity_tracks(
    objects_by_time: list[list[PressureObject]], area_flat: np.ndarray
) -> tuple[dict[int, list[TrackPoint]], dict[str, int]]:
    """One-to-one consecutive identity links used before applying mobility thresholds.

    Links require positive mask overlap, area ratio >=0.25, distance <=2500 km,
    and either >=5% area-weighted overlap or centroid displacement <= one mean
    equivalent radius. The positive-overlap requirement prevents jumps to an
    unrelated nearby object. Splits/mergers retain only the highest-scoring link;
    unmatched branches begin/end tracks. No temporary disappearance is bridged.
    """
    active: dict[int, PressureObject] = {}
    tracks: dict[int, list[TrackPoint]] = {}
    next_track_id = 1
    diagnostics = {
        "candidate_links": 0,
        "accepted_links": 0,
        "zero_overlap_links_rejected": 0,
        "area_ratio_links_rejected": 0,
        "distance_links_rejected": 0,
    }
    for time_index, objects in enumerate(objects_by_time):
        candidates: list[tuple[float, int, int, float, float, float]] = []
        for track_id, previous in active.items():
            previous_radius = math.sqrt(previous.area_km2 / math.pi)
            for object_index, current in enumerate(objects):
                diagnostics["candidate_links"] += 1
                current_radius = math.sqrt(current.area_km2 / math.pi)
                mean_radius = 0.5 * (previous_radius + current_radius)
                displacement = great_circle_km(
                    previous.centroid_lat,
                    previous.centroid_lon,
                    current.centroid_lat,
                    current.centroid_lon,
                )
                normalized = displacement / mean_radius if mean_radius > 0.0 else math.inf
                area_ratio = min(previous.area_km2, current.area_km2) / max(
                    previous.area_km2, current.area_km2
                )
                overlap = area_overlap_min(previous, current, area_flat)
                if overlap <= 0.0:
                    diagnostics["zero_overlap_links_rejected"] += 1
                    continue
                if area_ratio < 0.25:
                    diagnostics["area_ratio_links_rejected"] += 1
                    continue
                if displacement > 2500.0 or (overlap < 0.05 and normalized > 1.0):
                    diagnostics["distance_links_rejected"] += 1
                    continue
                score = 3.0 * overlap + 1.5 * math.exp(-normalized) + 0.5 * area_ratio
                candidates.append((score, track_id, object_index, overlap, displacement, normalized))
        candidates.sort(reverse=True)
        assignments: dict[int, tuple[int, float, float, float]] = {}
        used_tracks: set[int] = set()
        used_objects: set[int] = set()
        for _, track_id, object_index, overlap, displacement, normalized in candidates:
            if track_id in used_tracks or object_index in used_objects:
                continue
            used_tracks.add(track_id)
            used_objects.add(object_index)
            assignments[object_index] = (track_id, overlap, displacement, normalized)
            diagnostics["accepted_links"] += 1
        new_active: dict[int, PressureObject] = {}
        for object_index, current in enumerate(objects):
            if object_index in assignments:
                track_id, overlap, displacement, normalized = assignments[object_index]
            else:
                track_id = next_track_id
                next_track_id += 1
                overlap = displacement = normalized = math.nan
            tracks.setdefault(track_id, []).append(
                TrackPoint(
                    time_index=time_index,
                    obj=current,
                    overlap_from_previous=overlap,
                    displacement_km_from_previous=displacement,
                    normalized_displacement_from_previous=normalized,
                )
            )
            new_active[track_id] = current
        active = new_active
    return tracks, diagnostics


def track_duration(points: list[TrackPoint]) -> int:
    return len(points)


def path_metrics(points: list[TrackPoint]) -> dict[str, float]:
    if not points:
        return {
            "mean_centroid_path_speed_m_s-1": math.nan,
            "net_centroid_speed_m_s-1": math.nan,
            "mean_step_displacement_km": math.nan,
            "mean_normalized_step": math.nan,
        }
    distances = np.asarray(
        [point.displacement_km_from_previous for point in points[1:]], dtype=float
    )
    finite_distances = distances[np.isfinite(distances)]
    path_speed = (
        float(np.mean(finite_distances) * 1000.0 / 3600.0)
        if finite_distances.size
        else 0.0
    )
    if len(points) > 1:
        net_distance = great_circle_km(
            points[0].obj.centroid_lat,
            points[0].obj.centroid_lon,
            points[-1].obj.centroid_lat,
            points[-1].obj.centroid_lon,
        )
        net_speed = net_distance * 1000.0 / ((len(points) - 1) * 3600.0)
    else:
        net_speed = 0.0
    normalized = np.asarray(
        [point.normalized_displacement_from_previous for point in points[1:]], dtype=float
    )
    finite_normalized = normalized[np.isfinite(normalized)]
    return {
        "mean_centroid_path_speed_m_s-1": path_speed,
        "net_centroid_speed_m_s-1": float(net_speed),
        "mean_step_displacement_km": (
            float(np.mean(finite_distances)) if finite_distances.size else 0.0
        ),
        "mean_normalized_step": (
            float(np.mean(finite_normalized)) if finite_normalized.size else 0.0
        ),
    }


def summarize_track_points(points: list[TrackPoint]) -> dict[str, float | int]:
    metrics = path_metrics(points)
    return {
        "duration_h": int(len(points)),
        "start_hour": int(points[0].time_index + 1) if points else 0,
        "end_hour": int(points[-1].time_index + 1) if points else 0,
        "mean_object_area_km2": (
            float(np.mean([point.obj.area_km2 for point in points])) if points else math.nan
        ),
        "maximum_object_area_km2": (
            float(np.max([point.obj.area_km2 for point in points])) if points else math.nan
        ),
        "peak_eddy_slp_anomaly_hpa": (
            float(np.max([point.obj.peak_eddy_slp_hpa for point in points])) if points else math.nan
        ),
        **metrics,
    }


def select_longest_track(tracks: dict[int, list[TrackPoint]]) -> tuple[int | None, list[TrackPoint]]:
    if not tracks:
        return None, []
    track_id, points = max(
        tracks.items(),
        key=lambda item: (
            len(item[1]),
            max(point.obj.peak_eddy_slp_hpa for point in item[1]),
            np.mean([point.obj.area_km2 for point in item[1]]),
            item[0],
        ),
    )
    return track_id, points


def mobility_episodes(points: list[TrackPoint], threshold: float) -> list[list[TrackPoint]]:
    if not points:
        return []
    episodes: list[list[TrackPoint]] = []
    start = 0
    for index in range(1, len(points)):
        normalized = points[index].normalized_displacement_from_previous
        if np.isfinite(normalized) and normalized <= threshold:
            continue
        episodes.append(points[start:index])
        start = index
    episodes.append(points[start:])
    return [episode for episode in episodes if episode]


def summarize_mobility_definition(
    tracks: dict[int, list[TrackPoint]], threshold: float
) -> dict[str, float | int]:
    episodes_with_ids: list[tuple[int, list[TrackPoint]]] = []
    for track_id, points in tracks.items():
        episodes_with_ids.extend((track_id, episode) for episode in mobility_episodes(points, threshold))
    qualifying = [item for item in episodes_with_ids if len(item[1]) >= MIN_DURATION_H]
    if qualifying:
        primary_track_id, primary = max(
            qualifying,
            key=lambda item: (
                len(item[1]),
                max(point.obj.peak_eddy_slp_hpa for point in item[1]),
                np.mean([point.obj.area_km2 for point in item[1]]),
                item[0],
            ),
        )
    else:
        primary_track_id, primary = None, []
    pooled = [point for _, episode in qualifying for point in episode]
    primary_summary = summarize_track_points(primary)
    return {
        "maximum_qualifying_persistence_h": int(max((len(points) for _, points in qualifying), default=0)),
        "number_qualifying_tracks": int(len({track_id for track_id, _ in qualifying})),
        "number_qualifying_episodes": int(len(qualifying)),
        "selected_track_id": primary_track_id if primary_track_id is not None else math.nan,
        "selected_track_duration_h": primary_summary["duration_h"],
        "mean_qualifying_object_area_km2": (
            float(np.mean([point.obj.area_km2 for point in pooled])) if pooled else math.nan
        ),
        "maximum_qualifying_object_area_km2": (
            float(np.max([point.obj.area_km2 for point in pooled])) if pooled else math.nan
        ),
        "maximum_peak_eddy_slp_anomaly_hpa": (
            float(np.max([point.obj.peak_eddy_slp_hpa for point in pooled])) if pooled else math.nan
        ),
        **{key: value for key, value in primary_summary.items() if key not in {"duration_h"}},
    }


def run_contrack_mask(
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    time_values: np.ndarray,
    overlap_threshold: float,
) -> np.ndarray:
    data_array = xr.DataArray(
        mask.astype(np.int8, copy=False),
        dims=("time", "latitude", "longitude"),
        coords={"time": time_values, "latitude": latitude, "longitude": longitude},
        name="retained_positive_eddy_object",
    )
    tracker = ContrackClass()
    tracker.ds = data_array.to_dataset()
    with open(os.devnull, "w") as null, redirect_stdout(null), redirect_stderr(null):
        tracker.run_contrack(
            variable="retained_positive_eddy_object",
            threshold=0.5,
            gorl=">=",
            overlap=overlap_threshold,
            persistence=MIN_DURATION_H,
            twosided=False,
        )
    return tracker.ds["flag"].fillna(0).astype(np.int32).values


def summarize_contrack_flag(
    flag: np.ndarray,
    eddy_pressure_pa: np.ndarray,
    area2d: np.ndarray,
    latitude2d: np.ndarray,
    longitude2d: np.ndarray,
) -> dict[str, float | int]:
    event_ids = [int(value) for value in np.unique(flag) if value > 0]
    events: list[dict[str, object]] = []
    area_flat = area2d.ravel()
    latitude_flat = latitude2d.ravel()
    longitude_flat = longitude2d.ravel()
    for event_id in event_ids:
        time_indices = np.flatnonzero(np.any(flag == event_id, axis=(1, 2)))
        points: list[TrackPoint] = []
        for time_index in time_indices:
            pixels = np.flatnonzero(flag[time_index].ravel() == event_id).astype(np.int32)
            if not pixels.size:
                continue
            centroid_lat, centroid_lon = weighted_spherical_centroid(
                pixels, latitude_flat, longitude_flat, area_flat
            )
            field_flat = eddy_pressure_pa[time_index].ravel()
            obj = PressureObject(
                object_id=event_id,
                pixels=pixels,
                area_km2=float(np.sum(area_flat[pixels])),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                peak_eddy_slp_hpa=float(np.nanmax(field_flat[pixels]) / 100.0),
            )
            point = TrackPoint(time_index=int(time_index), obj=obj)
            if points:
                point.displacement_km_from_previous = great_circle_km(
                    points[-1].obj.centroid_lat,
                    points[-1].obj.centroid_lon,
                    centroid_lat,
                    centroid_lon,
                )
                mean_radius = 0.5 * (
                    math.sqrt(points[-1].obj.area_km2 / math.pi)
                    + math.sqrt(obj.area_km2 / math.pi)
                )
                point.normalized_displacement_from_previous = (
                    point.displacement_km_from_previous / mean_radius if mean_radius > 0.0 else math.nan
                )
            points.append(point)
        if points:
            events.append({"event_id": event_id, "points": points})
    if events:
        primary_event = max(
            events,
            key=lambda event: (
                len(event["points"]),
                max(point.obj.peak_eddy_slp_hpa for point in event["points"]),
                int(event["event_id"]),
            ),
        )
        primary_points = primary_event["points"]
        primary_event_id = int(primary_event["event_id"])
    else:
        primary_points = []
        primary_event_id = math.nan
    pooled = [point for event in events for point in event["points"]]
    primary_summary = summarize_track_points(primary_points)
    return {
        "maximum_qualifying_persistence_h": int(max((len(event["points"]) for event in events), default=0)),
        "number_qualifying_tracks": int(len(events)),
        "number_qualifying_episodes": int(len(events)),
        "selected_track_id": primary_event_id,
        "selected_track_duration_h": primary_summary["duration_h"],
        "mean_qualifying_object_area_km2": (
            float(np.mean([point.obj.area_km2 for point in pooled])) if pooled else math.nan
        ),
        "maximum_qualifying_object_area_km2": (
            float(np.max([point.obj.area_km2 for point in pooled])) if pooled else math.nan
        ),
        "maximum_peak_eddy_slp_anomaly_hpa": (
            float(np.max([point.obj.peak_eddy_slp_hpa for point in pooled])) if pooled else math.nan
        ),
        **{key: value for key, value in primary_summary.items() if key not in {"duration_h"}},
    }


def definition_id(method: str, threshold_hpa: int, criterion: float) -> str:
    if method == "contour_overlap":
        return f"O_plus{threshold_hpa}_o{criterion:.1f}"
    return f"M_plus{threshold_hpa}_m{criterion:.2f}"


def process_case(path_string: str) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    path = Path(path_string)
    metadata = parse_case_name(path.stem)
    with xr.open_dataset(path, decode_times=False, engine="netcdf4", mask_and_scale=True) as dataset:
        latitude_all = dataset["grid_yt"].values.astype(float)
        longitude = np.mod(dataset["grid_xt"].values.astype(float), 360.0)
        nh_indices = np.flatnonzero(latitude_all >= 0.0)
        latitude = latitude_all[nh_indices]
        pressure = dataset["PRESsfc"].isel(grid_yt=nh_indices).load()
        pressure = pressure.rename({"grid_yt": "latitude", "grid_xt": "longitude"})
        pressure = pressure.assign_coords(latitude=("latitude", latitude), longitude=("longitude", longitude))
        pressure = pressure.sortby("longitude")
        time_values = dataset["time"].values.astype(float)
    if pressure.shape != (360, latitude.size, longitude.size):
        raise RuntimeError(f"Unexpected PRESsfc shape for {path.name}: {pressure.shape}")
    if not np.allclose(np.diff(time_values), 1.0):
        raise RuntimeError(f"Non-hourly time coordinate in {path.name}")
    pressure_values = pressure.values.astype(np.float64, copy=False)
    eddy_pressure_pa = pressure_values - pressure_values.mean(axis=2, keepdims=True)
    if not np.isfinite(eddy_pressure_pa).all():
        raise RuntimeError(f"Missing eddy-SLP values in {path.name}")
    zonal_residual_hpa = float(np.max(np.abs(np.mean(eddy_pressure_pa, axis=2))) / 100.0)

    area2d = spherical_cell_areas_km2(latitude, longitude)
    latitude2d = np.broadcast_to(latitude[:, None], area2d.shape)
    longitude2d = np.broadcast_to(longitude[None, :], area2d.shape)
    area_flat = area2d.ravel()
    latitude_flat = latitude2d.ravel()
    longitude_flat = longitude2d.ravel()

    objects_by_threshold: dict[int, list[list[PressureObject]]] = {}
    masks_by_threshold: dict[int, np.ndarray] = {}
    object_audit_rows: list[dict[str, object]] = []
    for threshold_hpa in EDDY_THRESHOLDS_HPA:
        objects_by_time: list[list[PressureObject]] = []
        retained_masks: list[np.ndarray] = []
        audit_totals = {
            "raw_components": 0,
            "retained_components": 0,
            "area_rejected_components": 0,
            "dateline_crossing_retained_components": 0,
            "domain_boundary_touching_retained_components": 0,
        }
        for time_index in range(eddy_pressure_pa.shape[0]):
            objects, retained_mask, audit = objects_at_time(
                eddy_pressure_pa[time_index],
                threshold_hpa,
                area_flat,
                latitude_flat,
                longitude_flat,
            )
            objects_by_time.append(objects)
            retained_masks.append(retained_mask)
            for key in audit_totals:
                audit_totals[key] += int(audit[key])
        objects_by_threshold[threshold_hpa] = objects_by_time
        masks_by_threshold[threshold_hpa] = np.asarray(retained_masks, dtype=bool)
        object_audit_rows.append(
            {
                **metadata,
                "eddy_threshold_hpa": threshold_hpa,
                "minimum_area_km2": MIN_AREA_KM2,
                "maximum_absolute_zonal_mean_residual_hpa": zonal_residual_hpa,
                **audit_totals,
            }
        )

    s20_tracks = link_overlap_tracks(objects_by_threshold[10], S20_OVERLAP)
    s20_track_id, s20_points = select_longest_track(s20_tracks)
    s20_summary = summarize_track_points(s20_points)
    s20_row = {
        **metadata,
        "definition_id": "S20_plus10_overlap0.3_identity",
        "method": "independent_permissive_overlap_identity",
        "eddy_threshold_hpa": 10,
        "criterion_value": S20_OVERLAP,
        "minimum_area_km2": MIN_AREA_KM2,
        "minimum_duration_h": 0,
        "selected_track_id": s20_track_id if s20_track_id is not None else math.nan,
        "maximum_qualifying_persistence_h": s20_summary["duration_h"],
        "number_qualifying_tracks": int(sum(len(points) >= MIN_DURATION_H for points in s20_tracks.values())),
        "number_qualifying_episodes": int(sum(len(points) >= MIN_DURATION_H for points in s20_tracks.values())),
        "selected_track_duration_h": s20_summary["duration_h"],
        "mean_qualifying_object_area_km2": s20_summary["mean_object_area_km2"],
        "maximum_qualifying_object_area_km2": s20_summary["maximum_object_area_km2"],
        "maximum_peak_eddy_slp_anomaly_hpa": s20_summary["peak_eddy_slp_anomaly_hpa"],
        **{key: value for key, value in s20_summary.items() if key not in {"duration_h", "mean_object_area_km2", "maximum_object_area_km2", "peak_eddy_slp_anomaly_hpa"}},
        "source_file": str(path),
    }

    case_rows: list[dict[str, object]] = []
    for threshold_hpa in EDDY_THRESHOLDS_HPA:
        retained_mask = masks_by_threshold[threshold_hpa]
        for overlap_threshold in OVERLAP_THRESHOLDS:
            flag = run_contrack_mask(retained_mask, latitude, longitude, time_values, overlap_threshold)
            summary = summarize_contrack_flag(
                flag, eddy_pressure_pa, area2d, latitude2d, longitude2d
            )
            case_rows.append(
                {
                    **metadata,
                    "definition_id": definition_id("contour_overlap", threshold_hpa, overlap_threshold),
                    "method": "contour_overlap",
                    "eddy_threshold_hpa": threshold_hpa,
                    "criterion_value": overlap_threshold,
                    "criterion_label": f"consecutive overlap >= {overlap_threshold:.1f}",
                    "minimum_area_km2": MIN_AREA_KM2,
                    "minimum_duration_h": MIN_DURATION_H,
                    **summary,
                    "source_file": str(path),
                }
            )
        identity_tracks, identity_diagnostics = link_permissive_identity_tracks(
            objects_by_threshold[threshold_hpa], area_flat
        )
        for mobility_threshold in MOBILITY_THRESHOLDS:
            summary = summarize_mobility_definition(identity_tracks, mobility_threshold)
            case_rows.append(
                {
                    **metadata,
                    "definition_id": definition_id("size_normalized_mobility", threshold_hpa, mobility_threshold),
                    "method": "size_normalized_mobility",
                    "eddy_threshold_hpa": threshold_hpa,
                    "criterion_value": mobility_threshold,
                    "criterion_label": f"hourly centroid displacement / mean equivalent radius <= {mobility_threshold:.2f}",
                    "minimum_area_km2": MIN_AREA_KM2,
                    "minimum_duration_h": MIN_DURATION_H,
                    **summary,
                    **identity_diagnostics,
                    "source_file": str(path),
                }
            )

    no_area_mask = eddy_pressure_pa >= 1000.0
    no_area_flag = run_contrack_mask(no_area_mask, latitude, longitude, time_values, 0.8)
    no_area_summary = summarize_contrack_flag(
        no_area_flag, eddy_pressure_pa, area2d, latitude2d, longitude2d
    )
    requested_baseline = next(row for row in case_rows if row["definition_id"] == BASELINE_ID)
    baseline_validation = {
        **metadata,
        "source_file": str(path),
        "recomputed_no_area_duration_h": no_area_summary["maximum_qualifying_persistence_h"],
        "recomputed_no_area_event_count": no_area_summary["number_qualifying_tracks"],
        "requested_area_filtered_duration_h": requested_baseline["maximum_qualifying_persistence_h"],
        "requested_area_filtered_event_count": requested_baseline["number_qualifying_tracks"],
        "area_filter_duration_difference_h": int(requested_baseline["maximum_qualifying_persistence_h"])
        - int(no_area_summary["maximum_qualifying_persistence_h"]),
        "maximum_absolute_zonal_mean_residual_hpa": zonal_residual_hpa,
    }
    return s20_row, case_rows, baseline_validation, object_audit_rows, []


def category_order(frame: pd.DataFrame, parameter: str) -> str:
    means = frame.groupby(parameter)["maximum_qualifying_persistence_h"].mean()
    ordered = sorted(means.index.tolist(), key=lambda value: (-float(means.loc[value]), float(value)))
    return ">".join(f"{float(value):g}" for value in ordered)


def safe_spearman(first: pd.Series, second: pd.Series) -> float:
    result = stats.spearmanr(first.to_numpy(float), second.to_numpy(float), nan_policy="omit")
    return float(result.statistic)


def summarize_s21(case_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for definition_id_value, definition_frame in case_results.groupby("definition_id", sort=False):
        first = definition_frame.iloc[0]
        for parameter, categories in (("b", B_VALUES), ("n", N_VALUES), ("s", S_VALUES)):
            for category in categories:
                values = definition_frame.loc[
                    np.isclose(definition_frame[parameter].astype(float), float(category)),
                    "maximum_qualifying_persistence_h",
                ].astype(float)
                grouped_rows.append(
                    {
                        "definition_id": definition_id_value,
                        "method": first["method"],
                        "eddy_threshold_hpa": first["eddy_threshold_hpa"],
                        "criterion_value": first["criterion_value"],
                        "parameter": parameter,
                        "category": float(category),
                        "n_cases": int(len(values)),
                        "mean_maximum_persistence_h": float(values.mean()),
                        "median_maximum_persistence_h": float(values.median()),
                        "minimum_maximum_persistence_h": float(values.min()),
                        "maximum_maximum_persistence_h": float(values.max()),
                    }
                )
        audit_rows.append(
            {
                "definition_id": definition_id_value,
                "method": first["method"],
                "eddy_threshold_hpa": first["eddy_threshold_hpa"],
                "criterion_value": first["criterion_value"],
                "spearman_rho_b": safe_spearman(definition_frame["b"], definition_frame["maximum_qualifying_persistence_h"]),
                "spearman_rho_n": safe_spearman(definition_frame["n"], definition_frame["maximum_qualifying_persistence_h"]),
                "spearman_rho_s": safe_spearman(definition_frame["s"], definition_frame["maximum_qualifying_persistence_h"]),
                "ranking_b": category_order(definition_frame, "b"),
                "ranking_n": category_order(definition_frame, "n"),
                "ranking_s": category_order(definition_frame, "s"),
                "n_cases_positive_duration": int((definition_frame["maximum_qualifying_persistence_h"] > 0).sum()),
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    audit = pd.DataFrame(audit_rows)
    baseline = audit.set_index("definition_id").loc[BASELINE_ID]
    for parameter in ("b", "n", "s"):
        baseline_rho = float(baseline[f"spearman_rho_{parameter}"])
        audit[f"direction_agrees_with_baseline_{parameter}"] = (
            np.sign(audit[f"spearman_rho_{parameter}"]) == np.sign(baseline_rho)
        )
        audit[f"ranking_agrees_with_baseline_{parameter}"] = (
            audit[f"ranking_{parameter}"] == baseline[f"ranking_{parameter}"]
        )
    method_order = pd.Categorical(
        audit["method"], categories=["contour_overlap", "size_normalized_mobility"], ordered=True
    )
    audit = audit.assign(_method_order=method_order).sort_values(
        ["_method_order", "eddy_threshold_hpa", "criterion_value"]
    ).drop(columns="_method_order").reset_index(drop=True)
    return grouped, audit


def figure_definition_label(row: pd.Series) -> str:
    prefix = "O" if row["method"] == "contour_overlap" else "M"
    criterion = f"{row['criterion_value']:.1f}" if prefix == "O" else f"{row['criterion_value']:.2f}"
    return f"{prefix} +{int(row['eddy_threshold_hpa'])}/{criterion}"


def make_s20_figure(s20: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    lifetime_max = float(s20["maximum_qualifying_persistence_h"].max())
    speed_max = float(s20["mean_centroid_path_speed_m_s-1"].max())
    figure, axes = plt.subplots(2, 5, figsize=(10.4, 4.9), constrained_layout=True)
    lifetime_image = None
    speed_image = None
    for column, shift in enumerate(S_VALUES):
        subset = s20[s20["s"] == shift].copy()
        lifetime = np.full((len(B_VALUES), len(N_PLOT_ORDER)), np.nan)
        speed = np.full_like(lifetime, np.nan)
        for row_index, b_value in enumerate(B_VALUES):
            for n_index, n_value in enumerate(N_PLOT_ORDER):
                row = subset[np.isclose(subset["b"], b_value) & subset["n"].eq(n_value)]
                if len(row) == 1 and np.isfinite(float(row.iloc[0]["selected_track_id"])):
                    lifetime[row_index, n_index] = float(row.iloc[0]["maximum_qualifying_persistence_h"])
                    speed[row_index, n_index] = float(row.iloc[0]["mean_centroid_path_speed_m_s-1"])
        for row_number, (axis, values, cmap, upper, decimals) in enumerate(
            (
                (axes[0, column], lifetime, "YlGnBu", lifetime_max, 0),
                (axes[1, column], speed, "magma_r", speed_max, 1),
            )
        ):
            color_map = plt.get_cmap(cmap).copy()
            color_map.set_bad("white")
            image = axis.imshow(values, origin="lower", aspect="equal", cmap=color_map, vmin=0.0, vmax=upper)
            if row_number == 0:
                lifetime_image = image
            else:
                speed_image = image
            axis.set_xticks(range(len(N_PLOT_ORDER)), [str(value) for value in N_PLOT_ORDER])
            axis.set_yticks(range(len(B_VALUES)), [f"{value:g}" for value in B_VALUES])
            axis.set_title(rf"$s={shift:+d}^\circ$" if shift else r"$s=0^\circ$", pad=4)
            for row_index in range(values.shape[0]):
                for n_index in range(values.shape[1]):
                    value = values[row_index, n_index]
                    text = "—" if not np.isfinite(value) else f"{value:.{decimals}f}"
                    color = "white" if np.isfinite(value) and value > 0.62 * upper else "black"
                    axis.text(n_index, row_index, text, ha="center", va="center", fontsize=8.5, color=color)
            if column == 0:
                axis.set_ylabel(r"Parameter $b$")
            else:
                axis.tick_params(labelleft=False)
            if row_number == 1:
                axis.set_xlabel(r"Jet-width parameter $n$")
            else:
                axis.tick_params(labelbottom=False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
    axes[0, 0].text(-0.44, 0.5, "(a) Lifetime", transform=axes[0, 0].transAxes, rotation=90,
                    ha="center", va="center", fontsize=10.5, fontweight="bold")
    axes[1, 0].text(-0.44, 0.5, "(b) Path speed", transform=axes[1, 0].transAxes, rotation=90,
                    ha="center", va="center", fontsize=10.5, fontweight="bold")
    lifetime_cbar = figure.colorbar(lifetime_image, ax=axes[0, :], location="right", shrink=0.88, pad=0.015)
    lifetime_cbar.set_label("Longest-track lifetime (h)")
    speed_cbar = figure.colorbar(speed_image, ax=axes[1, :], location="right", shrink=0.88, pad=0.015)
    speed_cbar.set_label(r"Mean centroid path speed (m s$^{-1}$)")
    png = outdir / "Figure_S20_eddy_slp_lifetime_centroid_speed.png"
    pdf = outdir / "Figure_S20_eddy_slp_lifetime_centroid_speed.pdf"
    figure.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def make_s21_figure(grouped: pd.DataFrame, audit: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    ordered = audit.copy().reset_index(drop=True)
    rho_matrix = ordered[["spearman_rho_b", "spearman_rho_n", "spearman_rho_s"]].to_numpy(float)
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    axis_heat, axis_b, axis_n, axis_s = axes.ravel()
    image = axis_heat.imshow(rho_matrix, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis_heat.set_xticks(range(3), [r"$b$", r"$n$", r"$s$"])
    axis_heat.set_yticks(range(len(ordered)), [figure_definition_label(row) for _, row in ordered.iterrows()])
    axis_heat.set_xlabel("Prescribed parameter")
    axis_heat.set_ylabel("Definition: method, eddy threshold, criterion")
    for row_index in range(rho_matrix.shape[0]):
        for column_index in range(rho_matrix.shape[1]):
            value = rho_matrix[row_index, column_index]
            axis_heat.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(value) >= 0.55 else "black",
            )
    baseline_row = int(np.flatnonzero(ordered["definition_id"].eq(BASELINE_ID))[0])
    axis_heat.add_patch(Rectangle((-0.49, baseline_row - 0.49), 2.98, 0.98, fill=False, ec="black", lw=2.2))
    axis_heat.axhline(8.5, color="0.25", lw=1.0)
    colorbar = figure.colorbar(image, ax=axis_heat, orientation="horizontal", pad=0.08, fraction=0.06)
    colorbar.set_label(r"Spearman $\rho$")

    threshold_colors = {5: "#6F93A6", 10: "#B56A55", 15: "#75658F"}
    maximum_group_mean = float(grouped["mean_maximum_persistence_h"].max())
    for axis, parameter, categories in (
        (axis_b, "b", B_VALUES),
        (axis_n, "n", N_VALUES),
        (axis_s, "s", S_VALUES),
    ):
        for _, definition in ordered.iterrows():
            rows = grouped[
                grouped["definition_id"].eq(definition["definition_id"])
                & grouped["parameter"].eq(parameter)
            ]
            values = rows.set_index("category")["mean_maximum_persistence_h"].to_dict()
            y_values = [values[float(category)] for category in categories]
            is_baseline = definition["definition_id"] == BASELINE_ID
            axis.plot(
                np.arange(len(categories)),
                y_values,
                color="black" if is_baseline else threshold_colors[int(definition["eddy_threshold_hpa"])],
                lw=2.8 if is_baseline else 1.05,
                alpha=1.0 if is_baseline else 0.58,
                linestyle="-" if definition["method"] == "contour_overlap" else "--",
                marker="o" if is_baseline else None,
                markersize=4.2,
                zorder=5 if is_baseline else 2,
            )
        axis.set_xticks(np.arange(len(categories)), [f"{value:g}" for value in categories])
        axis.set_xlabel(rf"${parameter}$")
        axis.set_ylabel("Group-mean maximum persistence (h)")
        axis.set_ylim(-4, maximum_group_mean + 12)
        axis.axhline(0.0, color="0.55", lw=0.7)
        axis.grid(axis="y", color="0.88", lw=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.8, marker="o", label="Baseline +10 hPa / 0.8"),
        Line2D([0], [0], color="0.35", lw=1.4, ls="-", label="Contour overlap (O)"),
        Line2D([0], [0], color="0.35", lw=1.4, ls="--", label="Size-normalized mobility (M)"),
    ]
    legend_handles.extend(
        Line2D([0], [0], color=color, lw=2.0, label=f"+{threshold} hPa")
        for threshold, color in threshold_colors.items()
    )
    figure.legend(handles=legend_handles, ncol=6, loc="outside upper center",
                  frameon=False, columnspacing=1.0, handlelength=2.2)
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), (axis_heat, axis_b, axis_n, axis_s)):
        axis.text(0.01, 0.99, label, transform=axis.transAxes, ha="left", va="top",
                  fontsize=11.5, fontweight="bold",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.82})
    png = outdir / "Figure_S21_eddy_slp_definition_sensitivity.png"
    pdf = outdir / "Figure_S21_eddy_slp_definition_sensitivity.pdf"
    figure.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def write_reports(
    s20: pd.DataFrame,
    s21: pd.DataFrame,
    grouped: pd.DataFrame,
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    object_audit: pd.DataFrame,
    outdir: Path,
) -> None:
    baseline = audit.set_index("definition_id").loc[BASELINE_ID]
    exact_main = int(validation["matches_current_main_catalog_no_area"].sum())
    exact_requested = int(validation["requested_area_filtered_matches_current_main_catalog"].sum())
    area_differences = validation[~validation["requested_area_filtered_matches_current_main_catalog"]]
    direction_counts = {
        parameter: int(audit[f"direction_agrees_with_baseline_{parameter}"].sum())
        for parameter in ("b", "n", "s")
    }
    ranking_counts = {
        parameter: int(audit[f"ranking_agrees_with_baseline_{parameter}"].sum())
        for parameter in ("b", "n", "s")
    }
    old_baseline_text = "Unavailable"
    if OLD_S21_AUDIT.is_file():
        old = pd.read_csv(OLD_S21_AUDIT)
        old_row = old[old["definition_id"].eq("overlap_p1010_o0.8")]
        if len(old_row) == 1:
            row = old_row.iloc[0]
            old_baseline_text = (
                f"b={row['spearman_rho_b']:+.3f}, n={row['spearman_rho_n']:+.3f}, "
                f"s={row['spearman_rho_s']:+.3f}"
            )
    differing_lines = (
        "\n".join(
            f"- `{row.case}`: current catalog {row.current_main_catalog_duration_h:g} h; "
            f"area-filtered recomputation {row.requested_area_filtered_duration_h:g} h."
            for row in area_differences.itertuples(index=False)
        )
        if len(area_differences)
        else "- None. The 100,000-km² filter does not change the +10-hPa/0.8 baseline durations."
    )
    sign_exceptions = []
    ranking_exceptions = []
    for parameter in ("b", "n", "s"):
        for definition in audit.loc[~audit[f"direction_agrees_with_baseline_{parameter}"], "definition_id"]:
            sign_exceptions.append(f"{parameter}: {definition}")
        for definition in audit.loc[~audit[f"ranking_agrees_with_baseline_{parameter}"], "definition_id"]:
            ranking_exceptions.append(f"{parameter}: {definition}")
    report = f"""# Eddy-SLP recomputation audit for Figures S20 and S21

## Definitions

At every hourly output and latitude, eddy surface pressure is calculated as

\[
p_s'(\lambda,\phi,t)=p_s(\lambda,\phi,t)-[p_s](\phi,t).
\]

The zonal mean is removed independently at every time and latitude before the Northern Hemisphere object masks are constructed. The largest absolute residual zonal mean in the 45 processed files is `{object_audit['maximum_absolute_zonal_mean_residual_hpa'].max():.3e}` hPa.

All new object definitions use positive eddy-SLP thresholds of +5, +10, or +15 hPa and a minimum spherical area of 100,000 km². No full-field 1005-, 1010-, or 1015-hPa object threshold is used in the new calculations. Eight-neighbor spatial connectivity and cyclic longitude treatment follow the existing ConTrack implementation. Object areas use exact spherical grid-cell areas; centroids use area-weighted Cartesian unit vectors.

## Figure S20

Figure S20 uses +10-hPa eddy-SLP objects and the independent permissive identity tracker with an overlap threshold of 0.3. As in the original S20 tracker, overlap is intersection divided by the smaller pixel count and is used only to maintain one-to-one identity. The longest track is selected in every case without adding a 72-h selection gate. Mean path speed is the mean great-circle displacement between consecutive hourly area-weighted centroids divided by one hour.

Across the 45 cases, longest-track lifetime ranges from `{s20['maximum_qualifying_persistence_h'].min():.0f}` to `{s20['maximum_qualifying_persistence_h'].max():.0f}` h, and mean centroid path speed ranges from `{s20['mean_centroid_path_speed_m_s-1'].min():.2f}` to `{s20['mean_centroid_path_speed_m_s-1'].max():.2f}` m s⁻¹.

## Figure S21 baseline validation

The saved current main-analysis catalog is `{CURRENT_MAIN_CATALOG}`. Re-running its exact +10-hPa/0.8 definition without the newly requested area filter reproduces `{exact_main}/45` cases exactly. The requested +10-hPa/0.8 definition with the 100,000-km² minimum-area filter matches the saved main catalog in `{exact_requested}/45` cases.

{differing_lines}

The requested eddy-SLP baseline Spearman relationships are:

- b: ρ = `{baseline['spearman_rho_b']:+.3f}`
- n: ρ = `{baseline['spearman_rho_n']:+.3f}`
- s: ρ = `{baseline['spearman_rho_s']:+.3f}`

For comparison, the old full-field 1010-hPa/0.8 analysis reported `{old_baseline_text}`.

## Sensitivity and rankings

Across the 18 eddy-SLP definitions, the baseline relationship direction is retained in `{direction_counts['b']}/18` definitions for b, `{direction_counts['n']}/18` for n, and `{direction_counts['s']}/18` for s. The exact baseline category ranking is retained in `{ranking_counts['b']}/18`, `{ranking_counts['n']}/18`, and `{ranking_counts['s']}/18`, respectively.

Direction exceptions: {', '.join(sign_exceptions) if sign_exceptions else 'none'}.

Exact-ranking exceptions: {', '.join(ranking_exceptions) if ranking_exceptions else 'none'}.

The complete correlations, rankings, and agreement flags are in `S21_definition_audit.csv`.

## Mobility identity handling

The size-normalized analysis first constructs consecutive one-to-one identity tracks without imposing the 0.8 persistence gate. Candidate links require positive spatial overlap, area ratio ≥0.25, separation ≤2500 km, and either at least 5% area-weighted overlap or separation no greater than one mean equivalent radius. Requiring positive overlap prevents a jump to an unrelated nearby object. Splits and mergers retain only the highest-scoring parent-child link; unmatched branches start or end tracks. Temporary disappearance is not bridged. Qualifying mobility episodes require Δd/r_eq to remain at or below the selected 0.20, 0.30, or 0.40 bound at every consecutive hourly step for at least 72 h.

## Full-field versus eddy-SLP interpretation

The old figures tracked absolute high-pressure contours, whereas the new figures track anomalies relative to the evolving instantaneous zonal mean. The eddy-SLP results therefore isolate regional pressure anomalies and should not be expected to preserve every case duration, correlation magnitude, or exact category ranking from the full-field analysis. The comparison is methodological rather than a relabeling of the original values.
"""
    (outdir / "validation_report.md").write_text(report)

    readme = f"""# Figures S20 and S21 recomputed from eddy surface pressure

This package contains a complete, non-destructive recomputation for the 45-member constant-u0/standard ensemble. Existing manuscript, Supporting Information, response-letter, scripts, data tables, and figures were not overwritten.

## Main files

- `Figure_S20_eddy_slp_lifetime_centroid_speed.png` and `.pdf`
- `Figure_S21_eddy_slp_definition_sensitivity.png` and `.pdf`
- `eddy_slp_anticyclone_case_level_all_definitions.csv`
- `S20_case_level_results.csv`
- `S21_case_level_results.csv`
- `S21_grouped_results.csv`
- `S21_definition_audit.csv`
- `S21_baseline_validation.csv`
- `eddy_slp_object_catalog_audit.csv`
- `validation_report.md`
- `recompute_figures_s20_s21_eddy_slp.py`

## Exact definitions

- Eddy SLP: `PRESsfc - instantaneous longitude mean(PRESsfc)` at each time and latitude.
- Domain: Northern Hemisphere latitude centers from 0.5° to 89.5°N.
- Sampling: all 360 instantaneous hourly fields.
- Minimum object area: 100,000 km².
- Figure S20: +10 hPa, independent 0.3 overlap identity linking, longest track.
- Figure S21 O definitions: +5/+10/+15 hPa and ConTrack overlap 0.6/0.8/0.9, minimum 72 h.
- Figure S21 M definitions: +5/+10/+15 hPa and Δd/r_eq ≤0.20/0.30/0.40, minimum 72 h.
- Baseline: +10 hPa and overlap 0.8.

## Reproduce

```bash
cd {ROOT}
MPLCONFIGDIR=/data/keeling/a/mingfei5/local/tmp/matplotlib \\
  $HOME/anaconda3/envs/rwb/bin/python \\
  {outdir / 'recompute_figures_s20_s21_eddy_slp.py'} \\
  --outdir {outdir} --workers 3
```

The 45 original approximately 8-GB model files are inputs and are not duplicated in this output package. Their exact paths and checksums are listed in `source_file_manifest.csv`.
"""
    (outdir / "README.md").write_text(readme)


def write_environment(outdir: Path) -> None:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "xarray": xr.__version__,
        "matplotlib": matplotlib.__version__,
        "netCDF4": __import__("netCDF4").__version__,
        "contrack_source": __import__("inspect").getfile(ContrackClass),
    }
    (outdir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in versions.items()) + "\n"
    )


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_source_manifest(paths: list[Path], outdir: Path) -> None:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({
            "case": path.stem,
            "path": str(path),
            "bytes": stat.st_size,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(rows).to_csv(outdir / "source_file_manifest.csv", index=False)


def write_checksums(outdir: Path) -> None:
    files = sorted(path for path in outdir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    text = "\n".join(f"{sha256_file(path)}  {path.name}" for path in files) + "\n"
    (outdir / "SHA256SUMS").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-source-checksums", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = standard_case_paths()

    s20_rows: list[dict[str, object]] = []
    s21_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    object_audit_rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_path = {executor.submit(process_case, str(path)): path for path in paths}
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_path), start=1):
            path = future_to_path[future]
            s20_row, case_rows, validation_row, object_rows, _ = future.result()
            s20_rows.append(s20_row)
            s21_rows.extend(case_rows)
            validation_rows.append(validation_row)
            object_audit_rows.extend(object_rows)
            print(f"[{completed:02d}/45] {path.stem}", flush=True)

    s20 = pd.DataFrame(s20_rows).sort_values(["s", "b", "n"]).reset_index(drop=True)
    s21 = pd.DataFrame(s21_rows).sort_values(
        ["method", "eddy_threshold_hpa", "criterion_value", "b", "n", "s"]
    ).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values(["b", "n", "s"]).reset_index(drop=True)
    object_audit = pd.DataFrame(object_audit_rows).sort_values(
        ["eddy_threshold_hpa", "b", "n", "s"]
    ).reset_index(drop=True)
    if len(s20) != 45:
        raise RuntimeError(f"Expected 45 S20 rows, found {len(s20)}")
    if len(s21) != 45 * 18:
        raise RuntimeError(f"Expected 810 S21 rows, found {len(s21)}")

    current = pd.read_csv(CURRENT_MAIN_CATALOG)
    current = current[current["ensemble"].eq("constant u0")][
        ["case", "eddy_duration_h", "eddy_retained_event_count"]
    ].rename(columns={
        "eddy_duration_h": "current_main_catalog_duration_h",
        "eddy_retained_event_count": "current_main_catalog_event_count",
    })
    validation = validation.merge(current, on="case", how="left", validate="one_to_one")
    validation["matches_current_main_catalog_no_area_duration"] = (
        validation["recomputed_no_area_duration_h"] == validation["current_main_catalog_duration_h"]
    )
    validation["matches_current_main_catalog_no_area_event_count"] = (
        validation["recomputed_no_area_event_count"] == validation["current_main_catalog_event_count"]
    )
    validation["matches_current_main_catalog_no_area"] = (
        validation["matches_current_main_catalog_no_area_duration"]
        & validation["matches_current_main_catalog_no_area_event_count"]
    )
    validation["requested_area_filtered_matches_current_main_catalog"] = (
        validation["requested_area_filtered_duration_h"] == validation["current_main_catalog_duration_h"]
    )
    validation["no_area_catalog_difference_h"] = (
        validation["recomputed_no_area_duration_h"] - validation["current_main_catalog_duration_h"]
    )
    validation["discrepancy_explanation"] = np.where(
        validation["requested_area_filtered_matches_current_main_catalog"],
        "none; the mandated 100,000-km2 area filter does not change the saved duration",
        np.where(
            (validation["current_main_catalog_duration_h"] >= MIN_DURATION_H)
            & (validation["requested_area_filtered_duration_h"] == 0),
            "the mandated 100,000-km2 hourly component filter interrupts the saved event, leaving no continuous segment of at least 72 h",
            "the mandated 100,000-km2 hourly component filter removes one or more small edge/onset/decay components and shortens the saved event",
        ),
    )

    grouped, audit = summarize_s21(s21)
    combined = pd.concat([s20, s21], ignore_index=True, sort=False)
    s20.to_csv(args.outdir / "S20_case_level_results.csv", index=False)
    s21.to_csv(args.outdir / "S21_case_level_results.csv", index=False)
    combined.to_csv(args.outdir / "eddy_slp_anticyclone_case_level_all_definitions.csv", index=False)
    grouped.to_csv(args.outdir / "S21_grouped_results.csv", index=False)
    audit.to_csv(args.outdir / "S21_definition_audit.csv", index=False)
    validation.to_csv(args.outdir / "S21_baseline_validation.csv", index=False)
    object_audit.to_csv(args.outdir / "eddy_slp_object_catalog_audit.csv", index=False)

    make_s20_figure(s20, args.outdir)
    make_s21_figure(grouped, audit, args.outdir)
    write_reports(s20, s21, grouped, audit, validation, object_audit, args.outdir)
    write_environment(args.outdir)
    if not args.skip_source_checksums:
        write_source_manifest(paths, args.outdir)
    else:
        pd.DataFrame({
            "case": [path.stem for path in paths],
            "path": [str(path) for path in paths],
            "bytes": [path.stat().st_size for path in paths],
            "sha256": ["not computed; use sha256sum on the source file if required" for path in paths],
        }).to_csv(args.outdir / "source_file_manifest.csv", index=False)
    run_script = args.outdir / "run_analysis.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {ROOT}\n"
        "export MPLCONFIGDIR=/data/keeling/a/mingfei5/local/tmp/matplotlib\n"
        f"$HOME/anaconda3/envs/rwb/bin/python {args.outdir / 'recompute_figures_s20_s21_eddy_slp.py'} "
        f"--outdir {args.outdir} --workers \"${{S20_S21_WORKERS:-3}}\"\n"
    )
    run_script.chmod(0o755)
    write_checksums(args.outdir)
    print(f"Outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
