#!/usr/bin/env python3
"""Segment rotation phases (per sensor) and run PCA on segmented force data.

This script is intended for auditing and analysis of the rotary manipulation
dataset under `data/`. It:

1) Segments each run into rotation "bursts" using *only* absolute raw Fz per
   sensor (Thumb/Ring/Index).
2) Combines the per-sensor burst windows into a single window (currently via
   intersection).
3) Runs PCA on the segmented samples for:
   - Raw Force  (Fx/Fy/Fz per sensor)
   - Transformed Force (Grav/Lat/Norm per sensor; taken from transformed_data)
4) Writes CSV outputs and (optionally) plotting panels for visual inspection.

Expected input layout (per run):
  `data/{user}/{movement}/test_{n}/raw_data.csv`
  `data/{user}/{movement}/test_{n}/transformed_data.csv`

Expected file format:
  The acquisition is commonly 18 columns (3 sensors × 6 channels), but this
  script only requires the force channels:
    - S1: cols 0..2  (Thumb Fx/Fy/Fz)
    - S2: cols 6..8  (Ring  Fx/Fy/Fz)
    - S3: cols 12..14 (Index Fx/Fy/Fz)

Primary outputs (default paths; can be overridden by CLI args):
  - `rotation_phase_segments.csv` (segmentation boundaries and audit metadata)
  - `pca_results_segmented_rotation.csv` (merged PCA results across all runs)
  - `pca_results_segmented_{user}_{movement}_{test}.csv` (per-run PCA results)
  - `plots_segmented_rotation/*.png` (if plotting enabled)

Notes for auditing:
  - Segmentation uses percentile thresholds on the activity envelope; it does
    *not* use timestamps, velocity, or any torque channels.
  - PCA is computed via eigen-decomposition of the covariance matrix of the
    centered data (no sklearn dependency).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The defaults are chosen to match the common data layout in this repo:
    - users: `kaiwen`, `tianshu`
    - movements: `yaw`, `roll`, `pitch`
    - tests: `1..5`

    Returns
    -------
    argparse.Namespace
        Parsed arguments for use by `main()`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="data", help="Root directory containing user/axis/test folders")
    parser.add_argument(
        "--users",
        nargs="+",
        default=["kaiwen", "tianshu"],
        help="Users to process",
    )
    parser.add_argument(
        "--movements",
        nargs="+",
        default=["yaw", "roll", "pitch"],
        help="Manipulation axes to process",
    )
    parser.add_argument("--tests", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Test numbers to process")
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=25.0,
        help="Percentile threshold for low-force detection in each sensor envelope",
    )
    parser.add_argument("--min-low-len", type=int, default=10, help="Minimum samples for low-force segment")
    parser.add_argument("--min-burst-len", type=int, default=8, help="Minimum samples for candidate burst")
    parser.add_argument(
        "--peak-percentile",
        type=float,
        default=65.0,
        help="Within-burst percentile threshold for valid peaks",
    )
    parser.add_argument("--peak-min-distance", type=int, default=3, help="Minimum sample distance between peaks")
    parser.add_argument(
        "--combine-mode",
        choices=["intersection"],
        default="intersection",
        help="How to combine per-sensor windows for final PCA slicing",
    )
    parser.add_argument(
        "--output-segment-csv",
        default="rotation_phase_segments.csv",
        help="CSV path for segmentation boundaries",
    )
    parser.add_argument(
        "--output-pca-csv",
        default="pca_results_segmented_rotation.csv",
        help="CSV path for merged segmented PCA results",
    )
    parser.add_argument(
        "--per-run-prefix",
        default="pca_results_segmented",
        help="Prefix for per-run PCA result CSV files",
    )
    parser.add_argument(
        "--plot-dir",
        default="plots_segmented_rotation",
        help="Output directory for panel figures",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--template-sign-align",
        action="store_true",
        help="Apply template-correlation sign alignment before plotting",
    )
    return parser.parse_args()


def sensor_block_indices(sensor_name: str) -> List[int]:
    """Return the (Fx, Fy, Fz) column indices for a given sensor block.

    The acquisition commonly writes 18 columns as 3 sensors × 6 channels. This
    pipeline assumes the block order:
      - Thumb (Sensor 1) starts at col 0
      - Ring  (Sensor 2) starts at col 6
      - Index (Sensor 3) starts at col 12

    Parameters
    ----------
    sensor_name:
        One of: `"Thumb"`, `"Ring"`, `"Index"`.

    Returns
    -------
    list[int]
        Three indices `[Fx_col, Fy_col, Fz_col]`.
    """
    base = {
        "Thumb": 0,
        "Ring": 6,
        "Index": 12,
    }[sensor_name]
    return [base, base + 1, base + 2]


def read_matrix(csv_path: str) -> "np.ndarray":
    """Load a numeric CSV into a 2D numpy array.

    Parameters
    ----------
    csv_path:
        Path to a numeric CSV file. Typically `raw_data.csv` or
        `transformed_data.csv`.

    Returns
    -------
    np.ndarray
        Shape (N, D), where N is the number of samples and D is the number of
        columns.

    Raises
    ------
    ValueError
        If the file has too few columns to support the force indices used by
        this script (requires at least columns 0..14).
    """
    import numpy as np

    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 15:
        raise ValueError(f"Unexpected column count in {csv_path}: {data.shape[1]}")
    return data


def contiguous_true_segments(mask: "np.ndarray", min_len: int) -> List[Tuple[int, int]]:
    """Convert a boolean mask into inclusive contiguous segments.

    Parameters
    ----------
    mask:
        Boolean vector.
    min_len:
        Minimum inclusive segment length to keep.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive (start, end) segments where mask is True for all samples.
    """
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, is_true in enumerate(mask):
        if is_true and start is None:
            start = idx
        elif not is_true and start is not None:
            end = idx - 1
            if end - start + 1 >= min_len:
                segments.append((start, end))
            start = None
    if start is not None:
        end = len(mask) - 1
        if end - start + 1 >= min_len:
            segments.append((start, end))
    return segments


def select_peaks(signal: "np.ndarray", peak_percentile: float, min_distance: int) -> List[int]:
    """Select peaks (local maxima) from a 1D signal.

    A "peak" is a sample that is greater than the previous sample and greater
    than or equal to the next sample. Peaks are filtered by:
    - amplitude: must be >= `percentile(signal, peak_percentile)`
    - spacing: enforce at least `min_distance` samples between kept peaks

    Parameters
    ----------
    signal:
        1D signal vector (typically a burst of the activity envelope).
    peak_percentile:
        Percentile threshold for amplitude filtering.
    min_distance:
        Minimum sample distance enforced between peaks.

    Returns
    -------
    list[int]
        Sorted list of peak indices (relative to the input `signal`).
    """
    import numpy as np

    if len(signal) < 3:
        return []

    candidates = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]))[0] + 1
    if len(candidates) == 0:
        return []

    amp_threshold = np.percentile(signal, peak_percentile)
    candidates = [idx for idx in candidates if signal[idx] >= amp_threshold]
    if not candidates:
        return []

    # Keep stronger peaks first, then enforce spacing.
    candidates_sorted = sorted(candidates, key=lambda i: float(signal[i]), reverse=True)
    kept: List[int] = []
    for idx in candidates_sorted:
        if all(abs(idx - k) >= min_distance for k in kept):
            kept.append(idx)

    return sorted(kept)


def segment_sensor_bursts(
    activity: "np.ndarray",
    low_percentile: float,
    min_low_len: int,
    min_burst_len: int,
    peak_percentile: float,
    peak_min_distance: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Segment a single sensor activity envelope into bursts.

    The segmentation logic is designed to be deterministic and audit-friendly:
    1) Compute a low-force threshold using `low_percentile`.
    2) Find contiguous low-force segments (length >= `min_low_len`).
    3) For each interval between consecutive low-force segments, treat it as a
       candidate burst (length >= `min_burst_len`).
    4) Within each candidate burst, select peaks using `select_peaks`.
    5) Trim the burst to [first_peak, last_peak] if >=2 peaks are found;
       otherwise keep the full candidate burst.

    Parameters
    ----------
    activity:
        1D activity envelope (e.g., `abs(Fz)` for one sensor).
    low_percentile:
        Percentile threshold for low-force detection.
    min_low_len:
        Minimum length of a low-force segment.
    min_burst_len:
        Minimum length of a candidate burst between low segments.
    peak_percentile:
        Percentile threshold within the burst for peak filtering.
    peak_min_distance:
        Minimum distance between peaks.

    Returns
    -------
    (low_segments, raw_bursts, trimmed_bursts)
        Each element is a list of inclusive (start, end) intervals.
    """
    import numpy as np

    # Low-force is defined by a percentile threshold on the activity envelope,
    # which is robust to scale differences between sensors/runs.
    low_threshold = float(np.percentile(activity, low_percentile))
    low_mask = activity <= low_threshold
    low_segments = contiguous_true_segments(low_mask, min_low_len)

    raw_bursts: List[Tuple[int, int]] = []
    trimmed_bursts: List[Tuple[int, int]] = []

    for idx in range(len(low_segments) - 1):
        burst_start = low_segments[idx][1] + 1
        burst_end = low_segments[idx + 1][0] - 1
        if burst_end - burst_start + 1 < min_burst_len:
            continue
        raw_bursts.append((burst_start, burst_end))

        # Find peaks in the candidate burst and trim to the window bounded by
        # the first and last peaks. This removes leading/trailing low activity
        # within the burst while keeping an audit trail (raw vs trimmed).
        burst_signal = activity[burst_start : burst_end + 1]
        peaks = select_peaks(burst_signal, peak_percentile=peak_percentile, min_distance=peak_min_distance)
        if len(peaks) >= 2:
            trimmed_start = burst_start + peaks[0]
            trimmed_end = burst_start + peaks[-1]
        else:
            trimmed_start = burst_start
            trimmed_end = burst_end
        if trimmed_end >= trimmed_start:
            trimmed_bursts.append((trimmed_start, trimmed_end))

    return low_segments, raw_bursts, trimmed_bursts


def intersect_two(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Compute the intersection of two inclusive interval lists.

    Parameters
    ----------
    a, b:
        Sequences of inclusive (start, end) intervals. Intervals are assumed to
        be sorted and non-overlapping within each list (as produced by this
        script).

    Returns
    -------
    list[tuple[int, int]]
        Inclusive intersections of a and b, sorted by start index.
    """
    out: List[Tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start <= end:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def intersect_three(
    a: Sequence[Tuple[int, int]],
    b: Sequence[Tuple[int, int]],
    c: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Compute the intersection of three inclusive interval lists."""
    return intersect_two(intersect_two(a, b), c)


def intervals_to_indices(intervals: Sequence[Tuple[int, int]]) -> List[int]:
    """Expand inclusive intervals into a flat list of sample indices.

    This is used to slice the force matrices so that PCA is computed only on
    the combined burst samples.
    """
    indices: List[int] = []
    for start, end in intervals:
        indices.extend(range(start, end + 1))
    return indices


def compute_run_pca(matrix_9: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Compute PCA via eigen-decomposition of the covariance matrix.

    Parameters
    ----------
    matrix_9:
        Array of shape (N, 9) where N is the number of segmented samples and
        the 9 features are 3 sensors × 3 force components.

    Returns
    -------
    (eigvals, eigvecs, var_ratio)
        - eigvals: eigenvalues (descending)
        - eigvecs: eigenvectors / PCA loadings (columns correspond to PCs)
        - var_ratio: eigvals normalized to sum to 1 (if possible)
    """
    import numpy as np

    if matrix_9.shape[0] < 3:
        raise ValueError("Not enough segmented samples for PCA.")

    # Center the data (no scaling) to preserve physical relationships between
    # force components.
    centered = matrix_9 - np.mean(matrix_9, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.sum(eigvals))
    if total <= 0:
        var_ratio = np.zeros_like(eigvals)
    else:
        var_ratio = eigvals / total
    return eigvals, eigvecs, var_ratio


def collect_runs(args: argparse.Namespace) -> List[Tuple[str, str, int, str, str]]:
    """Collect valid runs from disk based on CLI filters.

    A "run" is considered valid if both `raw_data.csv` and `transformed_data.csv`
    exist under:
      `data_root/user/movement/test_{n}/`
    """
    runs: List[Tuple[str, str, int, str, str]] = []
    for user in args.users:
        for movement in args.movements:
            for test in args.tests:
                run_dir = os.path.join(args.data_root, user, movement, f"test_{test}")
                raw_path = os.path.join(run_dir, "raw_data.csv")
                transformed_path = os.path.join(run_dir, "transformed_data.csv")
                if os.path.exists(raw_path) and os.path.exists(transformed_path):
                    runs.append((user, movement, test, raw_path, transformed_path))
    return runs


def write_segment_rows(rows: Sequence[Dict[str, str]], out_csv: str) -> None:
    """Write segmentation audit rows to a CSV file."""
    if not rows:
        return
    fieldnames = [
        "User",
        "Movement",
        "Test Number",
        "Sensor",
        "Segment Kind",
        "Start",
        "End",
        "Length",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pca_rows(rows: Sequence[Dict[str, str]], out_csv: str) -> None:
    """Write PCA result rows to a CSV file."""
    if not rows:
        return
    fieldnames = [
        "Data Type",
        "User",
        "Rotation Axis",
        "Movement",
        "Test Number",
        "Principal Component",
        "Sensor",
        "Loading Weight",
        "Eigenvalue",
        "Variance Ratio",
        "Segmented Samples",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def apply_template_sign_alignment(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Apply sign alignment to PCA loading vectors for plotting/comparison.

    PCA eigenvectors are sign-ambiguous: if v is an eigenvector, then -v is
    equally valid. This function chooses a consistent sign across runs by:
    - Building a per-(Data Type, Movement, PC) template vector whose components
      are the median loading per Sensor.
    - For each run vector, flipping its sign if the dot product with the
      template is negative.

    This does not change the underlying PCA model; it only makes plots and
    across-run comparisons easier to interpret.
    """
    import numpy as np

    corrected = [dict(r) for r in rows]
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in corrected:
        grouped[(row["Data Type"], row["Movement"], row["Principal Component"])].append(row)

    run_vectors: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = defaultdict(dict)
    for row in corrected:
        rkey = (
            row["Data Type"],
            row["Movement"],
            row["User"],
            row["Test Number"],
            row["Principal Component"],
        )
        run_vectors[rkey][row["Sensor"]] = float(row["Loading Weight"])

    template_vectors: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for gkey, group_rows in grouped.items():
        sensors = sorted({row["Sensor"] for row in group_rows})
        template: Dict[str, float] = {}
        for sensor in sensors:
            values = [float(r["Loading Weight"]) for r in group_rows if r["Sensor"] == sensor]
            template[sensor] = float(np.median(values))
        template_vectors[gkey] = template

    flip_sign: Dict[Tuple[str, str, str, str, str], float] = {}
    for rkey, vec in run_vectors.items():
        tkey = (rkey[0], rkey[1], rkey[4])
        template = template_vectors.get(tkey, {})
        common = set(vec.keys()) & set(template.keys())
        dot = sum(vec[s] * template[s] for s in common)
        flip_sign[rkey] = -1.0 if dot < 0 else 1.0

    for row in corrected:
        rkey = (
            row["Data Type"],
            row["Movement"],
            row["User"],
            row["Test Number"],
            row["Principal Component"],
        )
        row["Loading Weight"] = str(float(row["Loading Weight"]) * flip_sign.get(rkey, 1.0))

    return corrected


def pc_key(pc: str) -> int:
    """Sort key for PC labels like 'PC1', 'PC2', ... (unknowns go last)."""
    if pc.startswith("PC"):
        try:
            return int(pc[2:])
        except ValueError:
            pass
    return 10**9


def sensor_plot_order(sensor: str) -> Tuple[int, int, str]:
    """Sort sensors by digit (Thumb->Index->Ring...) then component (Fx/Fy/Fz)."""
    digit, component = sensor.split()
    digit_priority = {"Thumb": 0, "Index": 1, "Ring": 2, "Middle": 3, "Pinky": 4}
    comp_priority = {"Fx": 0, "Fy": 1, "Fz": 2, "Grav": 0, "Lat": 1, "Norm": 2}
    return (digit_priority.get(digit, 999), comp_priority.get(component, 999), sensor)


def sensor_plot_order_component_grouped(sensor: str) -> Tuple[int, int, str]:
    """Sort sensors by component (Fx/Fy/Fz or Grav/Lat/Norm) then digit."""
    digit, component = sensor.split()
    digit_priority = {"Thumb": 0, "Index": 1, "Ring": 2, "Middle": 3, "Pinky": 4}
    comp_priority = {"Fx": 0, "Fy": 1, "Fz": 2, "Grav": 0, "Lat": 1, "Norm": 2}
    return (comp_priority.get(component, 999), digit_priority.get(digit, 999), sensor)


def plot_panel(rows: Sequence[Dict[str, str]], plot_dir: str, component_grouped: bool) -> None:
    """Create 9×1 PCA loading panels per movement and data type.

    Parameters
    ----------
    rows:
        PCA result rows (typically `pca_rows` or sign-aligned variant).
    plot_dir:
        Output directory for `.png` files.
    component_grouped:
        If True, sort x-axis as (component, digit). If False, sort as
        (digit, component).
    """
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    data_types = sorted({r["Data Type"] for r in rows})

    for data_type in data_types:
        dt_rows = [r for r in rows if r["Data Type"] == data_type]
        movements = sorted({r["Movement"] for r in dt_rows})
        pcs = [pc for pc in sorted({r["Principal Component"] for r in dt_rows}, key=pc_key) if pc_key(pc) <= 9]

        for movement in movements:
            sub = [r for r in dt_rows if r["Movement"] == movement]
            if not sub:
                continue

            sensor_key = sensor_plot_order_component_grouped if component_grouped else sensor_plot_order
            sensors = sorted({r["Sensor"] for r in sub}, key=sensor_key)
            sensor_to_x = {s: i for i, s in enumerate(sensors)}
            users = sorted({r["User"] for r in sub})
            tests = sorted({int(r["Test Number"]) for r in sub})

            values: Dict[Tuple[str, str, int, str], float] = {}
            for r in sub:
                values[(r["Principal Component"], r["User"], int(r["Test Number"]), r["Sensor"])] = float(
                    r["Loading Weight"]
                )

            # One figure per (data_type, movement). Subplots are PC1..PC9, with
            # each line corresponding to a (user, test) run.
            fig, axes = plt.subplots(
                nrows=len(pcs),
                ncols=1,
                figsize=(max(10.0, len(sensors) * 0.7), max(12.0, len(pcs) * 2.1)),
                sharex=True,
            )
            if len(pcs) == 1:
                axes = [axes]

            user_colors = {"kaiwen": "#1f77b4", "tianshu": "#d62728"}
            fallback = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e"])
            user_to_color = {u: user_colors.get(u, fallback[idx % len(fallback)]) for idx, u in enumerate(users)}
            linestyles = ["-", "--", "-.", ":"]
            markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

            legend_handles = {}
            for ridx, pc in enumerate(pcs):
                ax = axes[ridx]
                for user in users:
                    for test in tests:
                        points = []
                        for sensor in sensors:
                            value = values.get((pc, user, test, sensor))
                            if value is not None:
                                points.append((sensor_to_x[sensor], value))
                        if not points:
                            continue

                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        label = f"{user} T{test}"
                        handle = ax.plot(
                            xs,
                            ys,
                            color=user_to_color[user],
                            linestyle=linestyles[(test - 1) % len(linestyles)],
                            marker=markers[(test - 1) % len(markers)],
                            markersize=3.5,
                            linewidth=1.0,
                            alpha=0.9,
                            label=label,
                        )[0]
                        legend_handles[label] = handle

                ax.axhline(0.0, color="0.75", linewidth=0.7)
                ax.set_ylabel(f"{pc}\nLoading", fontsize=8)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", linewidth=0.4, alpha=0.25)
                ax.set_xlim(-0.5, len(sensors) - 0.5)
                if ridx < len(pcs) - 1:
                    ax.tick_params(labelbottom=False)

            axes[-1].set_xticks(list(range(len(sensors))))
            axes[-1].set_xticklabels(sensors, rotation=45, ha="right", fontsize=7)
            fig.suptitle(f"{data_type} (Segmented Rotation) - {movement}", fontsize=12)
            if component_grouped:
                fig.supxlabel("Sensor (Component grouped: Grav/Lat/Norm or Fx/Fy/Fz; then Thumb/Index/Ring)", fontsize=10)
            else:
                fig.supxlabel("Sensor (Digit grouped: Thumb -> Index -> Ring)", fontsize=10)
            fig.supylabel("Loading Weight", fontsize=10)
            if legend_handles:
                fig.legend(
                    list(legend_handles.values()),
                    list(legend_handles.keys()),
                    loc="upper right",
                    fontsize=7,
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=(0.995, 0.995),
                )
            fig.tight_layout(rect=[0.03, 0.03, 0.96, 0.98])

            slug = data_type.lower().replace(" ", "_")
            order_slug = "component_grouped" if component_grouped else "digit_grouped"
            out_path = os.path.join(plot_dir, f"{slug}_{movement}_pc1_to_pc9_segmented_{order_slug}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)


def main() -> None:
    """Run segmentation + PCA over all selected runs and write outputs."""
    args = parse_args()

    import numpy as np

    # This order must match the acquisition column blocks assumed by
    # `sensor_block_indices()`.
    sensor_order = ["Thumb", "Ring", "Index"]

    runs = collect_runs(args)
    if not runs:
        raise SystemExit("No runs found with both raw_data.csv and transformed_data.csv")

    segment_rows: List[Dict[str, str]] = []
    pca_rows: List[Dict[str, str]] = []

    for user, movement, test, raw_path, transformed_path in runs:
        # Load the run matrices. We keep full arrays, but only slice out the
        # force columns needed for segmentation and PCA.
        raw = read_matrix(raw_path)
        transformed = read_matrix(transformed_path)

        fz_activity: Dict[str, np.ndarray] = {}
        for sensor in sensor_order:
            cols = sensor_block_indices(sensor)
            # Segmentation is based only on raw Fz force (absolute amplitude).
            # This is intentionally simple and easy to audit.
            sensor_fz = raw[:, cols[2]]
            fz_activity[sensor] = np.abs(sensor_fz)

        sensor_trimmed: Dict[str, List[Tuple[int, int]]] = {}
        for sensor in sensor_order:
            # Segment each sensor independently, then later combine windows.
            low_segments, raw_bursts, trimmed = segment_sensor_bursts(
                fz_activity[sensor],
                low_percentile=args.low_percentile,
                min_low_len=args.min_low_len,
                min_burst_len=args.min_burst_len,
                peak_percentile=args.peak_percentile,
                peak_min_distance=args.peak_min_distance,
            )
            sensor_trimmed[sensor] = trimmed

            # Persist all intermediate segments for auditing/debugging.
            for start, end in low_segments:
                segment_rows.append(
                    {
                        "User": user,
                        "Movement": movement,
                        "Test Number": str(test),
                        "Sensor": sensor,
                        "Segment Kind": "low_force",
                        "Start": str(start),
                        "End": str(end),
                        "Length": str(end - start + 1),
                    }
                )
            for start, end in raw_bursts:
                segment_rows.append(
                    {
                        "User": user,
                        "Movement": movement,
                        "Test Number": str(test),
                        "Sensor": sensor,
                        "Segment Kind": "burst_between_low",
                        "Start": str(start),
                        "End": str(end),
                        "Length": str(end - start + 1),
                    }
                )
            for start, end in trimmed:
                segment_rows.append(
                    {
                        "User": user,
                        "Movement": movement,
                        "Test Number": str(test),
                        "Sensor": sensor,
                        "Segment Kind": "trimmed_first_last_peak",
                        "Start": str(start),
                        "End": str(end),
                        "Length": str(end - start + 1),
                    }
                )

        combined_intervals = intersect_three(
            sensor_trimmed["Thumb"],
            sensor_trimmed["Ring"],
            sensor_trimmed["Index"],
        )

        # The combined intervals are the only samples used downstream for PCA.
        for start, end in combined_intervals:
            segment_rows.append(
                {
                    "User": user,
                    "Movement": movement,
                    "Test Number": str(test),
                    "Sensor": "Combined",
                    "Segment Kind": "intersection",
                    "Start": str(start),
                    "End": str(end),
                    "Length": str(end - start + 1),
                }
            )

        sample_idx = intervals_to_indices(combined_intervals)
        if len(sample_idx) < 3:
            # Not enough segmented samples to compute a stable covariance/PCA.
            continue

        # Construct the 9D feature vectors as 3 sensors × 3 force components.
        raw_force_cols = sensor_block_indices("Thumb") + sensor_block_indices("Ring") + sensor_block_indices("Index")
        transformed_force_cols = raw_force_cols

        for data_type, matrix, cols, components in [
            ("Raw Force", raw, raw_force_cols, ["Fx", "Fy", "Fz"]),
            ("Transformed Force", transformed, transformed_force_cols, ["Grav", "Lat", "Norm"]),
        ]:
            sampled = matrix[sample_idx, :][:, cols]
            try:
                eigvals, eigvecs, var_ratio = compute_run_pca(sampled)
            except ValueError:
                continue

            # The columns of eigvecs are PCs; rows correspond to the 9 features
            # in the order used to build `sampled`.
            digits = ["Thumb", "Ring", "Index"]
            sensor_labels = [f"{digit} {comp}" for digit in digits for comp in components]

            per_run_rows: List[Dict[str, str]] = []
            for pc_idx in range(min(9, eigvecs.shape[1])):
                pc_name = f"PC{pc_idx + 1}"
                for feat_idx, sensor_name in enumerate(sensor_labels):
                    row = {
                        "Data Type": data_type,
                        "User": user,
                        "Rotation Axis": movement,
                        "Movement": movement,
                        "Test Number": str(test),
                        "Principal Component": pc_name,
                        "Sensor": sensor_name,
                        "Loading Weight": str(float(eigvecs[feat_idx, pc_idx])),
                        "Eigenvalue": str(float(eigvals[pc_idx])),
                        "Variance Ratio": str(float(var_ratio[pc_idx])),
                        "Segmented Samples": str(len(sample_idx)),
                    }
                    pca_rows.append(row)
                    per_run_rows.append(row)

            per_run_file = f"{args.per_run_prefix}_{user}_{movement}_{test}.csv"
            write_pca_rows(per_run_rows, per_run_file)

    # Write merged outputs after all runs are processed.
    write_segment_rows(segment_rows, args.output_segment_csv)
    write_pca_rows(pca_rows, args.output_pca_csv)

    plot_rows = pca_rows
    if args.template_sign_align:
        # Only affects plotting; the raw CSV outputs remain unaligned.
        plot_rows = apply_template_sign_alignment(plot_rows)

    if not args.skip_plots and plot_rows:
        try:
            plot_panel(plot_rows, args.plot_dir, component_grouped=False)
            plot_panel(plot_rows, args.plot_dir, component_grouped=True)
        except ImportError:
            raise SystemExit("matplotlib is required for plotting. Install it or run with --skip-plots.")

    print(f"Processed runs: {len(runs)}")
    print(f"Saved segment boundaries: {args.output_segment_csv}")
    print(f"Saved merged segmented PCA: {args.output_pca_csv}")
    print(f"Saved per-run PCA files with prefix: {args.per_run_prefix}")
    if args.skip_plots:
        print("Plot generation skipped (--skip-plots).")
    else:
        print(f"Saved panel figures in: {args.plot_dir}")


if __name__ == "__main__":
    main()
