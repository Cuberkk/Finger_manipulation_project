#!/usr/bin/env python3
"""Create segmentation audit plots by overlaying segmentation points on trajectories.

Segmentation follows `segment_rotation_pca_pipeline.py`:
- per-sensor absolute Fz activity signal
- per-sensor low-force segments
- bursts between consecutive low-force segments
- trimmed bursts from first to last peak
- final combined windows via intersection across Thumb/Ring/Index
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Sequence, Tuple

from segment_rotation_pca_pipeline import (
    collect_runs,
    read_matrix,
    sensor_block_indices,
    segment_sensor_bursts,
    intersect_three,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root directory containing user/axis/test folders")
    parser.add_argument("--users", nargs="+", default=["kaiwen", "tianshu"], help="Users to process")
    parser.add_argument("--movements", nargs="+", default=["yaw", "roll", "pitch"], help="Axes to process")
    parser.add_argument("--tests", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Test numbers")
    parser.add_argument(
        "--visualize-data",
        choices=["raw", "transformed"],
        default="transformed",
        help="Which trajectory data to plot under segmentation overlays",
    )
    parser.add_argument("--low-percentile", type=float, default=25.0, help="Low-force threshold percentile")
    parser.add_argument("--min-low-len", type=int, default=10, help="Minimum low-force segment length")
    parser.add_argument("--min-burst-len", type=int, default=8, help="Minimum burst length")
    parser.add_argument("--peak-percentile", type=float, default=65.0, help="Peak threshold percentile")
    parser.add_argument("--peak-min-distance", type=int, default=3, help="Minimum peak spacing in samples")
    parser.add_argument(
        "--output-dir",
        default="plots_segment_audit",
        help="Directory for audit figures",
    )
    return parser.parse_args()


def extract_force_9(matrix18: Any) -> Any:
    cols = sensor_block_indices("Thumb") + sensor_block_indices("Ring") + sensor_block_indices("Index")
    return matrix18[:, cols]


def draw_interval_spans(ax: Any, intervals: Sequence[Tuple[int, int]], color: str, alpha: float) -> None:
    for start, end in intervals:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)


def add_boundary_markers(ax: Any, intervals: Sequence[Tuple[int, int]], color: str, label: str) -> None:
    starts = [s for s, _ in intervals]
    ends = [e for _, e in intervals]
    if starts:
        y0, y1 = ax.get_ylim()
        marker_y = y0 + 0.95 * (y1 - y0)
        ax.plot(starts, [marker_y] * len(starts), marker="|", linestyle="None", color=color, markersize=10, label=label)
        ax.plot(ends, [marker_y] * len(ends), marker="|", linestyle="None", color=color, markersize=10)


def run_audit(args: argparse.Namespace) -> int:
    try:
        import numpy as np
    except ImportError:
        print("numpy is required for audit plotting. Install it and rerun.")
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for audit plotting. Install it and rerun.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    runs = collect_runs(args)
    if not runs:
        print("No runs found with both raw_data.csv and transformed_data.csv")
        return 1

    sensor_order = ["Thumb", "Ring", "Index"]
    component_labels = {
        "raw": ["Fx", "Fy", "Fz"],
        "transformed": ["Grav", "Lat", "Norm"],
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for user, movement, test, raw_path, transformed_path in runs:
        raw = read_matrix(raw_path)
        transformed = read_matrix(transformed_path)
        base = transformed if args.visualize_data == "transformed" else raw
        force9 = extract_force_9(base)

        sensor_segments = {}
        for sensor in sensor_order:
            cols = sensor_block_indices(sensor)
            # Segmentation is based only on Fz force magnitude for this sensor.
            envelope = np.abs(raw[:, cols[2]])
            low_segments, raw_bursts, trimmed = segment_sensor_bursts(
                envelope,
                low_percentile=args.low_percentile,
                min_low_len=args.min_low_len,
                min_burst_len=args.min_burst_len,
                peak_percentile=args.peak_percentile,
                peak_min_distance=args.peak_min_distance,
            )
            sensor_segments[sensor] = {
                "low": low_segments,
                "raw_bursts": raw_bursts,
                "trimmed": trimmed,
            }

        combined = intersect_three(
            sensor_segments["Thumb"]["trimmed"],
            sensor_segments["Ring"]["trimmed"],
            sensor_segments["Index"]["trimmed"],
        )

        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for sidx, sensor in enumerate(sensor_order):
            ax = axes[sidx]
            base_col = sidx * 3
            for cidx in range(3):
                ax.plot(
                    force9[:, base_col + cidx],
                    color=colors[cidx],
                    linewidth=1.0,
                    label=component_labels[args.visualize_data][cidx],
                )

            draw_interval_spans(ax, sensor_segments[sensor]["low"], color="#bdbdbd", alpha=0.15)
            draw_interval_spans(ax, sensor_segments[sensor]["raw_bursts"], color="#9ecae1", alpha=0.10)
            draw_interval_spans(ax, sensor_segments[sensor]["trimmed"], color="#3182bd", alpha=0.12)
            draw_interval_spans(ax, combined, color="#e41a1c", alpha=0.10)
            add_boundary_markers(ax, sensor_segments[sensor]["trimmed"], color="#08519c", label="trim boundaries")

            ax.set_title(f"{sensor} ({args.visualize_data})")
            ax.set_ylabel("Force")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8, ncol=4)

        axes[-1].set_xlabel("Sample")
        fig.suptitle(
            f"Segmentation Audit: {user} | {movement} | test_{test}"
            f"\nGray=low-force, LightBlue=between-low burst, Blue=trimmed burst, Red=intersection",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_name = f"segment_audit_{args.visualize_data}_{user}_{movement}_{test}.png"
        out_path = os.path.join(args.output_dir, out_name)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    print(f"Saved audit figures to: {args.output_dir}")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_audit(args))


if __name__ == "__main__":
    main()
