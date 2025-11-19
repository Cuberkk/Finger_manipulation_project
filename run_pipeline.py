#!/usr/bin/env python3
"""
run_pipeline.py
---------------
High-level pipeline that uses readers.py:

Flow:
  1) Compute bias FIRST (default 30 s @ 20 Hz) while sensors are unloaded.
  2) Manipulation phase: collect RAW calibrated data (default 20 s @ 20 Hz) while you move sensors.
  3) Save RAW and BIASED CSVs for that manipulation window.
  4) Plot per-sensor Forces and Torques with labels and grids.

Tweak durations, filenames, or plotting here without touching the hardware code.
"""

import os
import time
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from readers import ThreeSensorReader  # ← our hardware layer


# ─────────────────────────────────────────────────────────────────────────────
# CSV streaming helpers
# ─────────────────────────────────────────────────────────────────────────────
def stream_to_csv(read_fn, hz, seconds, out_csv):
    """
    Call `read_fn()` (returns np.ndarray (18,)) at `hz` for `seconds`,
    write timestamped rows into `out_csv` with labeled columns.
    """
    folder = os.path.dirname(out_csv)
    path = os.path.join("data", folder) if folder else "data"
    os.makedirs(path, exist_ok=True)
    # os.makedirs(rf"{os.path.dirname(out_csv)}" or "./data", exist_ok=True)

    # header: timestamp + Fx1..Tz1 + Fx2..Tz2 + Fx3..Tz3
    header = ["timestamp_s"] + [f"{lbl}{i+1}" for i in range(3) for lbl in ["Fx","Fy","Fz","Tx","Ty","Tz"]]

    total = int(seconds * hz)
    print(f"[Stream] {seconds:.1f}s @ {hz:.1f}Hz → {out_csv}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        # phase-locked loop to keep cadence tight
        next_t = time.perf_counter()
        for _ in range(total):
            ts = time.time()
            data = read_fn()  # shape (18,)
            w.writerow([ts] + data.tolist())

            next_t += 1.0 / hz
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_csv(csv_path, title_prefix):
    """
    Read a timestamped CSV produced by stream_to_csv() and create six labeled plots per sensor:
      - Forces (Fx,Fy,Fz)
      - Torques (Tx,Ty,Tz)
    Saves PNGs next to the CSV.
    """
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    if arr.size == 0:
        print(f"[Plot] No data in {csv_path}. Skipping plots.")
        return

    # Convert to time from start
    t = arr["timestamp_s"] - arr["timestamp_s"][0]
    labels = ["Fx","Fy","Fz","Tx","Ty","Tz"]

    for si in (1, 2, 3):
        # Gather columns for this sensor
        chans = [arr[f"{lbl}{si}"] for lbl in labels]

        # Forces
        plt.figure(figsize=(10, 4))
        plt.plot(t, chans[0], label="Fx")
        plt.plot(t, chans[1], label="Fy")
        plt.plot(t, chans[2], label="Fz")
        plt.title(f"{title_prefix} – Sensor {si} Forces")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_png = f"{os.path.splitext(csv_path)[0]}_S{si}_forces.png"
        plt.savefig(out_png, dpi=150)

        # Torques
        plt.figure(figsize=(10, 4))
        plt.plot(t, chans[3], label="Tx")
        plt.plot(t, chans[4], label="Ty")
        plt.plot(t, chans[5], label="Tz")
        plt.title(f"{title_prefix} – Sensor {si} Torques")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (N·mm)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_png = f"{os.path.splitext(csv_path)[0]}_S{si}_torques.png"
        plt.savefig(out_png, dpi=150)

    print(f"[Plot] Saved plots next to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI orchestration: bias → manipulation raw → manipulation biased → plots
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bias first, then manipulation phase (RAW + BIASED) with plots."
    )
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Sampling frequency (Hz) for bias and manipulation")
    parser.add_argument("--bias-seconds", type=float, default=15.0,
                        help="Seconds to average for bias (keep sensors unloaded)")
    parser.add_argument("--manip-seconds", type=float, default=20.0,
                        help="Seconds to record while you manipulate sensors")
    parser.add_argument("--lj-sensor", choices=["44297","44298"], default="44297",
                        help="LabJack sensor calibration to use")
    parser.add_argument("--ni-cal1", default="calibration_files/FT44298.cal",
                        help="ATI .cal XML for NI sensor #1 (sensor 1 → indices 0..5)")
    parser.add_argument("--ni-cal2", default="calibration_files/FT45281.cal",
                        help="ATI .cal XML for NI sensor #2 (sensor 2 → indices 6..11)")
    parser.add_argument("--out-raw", default="manip_raw_3sensors.csv",
                        help="Output CSV for manipulation RAW (calibrated) data")
    parser.add_argument("--out-biased", default="manip_biased_3sensors.csv",
                        help="Output CSV for manipulation BIASED (raw − bias) data")
    args = parser.parse_args()

    reader = ThreeSensorReader(
        cal1_path=args.ni_cal1,
        cal2_path=args.ni_cal2,
        lj_sensor=args.lj_sensor,
        hz=args.hz
    )

    try:
        # 1) Compute bias FIRST (sensors at rest, no load)
        print(f"[Bias] Averaging for {args.bias_seconds}s @ {args.hz}Hz (keep sensors unloaded).")
        reader.compute_bias(seconds=args.bias_seconds)

        # 2) Manipulation phase RAW (move sensors now)
        # print(f"[Manipulation] Recording RAW for {args.manip_seconds}s @ {args.hz}Hz (move sensors).")
        # stream_to_csv(reader.read_raw, args.hz, args.manip_seconds, args.out_raw)

        # 3) Manipulation phase BIASED (raw − bias) — another pass to get zeroed data
        print(f"[Manipulation] Recording BIASED for {args.manip_seconds}s @ {args.hz}Hz.")
        stream_to_csv(reader.read_biased, args.hz, args.manip_seconds, args.out_biased)

        # 4) Visualization
        # plot_csv(args.out_raw, "RAW (calibrated)")
        # plot_csv(args.out_biased, "BIASED (raw − bias)")

        print("[Done] Bias computed, manipulation data saved, and plots generated.")

    finally:
        reader.close()


if __name__ == "__main__":
    main()
