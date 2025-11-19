#!/usr/bin/env python3

# ─────────────────────────────────────────────────────────────────────────────
# Three-sensor F/T pipeline (2× NI + 1× LabJack), with raw & biased logging
# - Collects 30 s @ 20 Hz of raw (calibrated) data → ft_raw_3sensors.csv
# - Computes 30 s @ 20 Hz bias (mean) across all 18 channels
# - Collects 30 s @ 20 Hz of biased data (raw − bias) → ft_biased_3sensors.csv
# Mapping: Sensor 1 = indices 0..5, Sensor 2 = 6..11, Sensor 3 = 12..17
# ─────────────────────────────────────────────────────────────────────────────

import os                 # file paths / directory creation
import time               # timestamps and sleep for pacing
import csv                # write CSV files
import json               # (not used here but handy if you later save bias JSON)
import argparse           # simple CLI
import numpy as np        # arrays / matrix math
import xml.etree.ElementTree as ET  # to parse ATI .cal XML files safely

# Try to import NI-DAQmx; fall back to None so we can raise a friendly error later.
try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
except Exception:
    nidaqmx = None

# Try to import LabJack LJM; fall back to None so we can raise a friendly error later.
try:
    from labjack import ljm
except Exception:
    ljm = None


# ─────────────────────────────────────────────────────────────────────────────
# NI Dual Reader: returns 12 calibrated channels (6 per ATI sensor)
# ─────────────────────────────────────────────────────────────────────────────
class NIDAQReaderDual:
    """NI reader returning 12 calibrated channels (6 per ATI sensor: [Fx,Fy,Fz,Tx,Ty,Tz] × 2)."""

    def __init__(self, cal1, cal2, aq_rate=20, phys="Dev1/ai0:7,Dev1/ai16:19"):
        # Ensure nidaqmx is actually available; otherwise we can't proceed.
        if nidaqmx is None:
            raise RuntimeError("nidaqmx not available. Install NI-DAQmx driver + Python package.")

        # Save the desired sample/ pacing rate for this reader (we’ll use it for timing).
        self.aq_rate = aq_rate

        # Parse ATI .cal XML files into 6×6 matrices (Fx,Fy,Fz,Tx,Ty,Tz rows).
        self.cal1 = self._read_cal(cal1)   # For NI sensor #1 (e.g., FT44298)
        self.cal2 = self._read_cal(cal2)   # For NI sensor #2 (e.g., FT45281)

        # Build a 12×12 block-diagonal calibration matrix:
        # [ cal1   0  ]
        # [  0   cal2 ]
        zero6 = np.zeros((6, 6))
        self.matrix = np.vstack((
            np.hstack((self.cal1, zero6)),
            np.hstack((zero6, self.cal2))
        ))

        # Create one NI Task for both differential channel groups.
        self.task = nidaqmx.Task()

        # NI API prefers no spaces inside the physical channel string.
        phys_clean = phys.replace(", ", ",")

        # Add differential voltage channels covering ai0..ai7 and ai16..ai19.
        self.task.ai_channels.add_ai_voltage_chan(
            physical_channel=phys_clean,
            terminal_config=TerminalConfiguration.DIFF,  # differential mode
            min_val=-10.0, max_val=10.0                  # ±10 V range
        )

        # Configure sample clock timing; we’ll pace reads at aq_rate.
        self.task.timing.cfg_samp_clk_timing(
            rate=self.aq_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

    def _read_cal(self, path):
        """Read a 6×6 ATI calibration matrix from an XML .cal file, with safety checks."""
        tree = ET.parse(path)                                  # parse the XML
        root = tree.getroot()                                  # root element
        order = ['Fx','Fy','Fz','Tx','Ty','Tz']                # expected axis order (rows)
        axes = {e.get('Name'): e for e in root.findall('.//UserAxis')}  # map Name→element

        rows = []                                              # will collect 6 rows of 6 floats
        for n in order:
            if n not in axes:                                  # ensure axis exists
                raise KeyError(f'UserAxis "{n}" not found in {path}')
            vals = [float(v) for v in axes[n].get('values').split()]  # 6 numbers per row
            if len(vals) != 6:                                 # sanity check shape
                raise ValueError(f'Axis {n} has {len(vals)} values (expect 6)')
            rows.append(vals)                                  # add row

        return np.array(rows).reshape(6, 6)                    # return 6×6 matrix

    def read(self):
        """Read one 12-sample frame, calibrate to F/T, and return shape (12,) float array."""
        # Pull one sample from each NI AI channel (returns list-of-arrays; we flatten to (12,))
        volts = np.array(self.task.read(number_of_samples_per_channel=1), dtype=float).reshape(12,)
        # Multiply 12×12 calibration matrix by the 12×1 voltage vector → calibrated forces/torques
        return self.matrix.dot(volts)

    def close(self):
        """Close NI resources safely."""
        try:
            self.task.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# LabJack Reader: returns 6 calibrated channels for one ATI sensor
# ─────────────────────────────────────────────────────────────────────────────
class LabjackATIReader:
    """LabJack reader returning 6 calibrated channels [Fx,Fy,Fz,Tx,Ty,Tz]."""

    def __init__(self, sensor_num='44297'):
        # Ensure LJM is available.
        if ljm is None:
            raise RuntimeError("labjack.ljm not available. Install LabJack LJM.")

        # Open a T7 device (ANY connection/identifier).
        self.handle = ljm.openS("T7", "ANY", "ANY")

        # Load the proper 6×6 calibration matrix for your LabJack-connected ATI sensor.
        self.matrix = self._cal(sensor_num)

        # Configure differential pairs: AIN0-1, 2-3, 4-5, 6-7, 8-9, 10-11 for 6 channels total.
        for ch, neg in zip(range(0, 12, 2), range(1, 12, 2)):
            ljm.eWriteName(self.handle, f"AIN{ch}_NEGATIVE_CH", neg)

        # We will read the 6 positive channels only (AIN0,2,4,6,8,10).
        self.channels = [f"AIN{i}" for i in range(0, 12, 2)]

    def _cal(self, num):
        """Return your hard-coded 6×6 calibration matrix for sensor 44297 or 44298."""
        if num == '44297':
            return np.array([
                [-0.01852,  0.00308,  0.09314, -3.31759, -0.12893,  3.29239],
                [-0.17355,  4.01776,  0.04887, -1.93428,  0.09601, -1.89484],
                [ 3.76588, -0.07868,  3.76932, -0.06560,  3.81053, -0.07751],
                [-0.90533, 24.69259, 21.32183, -12.20765, -20.54142, -11.25907],
                [-24.54141,  0.43180, 11.58321, 20.15249, 12.91280, -20.36288],
                [-0.46556, 15.48853, -0.45853, 14.80759, -0.61616, 14.36472]
            ])
        else:
            # 44298
            return np.array([
                [ 0.02023,  0.09133, -0.06271, -3.37843,  0.10388,  3.45523],
                [-0.05780,  3.92989, -0.00098, -1.85396, -0.08598, -2.10499],
                [ 3.84106, -0.03460,  3.82438, -0.07386,  3.83271, -0.17920],
                [-0.92585, 24.07110, 20.98591, -11.99680, -22.19816, -11.63771],
                [-25.34569, -0.13776, 11.73755, 20.41884, 11.56503, -21.88444],
                [-0.08462, 15.04212,  0.25775, 14.23111,  0.22923, 15.15714]
            ])

    def read(self):
        """Read 6 voltages from the 6 differential channels, apply 6×6 matrix, return (6,) F/T."""
        vals = np.array(ljm.eReadNames(self.handle, len(self.channels), self.channels), dtype=float).reshape(6, 1)
        return self.matrix.dot(vals).flatten()

    def close(self):
        """Close LabJack resources safely."""
        try:
            ljm.close(self.handle)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Combined Reader: NI(12) + LabJack(6) → unified (18,) vector
# ─────────────────────────────────────────────────────────────────────────────
class ThreeSensorReader:
    """
    Combines:
      - NI dual (two 6D F/T sensors) → 12 channels
      - LabJack single (one 6D F/T sensor) → 6 channels
    Total: 18 channels per sample (order = S1[0:6], S2[6:12], S3[12:18]).
    """

    def __init__(self, cal1, cal2, lj_sensor='44297', hz=20.0):
        # Store pacing rate (Hz) for bias and streaming.
        self.hz = float(hz)

        # Initialize NI dual reader (uses ATI .cal files).
        self.ni = NIDAQReaderDual(cal1, cal2, aq_rate=int(self.hz))

        # Initialize LabJack single reader (uses hard-coded 6×6 matrix).
        self.lj = LabjackATIReader(lj_sensor)

        # Placeholder for the 18-element bias vector (set after compute_bias).
        self.bias = None

    def read_raw(self):
        """Return one (18,) vector: [S1 Fx..Tz, S2 Fx..Tz, S3 Fx..Tz], calibrated but no bias subtraction."""
        ni12 = self.ni.read()      # (12,) from NI: S1 + S2
        lj6  = self.lj.read()      # (6,)  from LJ: S3
        return np.concatenate([ni12, lj6], axis=0)  # (18,)

    def _average(self, seconds=30):
        """Average `seconds` of calibrated raw data at self.hz to compute the bias (returns (18,))."""
        total = int(seconds * self.hz)                   # total samples to take
        acc = np.zeros(18, dtype=float)                  # accumulator
        next_t = time.perf_counter()                     # for phase-locked timing
        for _ in range(total):
            acc += self.read_raw()                       # accumulate one (18,) frame
            next_t += 1.0 / self.hz                      # schedule next tick
            sleep = next_t - time.perf_counter()         # compute remaining time
            if sleep > 0:
                time.sleep(sleep)                        # sleep to hold steady rate
        return acc / max(total, 1)                       # mean (avoid div-by-zero)

    def record_raw_prebias(self, out_csv="ft_raw_3sensors.csv", seconds=30):
        """Stream raw (calibrated) data for `seconds` at self.hz to CSV (timestamped)."""
        print(f"[RAW] Collecting {seconds}s @ {self.hz:.1f}Hz → {out_csv}")
        self._stream(self.read_raw, seconds, out_csv)

    def compute_bias(self, seconds=30):
        """Compute and store the 18D bias vector by averaging raw frames for `seconds`."""
        print(f"[Bias] Measuring {seconds}s @ {self.hz:.1f}Hz ...")
        self.bias = self._average(seconds)               # store bias
        # Print per-sensor bias neatly for inspection (each sensor = 6 channels).
        for i in range(3):
            b = self.bias[6*i:6*(i+1)]
            print(f"  Sensor{i+1} offset: Fx={b[0]:.3f}, Fy={b[1]:.3f}, Fz={b[2]:.3f}, "
                  f"Tx={b[3]:.3f}, Ty={b[4]:.3f}, Tz={b[5]:.3f}")

    def read_biased(self):
        """Return (18,) = raw − bias; requires compute_bias() to have run first."""
        if self.bias is None:
            raise RuntimeError("Bias not computed. Call compute_bias() first.")
        return self.read_raw() - self.bias

    def record_biased(self, out_csv="ft_biased_3sensors.csv", seconds=30):
        """Stream biased (raw − bias) data for `seconds` at self.hz to CSV (timestamped)."""
        print(f"[Biased] Collecting {seconds}s @ {self.hz:.1f}Hz → {out_csv}")
        self._stream(self.read_biased, seconds, out_csv)

    @staticmethod
    def split_sensors(vec18):
        """Return (s1,s2,s3) slices from a (18,) vector; s1=0..5, s2=6..11, s3=12..17."""
        v = np.asarray(vec18).reshape(18,)
        return v[0:6], v[6:12], v[12:18]

    def _stream(self, read_fn, seconds, out_csv):
        """Generic streamer: call `read_fn()` at self.hz for `seconds`, write timestamped CSV."""
        # Make sure output directory exists (strip filename → dir, or "." if none).
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

        # Build header: first a timestamp, then Fx..Tz for S1, S2, S3 in that order.
        header = ["timestamp_s"] + [f"{lbl}{i+1}" for i in range(3) for lbl in ["Fx","Fy","Fz","Tx","Ty","Tz"]]

        # Compute total iterations to run.
        total = int(seconds * self.hz)

        # Open the CSV and write the header.
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

            # Phase-locked loop to reduce drift: aim to land each cycle exactly 1/hz apart.
            next_t = time.perf_counter()
            for _ in range(total):
                ts = time.time()                          # UNIX timestamp for logging & sync
                data = read_fn()                          # call the provided read function
                w.writerow([ts] + data.tolist())         # write one row (timestamp + 18 values)
                next_t += 1.0 / self.hz                   # schedule next tick
                sleep = next_t - time.perf_counter()      # compute remaining time
                if sleep > 0:
                    time.sleep(sleep)                     # sleep the remaining duration

    def close(self):
        """Close both devices cleanly (safe to call multiple times)."""
        try:
            self.ni.close()
        except Exception:
            pass
        try:
            self.lj.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI wrapper: runs raw→bias→biased sequence for the durations/rates you set
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Define command-line arguments for rate, durations, file paths, and sensor IDs.
    ap = argparse.ArgumentParser(description="3-sensor F/T with raw (pre-bias) and biased 30s streams")
    ap.add_argument("--hz", type=float, default=20.0,                    help="Streaming & bias frequency (Hz)")
    ap.add_argument("--duration", type=float, default=30.0,              help="Seconds for raw and biased recordings")
    ap.add_argument("--lj-sensor", choices=["44297","44298"], default="44297", help="Which LabJack ATI sensor matrix to use")
    ap.add_argument("--ni-cal1", default="calibration_files/FT44298.cal", help="ATI .cal XML for NI sensor #1")
    ap.add_argument("--ni-cal2", default="calibration_files/FT45281.cal", help="ATI .cal XML for NI sensor #2")
    ap.add_argument("--out-raw", default="ft_raw_3sensors.csv",          help="CSV path for raw (pre-bias) data")
    ap.add_argument("--out-biased", default="ft_biased_3sensors.csv",    help="CSV path for biased data")
    args = ap.parse_args()

    # Instantiate the combined reader (opens NI + LabJack devices).
    reader = ThreeSensorReader(
        cal1=args.ni_cal1,
        cal2=args.ni_cal2,
        lj_sensor=args.lj_sensor,
        hz=args.hz
    )

    try:
        # 1) Record RAW calibrated data (no bias subtraction) for duration at hz.
        reader.record_raw_prebias(out_csv=args.out_raw, seconds=args.duration)

        # 2) Compute bias as the mean of calibrated raw for the same duration at hz.
        reader.compute_bias(seconds=args.duration)

        # 3) Record BIASED (raw − bias) data for duration at hz.
        reader.record_biased(out_csv=args.out_biased, seconds=args.duration)

    finally:
        # Always close devices (even if the user Ctrl+C’s).
        reader.close()


# Standard Python entry point.
if __name__ == "__main__":
    main()
