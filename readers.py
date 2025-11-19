#!/usr/bin/env python3
"""
readers.py
----------
Device + reader layer: two NI sensors (12 channels total) + one LabJack sensor (6 channels),
combined into a single 18-element reading per sample.

Responsibility:
- Parse ATI .cal files (NI sensors) and apply 6x6 calibration matrices
- Configure NI task and LabJack T7 differential inputs
- Provide one calibrated read() per device
- Provide a small combiner class that returns a single (18,) vector

No CSVs, plotting, or CLI here—only hardware reading logic.
"""

import time
import numpy as np
import xml.etree.ElementTree as ET

# Optional imports so this file can be imported on machines without hardware drivers
try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
except Exception:
    nidaqmx = None

try:
    from labjack import ljm
except Exception:
    ljm = None


# ─────────────────────────────────────────────────────────────────────────────
# NI Dual Reader: returns 12 calibrated channels (6 per sensor)
# ─────────────────────────────────────────────────────────────────────────────
class NIDAQReaderDual:
    """
    Read two ATI sensors via NI-DAQ as a single 12-channel vector:
      Sensor1: [Fx,Fy,Fz,Tx,Ty,Tz] indices 0..5
      Sensor2: [Fx,Fy,Fz,Tx,Ty,Tz] indices 6..11
    """

    def __init__(self, cal1_path, cal2_path, aq_rate=20, phys="Dev1/ai0:7,Dev1/ai16:19"):
        # Guard: make failure explicit if NI-DAQmx isn't installed/available
        if nidaqmx is None:
            raise RuntimeError("nidaqmx not available. Install NI-DAQmx driver + Python package.")
        self.aq_rate = int(aq_rate)

        # Parse ATI .cal files (each → 6x6 matrix)
        self.cal1 = self._read_cal(cal1_path)
        self.cal2 = self._read_cal(cal2_path)

        # Build 12x12 block diagonal calibration matrix
        zero6 = np.zeros((6, 6))
        self.matrix = np.vstack((
            np.hstack((self.cal1, zero6)),
            np.hstack((zero6, self.cal2))
        ))

        # Configure an NI Task with differential channels
        self.task = nidaqmx.Task()
        phys_clean = phys.replace(", ", ",")  # NI prefers no spaces
        self.task.ai_channels.add_ai_voltage_chan(
            physical_channel=phys_clean,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0
        )
        self.task.timing.cfg_samp_clk_timing(
            rate=self.aq_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

    def _read_cal(self, path):
        """Parse an ATI .cal XML file and return a robust 6x6 matrix in Fx,Fy,Fz,Tx,Ty,Tz row order."""
        tree = ET.parse(path)
        root = tree.getroot()
        order = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        axes = {e.get('Name'): e for e in root.findall('.//UserAxis')}
        rows = []
        for n in order:
            if n not in axes:
                raise KeyError(f'UserAxis "{n}" not found in {path}')
            vals = [float(v) for v in axes[n].get('values').split()]
            if len(vals) != 6:
                raise ValueError(f'Axis {n} has {len(vals)} values (expect 6)')
            rows.append(vals)
        return np.array(rows).reshape(6, 6)

    def read(self):
        """
        Get one calibrated 12-element vector for the two NI sensors.
        Returns: np.ndarray shape (12,)
        """
        volts = np.array(self.task.read(number_of_samples_per_channel=1), dtype=float).reshape(12,)
        return self.matrix.dot(volts)

    def close(self):
        """Release NI resources."""
        try:
            self.task.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# LabJack Reader: returns 6 calibrated channels for one ATI sensor
# ─────────────────────────────────────────────────────────────────────────────
class LabjackATIReader:
    """
    Read one ATI sensor through a LabJack T7 as a 6-element vector:
      [Fx,Fy,Fz,Tx,Ty,Tz]
    """

    def __init__(self, sensor_num='44297'):
        # Guard: make failure explicit if LJM isn't installed/available
        if ljm is None:
            raise RuntimeError("labjack.ljm not available. Install LabJack LJM.")
        self.handle = ljm.openS("T7", "ANY", "ANY")
        self.matrix = self._hardcoded_cal(sensor_num)

        # Configure six differential pairs: (0-1),(2-3),(4-5),(6-7),(8-9),(10-11)
        for ch, neg in zip(range(0, 12, 2), range(1, 12, 2)):
            ljm.eWriteName(self.handle, f"AIN{ch}_NEGATIVE_CH", neg)
        self.channels = [f"AIN{i}" for i in range(0, 12, 2)]  # AIN0,2,4,6,8,10

    def _hardcoded_cal(self, num):
        """Return your provided 6x6 calibration matrix for 44297 or 44298."""
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
                [-25.34569, -0.13776, 11.73755, 20.41884,  11.56503, -21.88444],
                [-0.08462, 15.04212,  0.25775, 14.23111,  0.22923, 15.15714]
            ])

    def read(self):
        """
        Get one calibrated 6-element vector for the LabJack sensor.
        Returns: np.ndarray shape (6,)
        """
        vals = np.array(ljm.eReadNames(self.handle, len(self.channels), self.channels), dtype=float).reshape(6, 1)
        return self.matrix.dot(vals).flatten()

    def close(self):
        """Release LabJack resources."""
        try:
            ljm.close(self.handle)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ThreeSensorReader: combine NI(12) + LabJack(6) → (18,)
# ─────────────────────────────────────────────────────────────────────────────
class ThreeSensorReader:
    """
    Combines:
      - NI dual → 12 channels: Sensor1 (0..5) + Sensor2 (6..11)
      - LabJack → 6 channels:  Sensor3 (12..17)
    Provides:
      - read_raw(): np.ndarray shape (18,)
      - compute_bias(seconds, hz): set self.bias (18,) as average over time
      - read_biased(): raw − bias
    Note:
      - This class does NOT save files or plot; it's just data access.
    """

    def __init__(self, cal1_path, cal2_path, lj_sensor='44297', hz=20.0):
        self.hz = float(hz)
        self.ni = NIDAQReaderDual(cal1_path, cal2_path, aq_rate=int(self.hz))
        self.lj = LabjackATIReader(lj_sensor)
        self.bias = None  # 18-long vector to be computed once you call compute_bias

    def read_raw(self):
        """Return one (18,) frame: [S1 Fx..Tz | S2 Fx..Tz | S3 Fx..Tz]."""
        ni12 = self.ni.read()
        lj6  = self.lj.read()
        return np.concatenate([ni12, lj6], axis=0)

    def compute_bias(self, seconds=30):
        """Average calibrated raw readings for `seconds` at self.hz; store as self.bias (18,)."""
        total = int(seconds * self.hz)
        acc = np.zeros(18, dtype=float)
        next_t = time.perf_counter()
        for _ in range(total):
            acc += self.read_raw()
            next_t += 1.0 / self.hz
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        self.bias = acc / max(total, 1)

    def read_biased(self):
        """Return raw − bias (requires compute_bias() to have been called)."""
        if self.bias is None:
            raise RuntimeError("Bias not computed yet. Call compute_bias() first.")
        return self.read_raw() - self.bias

    @staticmethod
    def split_sensors(vec18):
        """Split a (18,) vector into three (6,) blocks: (S1, S2, S3)."""
        v = np.asarray(vec18).reshape(18,)
        return v[0:6], v[6:12], v[12:18]

    def close(self):
        """Close both devices cleanly."""
        try:
            self.ni.close()
        except Exception:
            pass
        try:
            self.lj.close()
        except Exception:
            pass
