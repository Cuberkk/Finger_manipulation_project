import time
import csv
import json
import numpy as np
import pandas as pd
from utils.NIDAQReaderDual import NIDAQReaderDual
from utils.LabjackReader import LabjackATIReader

# ─────────────────────────────────────────────────────────────────────────────
# ThreeSensorReader: combine NI(12) + LabJack(6) → (18,)
# ─────────────────────────────────────────────────────────────────────────────
class MainReader:
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

    def __init__(self, nidaq_cal1_path, nidaq_cal2_path, lj_cal_path, hz=60, bias_switch = True):
        self.hz = int(hz)
        self.ni = NIDAQReaderDual(nidaq_cal1_path, nidaq_cal2_path, self.hz)
        self.lj = LabjackATIReader(lj_cal_path, self.hz)
        self.bias_save_path = f"sensor_bias/Bias.csv"
        if bias_switch:
            print("Computing bias on initialization...")
            self.compute_bias(time_period=30)  # Automatically compute bias on init; adjust time_period as needed
        else:
            self.bias = np.zeros(18,)  # If not computing bias, set to zero

    def _read_raw(self):
        """Return one (18,) frame: [S1 Fx..Tz | S2 Fx..Tz | S3 Fx..Tz]."""
        ni12 = self.ni.read()
        lj6  = self.lj.read()
        return np.concatenate([ni12, lj6], axis=0)
    
    def compute_bias(self, time_period = 30):
        total_frames = int(self.hz * time_period)
        print(f"Computing bias over {time_period} seconds ({total_frames} frames at {self.hz} Hz)...")
        for i in range(total_frames):
            FT_arr = self._read_raw()
            mode = 'w' if i == 0 else 'a'
            with open(self.bias_save_path, mode, newline='') as csvfile:
                write = csv.writer(csvfile)
                write.writerow(FT_arr)
            time.sleep(1 / self.hz)
        
        bias = pd.read_csv(self.bias_save_path, delimiter=',', header=None).mean().values
        self.bias = np.array(bias).reshape(18,)
        bias_json = {"NIDAQ_S1": bias[0:6].tolist(),
                     "NIDAQ_S2": bias[6:12].tolist(),
                     "LabJack_S3": bias[12:18].tolist()}
        with open('sensor_bias/bias.json', 'w') as json_file:
            json.dump(bias_json, json_file, indent=4)
        return
    
    def read(self):
        """Return one (18,) frame with bias subtracted."""
        if self.bias is None:
            raise ValueError("Bias not computed yet. Call compute_bias() first.")
        raw = self._read_raw()
        biased_data = raw - self.bias
        return biased_data
    
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