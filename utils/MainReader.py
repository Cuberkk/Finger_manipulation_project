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

    def __init__(self, nidaq_cal1_path, nidaq_cal2_path, lj_cal_path, bias_time=30, hz=60, bias_switch = True):
        self.hz = int(hz)
        self.ni = NIDAQReaderDual(cal1_path = nidaq_cal1_path, cal2_path = nidaq_cal2_path, aq_rate = self.hz, bias_time=bias_time, bias_switch = bias_switch)
        self.lj = LabjackATIReader(lj_cal_path, self.hz, bias_time=bias_time, bias_switch=bias_switch)

    def read(self):
        """Return one (18,) frame: [S1 Fx..Tz | S2 Fx..Tz | S3 Fx..Tz]."""
        
        lj6  = self.lj.read()
        ni12 = self.ni.read()
        
        return np.concatenate([ni12, lj6], axis=0)

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

def main():
    nidaq_cal1_path = "calibration_files/FT44298.cal"
    nidaq_cal2_path = "calibration_files/FT45281.cal"
    lj_cal_path = "calibration_files/FT44297.cal"
    reader = MainReader(nidaq_cal1_path, nidaq_cal2_path, lj_cal_path, bias_time=30, hz=60, bias_switch = True)
    print("Initialization complete. Starting to read data...")
    while True:
        biased_data = reader.read()
        print(f"Thumb: Fx: {biased_data[0]}, Fy: {biased_data[1]}, Fz: {biased_data[2]}\n Index: Fx: {biased_data[6]}, Fy: {biased_data[7]}, Fz: {biased_data[8]},\n Ring: Fx: {biased_data[12]}, Fy: {biased_data[13]}, Fz: {biased_data[14]}\n")
        # Add any additional processing or saving of biased_data here

if __name__ == "__main__":
    main()