from utils.MainReader import MainReader

def main():
    nidaq_cal1_path = "calibration_files/FT44298.cal"
    nidaq_cal2_path = "calibration_files/FT45281.cal"
    lj_cal_path = "calibration_files/FT44297.cal"
    reader = MainReader(nidaq_cal1_path, nidaq_cal2_path, lj_cal_path, hz=60, bias_switch = True)
    print("Initialization complete. Starting to read data...")
    while True:
        biased_data = reader.read()
        print(biased_data)
        # Add any additional processing or saving of biased_data here
        # For example, you could save to a file, send over a network, etc.