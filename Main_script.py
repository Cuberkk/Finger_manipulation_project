from utils.MainReader import MainReader

def main():
    nidaq_cal1_path = "calibration_files/FT44298.cal"
    nidaq_cal2_path = "calibration_files/FT45281.cal"
    lj_cal_path = "calibration_files/FT44297.cal"
    reader = MainReader(nidaq_cal1_path, nidaq_cal2_path, lj_cal_path, hz=60, bias_switch = True)
    print("Initialization complete. Starting to read data...")
    while True:
        biased_data = reader.read()
        print(f"Thumb: Fx: {biased_data[0]}, Fy: {biased_data[1]}, Fz: {biased_data[2]}\n Index: Fx: {biased_data[6]}, Fy: {biased_data[7]}, Fz: {biased_data[8]},\n Ring: Fx: {biased_data[12]}, Fy: {biased_data[13]}, Fz: {biased_data[14]}\n")
        # Add any additional processing or saving of biased_data here
        # For example, you could save to a file, send over a network, etc.