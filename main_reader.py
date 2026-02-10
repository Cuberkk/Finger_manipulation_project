import csv
import argparse
import time
import os
from utils.MainReader import MainReader
from utils.Contact_estimator import ContactEstimator

def main(rotation_axis, test_num, record_time):
    nidaq_cal1_path = "calibration_files/FT44298.cal"
    nidaq_cal2_path = "calibration_files/FT45281.cal"
    labjack_cal_path = "calibration_files/FT44297.cal"
    bias_time = 5
    aq_hz = 200
    bias_switch = True
    print(f"Rotation axis: {rotation_axis}, Test number: {test_num}")
    save_dir = f"data/{rotation_axis}/test_{test_num}"
    os.makedirs(save_dir, exist_ok=True)
    raw_data_filename = f"{save_dir}/raw_data.csv"
    biased_data_filename = f"{save_dir}/transformed_data.csv"
    reader = MainReader(nidaq_cal1_path, nidaq_cal2_path, labjack_cal_path, bias_time, aq_hz, bias_switch)
    contact_estimator = ContactEstimator()
    first_frame = True
    all_time = record_time * 60  # Convert minutes to seconds

    print("Initialization complete. Starting to read data...")
    time.sleep(2)  # Short delay before starting to read data
    try:
        while True:
            if first_frame:
                print("Start!")
                mode = 'w'
                first_frame = False
                start_time = time.time()
            else:
                mode = 'a'
            FT_raw = reader.read()
            FT_transformed = contact_estimator.estimate_contact_tri(FT_raw)

            with open(raw_data_filename, mode, newline='') as raw_file:
                raw_writer = csv.writer(raw_file)
                raw_writer.writerow(FT_raw)

            with open(biased_data_filename, mode, newline='') as biased_file:
                biased_writer = csv.writer(biased_file)
                biased_writer.writerow(FT_transformed)
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds", end='\r')
            if elapsed_time >= all_time:  # Stop after the specified record time
                reader.close()
                print(f"\nFinished {record_time} minutes of data collection.")
                break
            time.sleep(1 / (aq_hz*6))
    except KeyboardInterrupt:
        reader.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Reader for NI and LabJack sensors")
    parser.add_argument("-q","--rotation_axis", type=str, default="yaw", help="Rotation axis for the manipulation")
    parser.add_argument("-t", "--test_num", type=int, default=1, help="Number of tests")
    parser.add_argument("-r", "--record_time", type=float, default=0.5, help="Test time in minutes")
    args = parser.parse_args()
    main(args.rotation_axis, args.test_num, args.record_time)