import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse
import time
import csv
import json

class NIDAQReaderDual(object):
    def __init__(self, calibrated_flag = False):
        self.calibration_matrix_44298 = self.ati_calibration_matrix_reader('calibration_files/FT44298.cal')
        self.calibration_matrix_44297 = self.ati_calibration_matrix_reader('calibration_files/FT44297.cal')
        self.calibration_matrix_45281 = self.ati_calibration_matrix_reader('calibration_files/FT45281.cal')

        zero_arr = np.zeros((6,6))
        self.calibration_matrix = np.vstack((np.hstack((self.calibration_matrix_44298, zero_arr)), np.hstack((zero_arr, self.calibration_matrix_45281))))

        self.bias_save_path = 'sensor_bias/dual_sensor/Bias_dual.csv'
        self.bias_read_path_s1 = 'sensor_bias/dual_sensor/ati_bias_dual_44298.json'
        self.bias_read_path_s2 = 'sensor_bias/dual_sensor/ati_bias_dual_45281.json'

        self.calibrated_flag = calibrated_flag
        if self.calibrated_flag:
            with open(self.bias_read_path_s1, 'r', encoding='utf-8') as file:
                bias_data = json.load(file)
            self.bias_s1 = np.array([bias_data['bias_x'],
                                  bias_data['bias_y'],
                                  bias_data['bias_z'],
                                  bias_data['bias_xtq'],
                                  bias_data['bias_ytq'],
                                  bias_data['bias_ztq']], dtype=float).reshape(6,)
            
            with open(self.bias_read_path_s2, 'r', encoding='utf-8') as file:
                bias_data = json.load(file)
            self.bias_s2 = np.array([bias_data['bias_x'],
                                  bias_data['bias_y'],
                                  bias_data['bias_z'],
                                  bias_data['bias_xtq'],
                                  bias_data['bias_ytq'],
                                  bias_data['bias_ztq']], dtype=float).reshape(6,)
            self.bias = np.concatenate((self.bias_s1, self.bias_s2))

        # Acquisition rate: 200 Hz
        self.aq_rate = 50
        # Open NI DAQ Device
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            physical_channel="Dev1/ai0:7, Dev1/ai16:19",      # Differential channels: 0 to 7 and 16 to 19
            name_to_assign_to_channel="DiffInput",
            terminal_config=TerminalConfiguration.DIFF,  # Differential mode
            min_val=-10.0,
            max_val=10.0)
        
        self.task.timing.cfg_samp_clk_timing(
            rate=self.aq_rate,
            sample_mode=AcquisitionType.CONTINUOUS)

    def read(self):
        # start_time = time.time()
        voltages = self.task.read(number_of_samples_per_channel=1)
        voltages = np.array(voltages).reshape(12,)
        ft_data = np.dot(self.calibration_matrix, voltages)

        if self.calibrated_flag:
            ft_data -= self.bias
        
        # Assign forces and torques
        Fx_s1, Fy_s1, Fz_s1, Tx_s1, Ty_s1, Tz_s1, Fx_s2, Fy_s2, Fz_s2, Tx_s2, Ty_s2, Tz_s2 = ft_data.flatten()

        if abs(Fx_s1) >= 24 or abs(Fy_s1) >= 24 or abs(Fz_s1) >= 24:
            print("Warning: Force of 44298 approaching Limit!")
        elif abs(Tx_s1) >= 240 or abs(Ty_s1) >= 240 or abs(Tz_s1) >= 240:
            print("Warning: Torque of 44298 approaching Limit!")

        if abs(Fx_s2) >= 24 or abs(Fy_s2) >= 24 or abs(Fz_s2) >= 24:
            print("Warning: Force of 45281 approaching Limit!")
        elif abs(Tx_s2) >= 240 or abs(Ty_s2) >= 240 or abs(Tz_s2) >= 240:
            print("Warning: Torque of 45281 approaching Limit!")
        
        # print(ft_44298)
        # print(ft_45281)
        # ft_total = np.concatenate(ft_44298, ft_45281)
        # end_time = time.time()
        # print(f'FPS: {1/(end_time - start_time):.3f}fps')

        return ft_data

    def bias_calibration(self, bias_time):
        total_frame = int(bias_time * self.aq_rate)
        print(f'Bias Calibration Start\nBias Time: {bias_time}s\nBias Frequency: {self.aq_rate}Hz')
        for i in range(total_frame):
            forces_torques = self.read()

            mode = 'w' if i == 0 else 'a'
            with open(self.bias_save_path, mode, newline='') as csvfile:
                write = csv.writer(csvfile)
                write.writerow(forces_torques.flatten())

            time.sleep(1 / (self.aq_rate * 2))
        print('Bias Calibration End')

    def ati_calibration_matrix_reader(self, cal_path):
        tree = ET.parse(cal_path)
        root = tree.getroot()
        # Expected order of axes
        order = ['Fx','Fy','Fz','Tx','Ty','Tz']
        # Collection of UserAxis elements by Name
        axes = {e.get('Name'): e for e in root.findall('.//UserAxis')}
        rows = []
        for name in order:
            e = axes.get(name)
            if e is None:
                raise KeyError(f'UserAxis "{name}" not found in {cal_path}')
            vals = [float(v) for v in e.get('values').split()]
            if len(vals) != 6:
                raise ValueError(f'Axis {name} has {len(vals)} values (expect 6)')
            rows.append(vals)
        return np.array(rows).reshape(6,6)  # shape (6,6)

def get_bias(Bias_file):
    bias = pd.read_csv(Bias_file, delimiter=',')

    col_means = bias.mean(axis=0).to_numpy()
    
    print("Sensor 1 Bias:")
    print(f" bias_x = {col_means[0]:.2f} N, bias_y = {col_means[1]:.2f} N, bias_z = {col_means[2]:.2f} N, bias_xtq = {col_means[3]:.2f} Nmm, bias_ytq = {col_means[4]:.2f} Nmm, bias_ztq = {col_means[5]:.2f} Nmm")
    bias_s1 = {'bias_x': col_means[0], 'bias_y': col_means[1], 'bias_z': col_means[2], 'bias_xtq': col_means[3], 'bias_ytq': col_means[4], 'bias_ztq': col_means[5]}

    print("Sensor 2 Bias:")
    print(f" bias_x = {col_means[6]:.2f} N, bias_y = {col_means[7]:.2f} N, bias_z = {col_means[8]:.2f} N, bias_xtq = {col_means[9]:.2f} Nmm, bias_ytq = {col_means[10]:.2f} Nmm, bias_ztq = {col_means[11]:.2f} Nmm")
    bias_s2 = {'bias_x': col_means[6], 'bias_y': col_means[7], 'bias_z': col_means[8], 'bias_xtq': col_means[9], 'bias_ytq': col_means[10], 'bias_ztq': col_means[11]}

    return bias_s1, bias_s2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="Mode of the operation:'bias' or 'read'")
    args = parser.parse_args()
    if not args.mode:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    if args.mode == 'bias':
        ati_reader = NIDAQReaderDual(False)
        bias_time = 20
        ati_reader.bias_calibration(bias_time)
        bias_s1, bias_s2 = get_bias(ati_reader.bias_save_path)
        with open(ati_reader.bias_read_path_s1, 'w') as json_file:
            json.dump(bias_s1, json_file)
        with open(ati_reader.bias_read_path_s2, 'w') as json_file:
            json.dump(bias_s2, json_file)
        print('Bias Written Successfully')
        ati_reader.task.close()
    elif args.mode == 'read':
        ati_reader = NIDAQReaderDual(True)
        try:
            while True:
                forces_torques = ati_reader.read()
                Fx1, Fy1, Fz1, Tx1, Ty1, Tz1, Fx2, Fy2, Fz2, Tx2, Ty2, Tz2 = forces_torques.flatten()
                print(f"FT44298 - Fx: {Fx1:.2f} N, Fy: {Fy1:.2f} N, Fz: {Fz1:.2f} N, Tx: {Tx1:.2f} Nmm, Ty: {Ty1:.2f} Nmm, Tz: {Tz1:.2f} Nmm")
                print(f"FT45281 - Fx: {Fx2:.2f} N, Fy: {Fy2:.2f} N, Fz: {Fz2:.2f} N, Tx: {Tx2:.2f} Nmm, Ty: {Ty2:.2f} Nmm, Tz: {Tz2:.2f} Nmm\n")
                time.sleep(1/(ati_reader.aq_rate * 2))
        except KeyboardInterrupt:
            print("Program Interrupted")
            ati_reader.task.close()
    else:
        print("Wrong mode input. Please type 'bias' or 'read'")

## Command to run in bias mode:
## python NIDAQReaderDual.py -m bias 

## Command to run in read mode:
## python NIDAQReaderDual.py -m read