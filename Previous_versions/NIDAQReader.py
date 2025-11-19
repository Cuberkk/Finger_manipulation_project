import nidaqmx
from nidaqmx.constants import TerminalConfiguration, READ_ALL_AVAILABLE,  AcquisitionType
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse
import time
import csv
import json

class NIDAQReader(object):
    def __init__(self, sensor_num, calibrated_flag = False):
        if sensor_num == '44298':
            self.calibration_matrix = self.ati_calibration_matrix_reader('calibration_files/FT44298.cal')
        elif sensor_num == '44297':
            self.calibration_matrix = self.ati_calibration_matrix_reader('calibration_files/FT44297.cal')
        elif sensor_num == '45281':
            self.calibration_matrix = self.ati_calibration_matrix_reader('calibration_files/FT45281.cal')

        self.bias_save_path = 'sensor_bias/single_sensor/Bias.csv'
        self.bias_read_path = 'sensor_bias/single_sensor/ati_bias.json'

        self.calibrated_flag = calibrated_flag
        if self.calibrated_flag:
            with open(self.bias_read_path, 'r', encoding='utf-8') as file:
                bias_data = json.load(file)
            self.bias = np.array([bias_data['bias_x'],
                                  bias_data['bias_y'],
                                  bias_data['bias_z'],
                                  bias_data['bias_xtq'],
                                  bias_data['bias_ytq'],
                                  bias_data['bias_ztq']], dtype=float).reshape(6,)
            print(f'Bias Loaded: {self.bias}')
        
        # Acquisition rate: 200 Hz
        self.aq_rate = 50
        # Open NI DAQ Device
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            physical_channel="Dev1/ai0:5",      # Differential channels: 0 to 5
            name_to_assign_to_channel="DiffInput",
            terminal_config=TerminalConfiguration.DIFF,  # Differential mode
            min_val=-10.0,
            max_val=10.0)
        
        self.task.timing.cfg_samp_clk_timing(
            rate=self.aq_rate,
            sample_mode=AcquisitionType.CONTINUOUS)
        self.pre_time = time.time()
        self.cur_time = None

    def read(self):
        voltages = self.task.read(number_of_samples_per_channel=1)
        voltages = np.array(voltages).reshape(6,)

        # Calculate forces and torques
        forces_torques = np.dot(self.calibration_matrix, voltages)
        if self.calibrated_flag:
            forces_torques -= self.bias

        # Assign forces and torques
        Fx, Fy, Fz, Tx, Ty, Tz = forces_torques.flatten()
            
        if abs(Fx) >= 24 or abs(Fy) >= 24 or abs(Fz) >= 24:
            print("Warning: Force approaching Limit!")
        elif abs(Tx) >= 240 or abs(Ty) >= 240 or abs(Tz) >= 240:
            print("Warning: Torque approaching Limit!")
        self.cur_time = time.time()
        delta_t = self.cur_time - self.pre_time
        self.pre_time = self.cur_time
        # print(f'{delta_t}s')
        # print(f'FPS: {1/delta_t:.3f}fps')

        return forces_torques

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

    forces_x = bias.iloc[:, 0].to_numpy()
    forces_y = bias.iloc[:, 1].to_numpy()
    forces_z = bias.iloc[:, 2].to_numpy()
    torques_x = bias.iloc[:, 3].to_numpy()
    torques_y = bias.iloc[:, 4].to_numpy()
    torques_z = bias.iloc[:, 5].to_numpy()
    
    bias_x = np.mean(forces_x)
    bias_y = np.mean(forces_y)
    bias_z = np.mean(forces_z)
    bias_xtq = np.mean(torques_x)
    bias_ytq = np.mean(torques_y)
    bias_ztq = np.mean(torques_z)
    
    print(f" bias_x = {bias_x:.2f} N, bias_y = {bias_y:.2f} N, bias_z = {bias_z:.2f} N, bias_xtq = {bias_xtq:.2f} Nmm, bias_ytq = {bias_ytq:.2f} Nmm, bias_ztq = {bias_ztq:.2f} Nmm")
    bias = {'bias_x': bias_x, 'bias_y': bias_y, 'bias_z': bias_z, 'bias_xtq': bias_xtq, 'bias_ytq': bias_ytq, 'bias_ztq': bias_ztq}
    return bias


if __name__ == "__main__":
    sensor_num = '45281'
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="Mode of the operation:'bias' or 'read'")
    args = parser.parse_args()
    if not args.mode:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    if args.mode == 'bias':
        ati_reader = NIDAQReader(sensor_num, False)
        bias_time = 20
        ati_reader.bias_calibration(bias_time)
        bias = get_bias(ati_reader.bias_save_path)
        with open(ati_reader.bias_read_path, 'w') as json_file:
            json.dump(bias, json_file)
        print('Bias Written Successfully')
        ati_reader.task.close()
    elif args.mode == 'read':
        ati_reader = NIDAQReader(sensor_num, True)
        try:
            while True:
                forces_torques = ati_reader.read()
                Fx, Fy, Fz, Tx, Ty, Tz = forces_torques.flatten()
                print(f"Fx: {Fx:.2f} N, Fy: {Fy:.2f} N, Fz: {Fz:.2f} N, Tx: {Tx:.2f} Nmm, Ty: {Ty:.2f} Nmm, Tz: {Tz:.2f} Nmm\n")
                time.sleep(1/(ati_reader.aq_rate * 2))
        except KeyboardInterrupt:
            print("Program Interrupted")
            ati_reader.task.close()
    else:
        print("Wrong mode input. Please type 'bias' or 'read'")