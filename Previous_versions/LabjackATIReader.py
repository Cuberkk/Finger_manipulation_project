from labjack import ljm
import numpy as np
import pandas as pd
import time
import csv
import json

## Frame Rate for Labjack ATI Reader is round 47hz +- 1hz

class LabjackATIReader(object):
    def __init__(self, sensor_num, ati_data):
        if sensor_num == '44297':
            self.calibration_matrix = np.array([
                [-0.01852,  0.00308,  0.09314, -3.31759, -0.12893,  3.29239],
                [-0.17355,  4.01776,  0.04887, -1.93428,  0.09601, -1.89484],
                [ 3.76588, -0.07868,  3.76932, -0.06560,  3.81053, -0.07751],
                [-0.90533, 24.69259, 21.32183, -12.20765, -20.54142, -11.25907],
                [-24.54141,  0.43180, 11.58321, 20.15249, 12.91280, -20.36288],
                [-0.46556, 15.48853, -0.45853, 14.80759, -0.61616, 14.36472]
            ])
        elif sensor_num == '44298':
            self.calibration_matrix = np.array([
                [ 0.02023,  0.09133, -0.06271, -3.37843,  0.10388,  3.45523],
                [-0.05780,  3.92989, -0.00098, -1.85396, -0.08598, -2.10499],
                [ 3.84106, -0.03460,  3.82438, -0.07386,  3.83271, -0.17920],
                [-0.92585, 24.07110, 20.98591, -11.99680, -22.19816, -11.63771],
                [-25.34569, -0.13776, 11.73755, 20.41884, 11.56503, -21.88444],
                [-0.08462, 15.04212, 0.25775, 14.23111, 0.22923, 15.15714]
            ])

        # Open T7 device
        self.handle = ljm.openS("T7", "ANY", "ANY") 
        self.ati_data = ati_data
        self.bias_save_path = '/c/Users/tians/OneDrive/Desktop/NIDAQPython-main/sensor_bias/single_sensor/Bias.csv'
        self.bias_read_path = '/c/Users/tians/OneDrive/Desktop/NIDAQPython-main/sensor_bias/single_sensor/ati_bias.json'
        with open(self.bias_read_path, 'r', encoding='utf-8') as file:
            bias_data = json.load(file)
        self.bias = [bias_data['bias_x'], bias_data['bias_y'], bias_data['bias_z']]
        # Differential channels
        ljm.eWriteName(self.handle, "AIN0_NEGATIVE_CH", 1)  # AIN0 - AIN1 (Fx)
        ljm.eWriteName(self.handle, "AIN2_NEGATIVE_CH", 3)  # AIN2 - AIN3 (Fy)
        ljm.eWriteName(self.handle, "AIN4_NEGATIVE_CH", 5)  # AIN4 - AIN5 (Fz)
        ljm.eWriteName(self.handle, "AIN6_NEGATIVE_CH", 7)  # AIN6 - AIN7 (Tx)
        ljm.eWriteName(self.handle, "AIN8_NEGATIVE_CH", 9)  # AIN8 - AIN9 (Ty)
        ljm.eWriteName(self.handle, "AIN10_NEGATIVE_CH", 11) # AIN10 - AIN11 (Tz)

        # Read channels
        self.channels = ["AIN0", "AIN2", "AIN4", "AIN6", "AIN8", "AIN10"]

    def read_xyz(self):
        start_time = time.time()
        voltages = ljm.eReadNames(self.handle, len(self.channels), self.channels)
        voltages = np.array(voltages).reshape((6, 1))

        # Calculate forces and torques
        forces_torques = np.dot(self.calibration_matrix, voltages)

        # Assign forces and torques
        Fx, Fy, Fz, Tx, Ty, Tz = forces_torques.flatten()
            
        if abs(Fx) >= 24 or abs(Fy) >= 24 or abs(Fz) >= 24:
            print("Warning: Force approaching Limit!")
        elif abs(Tx) >= 240 or abs(Ty) >= 240 or abs(Tz) >= 240:
            print("Warning: Torque approaching Limit!")
            
        force_lis = [Fx - self.bias[0], Fy - self.bias[1], Fz - self.bias[2]]
        self.ati_data[:] = force_lis
        end_time = time.time()
        # print(f'FPS: {1/(end_time - start_time):.3f}fps')
        return forces_torques

    def bias_calibration(self, bias_time, bias_freq):
        total_frame = int(bias_time * bias_freq)
        print(f'Bias Calibration Start\nBias Time: {bias_time}s\nBias Frequency: {bias_freq}Hz')
        for i in range(total_frame):
            # Read voltages
            voltages = ljm.eReadNames(self.handle, len(self.channels), self.channels)
            voltages = np.array(voltages).reshape((6, 1))

            # Calculate forces and torques
            forces_torques = np.dot(self.calibration_matrix, voltages)

            # Assign forces and torques
            Fx, Fy, Fz, Tx, Ty, Tz = forces_torques.flatten()
            mode = 'w' if i == 0 else 'a'
            with open(self.bias_save_path, mode, newline='') as csvfile:
                write = csv.writer(csvfile)
                write.writerow([Fx, Fy, Fz, Tx, Ty, Tz])

            time.sleep(1 / bias_freq)
        print('Bias Calibration End')

def get_bias(Bias_file):
    bias = pd.read_csv(Bias_file, delimiter=',', usecols=(0, 1, 2))

    forces_x = bias.iloc[:, 0].to_numpy()
    forces_y = bias.iloc[:, 1].to_numpy()
    forces_z = bias.iloc[:, 2].to_numpy()
    
    bias_x = np.mean(forces_x)
    bias_y = np.mean(forces_y)
    bias_z = np.mean(forces_z)
    
    print(f" bias_x = {bias_x}N bias_y = {bias_y}N, bias_z = {bias_z}N")
    bias = {'bias_x': bias_x, 'bias_y': bias_y, 'bias_z': bias_z}
    return bias


if __name__ == "__main__":
    sensor_num = '44297'
    ati_data = [0, 0, 0]
    ati_reader = LabjackATIReader(sensor_num, ati_data)
    bias_time = 20
    bias_freq = 47
    ati_reader.bias_calibration(bias_time, bias_freq)
    bias = get_bias(ati_reader.bias_save_path)
    with open(ati_reader.bias_read_path, 'w') as json_file:
        json.dump(bias, json_file)
    print('Bias Written Successfully')
        
