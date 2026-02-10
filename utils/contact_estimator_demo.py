import time
import numpy as np
import math
from utils.NIDAQReaderDual import NIDAQReaderDual

class ContactEstimator:
    def __init__(self, force_error=0.25):
        radian_42 = math.radians(42)
        Rz_42 = np.array([[math.cos(radian_42), math.sin(radian_42), 0],
                        [-math.sin(radian_42), math.cos(radian_42), 0],
                        [0, 0, 1]])
        zero3 = np.zeros((3,3))
        self.g_adj = np.block([[Rz_42, zero3],
                        [zero3, Rz_42]])
        self.l = 27.34744 # unit: mm
        self.height = 21.993 # unit: mm
        self.H = 21.993 /2 # unit: mm
        self.radius = 40 # unit: mm
        self.threshold = np.linalg.norm([force_error, force_error, force_error])
        return
    
    def estimate_contact(self, FT_arr):
        total_force = np.linalg.norm(FT_arr[0:3])
        if total_force < self.threshold:  # If force is too small, skip
            return np.zeros(6), 0, 0
        else:
            FT_prime = self.g_adj.dot(FT_arr)
            FT_finger = np.zeros(6)
            FT_finger[0] = FT_prime[1]  # Fx
            
            phi = math.atan(FT_prime[0]/-FT_prime[2])
            # phi = np.clip(phi, -np.pi/2, np.pi/2)
            print(f"phi: {phi/np.pi*180:.2f}")
            numerator = (FT_prime[0]* self.l + FT_prime[4])
            demominator = self.radius * np.sqrt(FT_prime[0]**2 + FT_prime[2]**2)
            sine_value = numerator / demominator
            sine_value = np.clip(sine_value, -1.0, 1.0)  # Ensure the value is within the valid range for arcsin
            # print(f"sine_value: {sine_value:.4f}")
            theta = (math.asin(sine_value) - phi)
            print(f"theta: {theta/np.pi*180:.2f}")
            FT_finger[1] = -FT_prime[0]*math.cos(theta) + FT_prime[2]*math.sin(theta)  # Fy
            FT_finger[2] = FT_prime[0]*math.sin(theta) + FT_prime[2]*math.cos(theta)  # Fz
            delta_h = (FT_prime[5] - FT_prime[1] *self.radius*math.sin(theta)) / FT_prime[0]
            print(f"delta_h: {delta_h:.2f}")
            h = self.H + delta_h
            h = np.clip(h, 0, self.height)
            print(f"h: {h:.2f}")
            print(f"FT_finger: {FT_finger}")
            print(f"FT_prime: {FT_prime}")
            print()
            return FT_finger, -theta, h

def main():
    nidaq_cal1_path = "calibration_files/FT44298.cal"
    nidaq_cal2_path = "calibration_files/FT45281.cal"
    ni_reader = NIDAQReaderDual(nidaq_cal1_path, nidaq_cal2_path, aq_rate=60)
    estimator = ContactEstimator()
    print("Computing bias for Force sensors...")
    ni_reader.compute_bias(time_period=5)  # Compute bias for 30 seconds
    itr = 0
    print("Starting to read data...")
    radian_42 = math.radians(42)
    Rz_42 = np.array([[math.cos(radian_42), math.sin(radian_42), 0],
                      [-math.sin(radian_42), math.cos(radian_42), 0],
                      [0, 0, 1]])
    zero3 = np.zeros((3,3))
    g_adj = np.block([[Rz_42, zero3],
                      [zero3, Rz_42]])
    try:
        while True:
            FT_arr = ni_reader.read_biased()
            s1 = FT_arr[0:6]
            estimator.estimate_contact(s1)
            time.sleep(1 / ni_reader.aq_rate)
    except KeyboardInterrupt:
        ni_reader.close()
    return

if __name__ == "__main__":
    main()