import numpy as np
import xml.etree.ElementTree as ET

try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
except Exception:
    nidaqmx = None

class NIDAQReaderDual:
    """
    Read two ATI sensors via NI-DAQ as a single 12-channel vector:
      Sensor1: [Fx,Fy,Fz,Tx,Ty,Tz] indices 0..5
      Sensor2: [Fx,Fy,Fz,Tx,Ty,Tz] indices 6..11
    """

    def __init__(self, cal1_path, cal2_path, aq_rate=60, phys="Dev1/ai0:7,Dev1/ai16:19"):
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
            min_val=-10.0, max_val=10.0 # Min and max stands for the voltage range expected from the sensors.
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
        samples_available = self.task.in_stream.avail_samp_per_chan
        if samples_available > 0:
            # Discard old samples to minimize latency; keep only the most recent
            raw_data = self.task.read(number_of_samples_per_channel=samples_available)
            volts = np.array(raw_data)[:, -1].reshape(12,)
            FT_lis = self.matrix.dot(volts).flatten()
            if abs(FT_lis[0]) >= 24 or abs(FT_lis[1]) >= 24 or abs(FT_lis[2]) >= 24:
                print("Warning: Sensor 1 Force approaching Limit!")
            elif abs(FT_lis[3]) >= 240 or abs(FT_lis[4]) >= 240 or abs(FT_lis[5]) >= 240:
                print("Warning: Sensor 1 Torque approaching Limit!")

            if abs(FT_lis[6]) >= 45 or abs(FT_lis[7]) >= 45 or abs(FT_lis[8]) >= 265:
                print("Warning: Sensor 2 Force approaching Limit!")
            elif abs(FT_lis[9]) >= 450 or abs(FT_lis[10]) >= 450 or abs(FT_lis[11]) >= 450:
                print("Warning: Sensor 2 Torque approaching Limit!")
        return FT_lis

    def close(self):
        """Release NI resources."""
        try:
            self.task.close()
        except Exception:
            pass
