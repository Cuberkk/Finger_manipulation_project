import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time

task = nidaqmx.Task()
# FT44298
task.ai_channels.add_ai_voltage_chan(
    physical_channel="Dev1/ai0:5",
    name_to_assign_to_channel="DiffInput",
    terminal_config=TerminalConfiguration.DIFF,  # Differential mode
    min_val=-10.0,
    max_val=10.0
)

aq_rate = 500

# Sample clock timing configuration
task.timing.cfg_samp_clk_timing(
    rate=aq_rate,                              # Sampling rate 500 Hz
    sample_mode=AcquisitionType.CONTINUOUS
)

print("Start continuous reading from ai0~ai6 (DIFFERENTIAL mode)...")
time.sleep(1)

data = task.read(number_of_samples_per_channel=1)
data = np.array(data).reshape(6,)
for i, v in enumerate(data):
    print(f"AI {i}: {v:.4f} V")

task.close()
