import numpy as np
import time
from LabjackATIReader import LabjackATIReader
from NIDAQReaderDual import NIDAQReaderDual

nidaq_reader = NIDAQReaderDual(True)
ljm_reader = LabjackATIReader('44297', [])

while True:
    ft_297 = ljm_reader.read_xyz()
    ft_dual = nidaq_reader.read()
    ft_298 = ft_dual[0:5]
    ft_281 = ft_dual[6:11]
    time.sleep(0.1)