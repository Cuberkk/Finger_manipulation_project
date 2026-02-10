import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
from utils.MeshVisualizer import MeshVisualizer


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Type can be "nidaq", "labjack", or "all" to visualize different sensor data
    window = MeshVisualizer(sensor_type="nidaq", aq_hz=30, bias_time=5)
    window.show()
    sys.exit(app.exec_())