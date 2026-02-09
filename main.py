import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
from utils.MeshVisualizer import MeshVisualizer


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MeshVisualizer()
    window.show()
    sys.exit(app.exec_())