import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
from utils.contact_estimator_demo import ContactEstimator
from utils.NIDAQReaderDual import NIDAQReaderDual

class MeshVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        aq_hz = 60
        nidaq_cal1_path = "calibration_files/FT44298.cal"
        nidaq_cal2_path = "calibration_files/FT45281.cal"
        self.ni_reader = NIDAQReaderDual(nidaq_cal1_path, nidaq_cal2_path, aq_rate=aq_hz)

        print("Computing bias for Force sensors...")
        self.ni_reader.compute_bias(time_period=5)

        self.contact_estimator = ContactEstimator()

        # UI layout
        self.setWindowTitle("STL and Red Sphere Real-time Rendering")
        self.central_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # Create PyVista's Qt rendering window
        self.plotter = QtInteractor(self.central_widget)
        self.layout.addWidget(self.plotter.interactor)
        # add axes
        
        plot_axis_actor = self.plotter.add_axes_at_origin(labels_off=False, line_width=5)
        plot_axis_actor.SetTotalLength(50, 50, 50)
        # --- Load STL and create sphere ---
        # Make sure there is a file named 'your_model.stl' in the current directory, or replace with a built-in model
        try:
            self.mesh = pv.read('CAD_models/80mm_manipu.stl')
        except:
            # If STL not found, use a built-in cylinder for demonstration
            self.mesh = pv.Cylinder()
            print("STL file not found, loading default model")

        # Add STL to the scene
        self.plotter.add_mesh(self.mesh, color="silver", opacity=0.7)
        self.mesh_radius = 40 # unit: mm

        # Initialize red sphere
        self.sphere = pv.Sphere(radius=1, center=(0, 0, 0))
        self.sphere_actor = self.plotter.add_mesh(self.sphere, color="red")

        # --- Real-time update settings ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_sphere_position)
        self.timer.start(30)  # Update every 30ms (approx 30 FPS)

        self.plotter.show()

    def update_sphere_position(self):
        """Callback function to update sphere position in real-time"""
        FT_arr = self.ni_reader.read_biased()[0:6]  # Read first 6 values for sensor 1
        FT_finger, theta, h = self.contact_estimator.estimate_contact(FT_arr)

        if theta == 0 and h == 0:
            return  # Skip update if no contact detected
        # Calculate new coordinates (e.g., moving along a helix)
        x = self.mesh_radius * np.cos(theta)
        y = self.mesh_radius * np.sin(theta)
        z = h
        
        # Core step: directly translate the sphere mesh
        # Either reset the sphere to the origin and then move to the new position, or directly modify coordinates
        new_sphere = pv.Sphere(radius=1, center=(x, y, z))
        
        # Update Actor's data without re-adding mesh to ensure smoothness
        self.sphere_actor.mapper.dataset.copy_from(new_sphere)
        
        # Refresh rendering
        self.plotter.render()