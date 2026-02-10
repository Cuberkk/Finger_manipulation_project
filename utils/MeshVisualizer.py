import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
from utils.contact_estimator_demo import ContactEstimator
from utils.LabjackReader import LabjackATIReader
from utils.NIDAQReaderDual import NIDAQReaderDual
from utils.MainReader import MainReader

class MeshVisualizer(QtWidgets.QMainWindow):
    def __init__(self, sensor_type="nidaq",aq_hz=60, bias_switch = True):
        super().__init__()

        nidaq_cal1_path = "calibration_files/FT44298.cal"
        nidaq_cal2_path = "calibration_files/FT45281.cal"
        labjack_cal_path = "calibration_files/FT44297.cal"
        self.sensor_type = sensor_type
        if self.sensor_type == "labjack":
            self.reader = LabjackATIReader(labjack_cal_path, aq_rate=aq_hz, bias_switch=bias_switch)
        elif self.sensor_type == "nidaq":
            self.reader = NIDAQReaderDual(nidaq_cal1_path, nidaq_cal2_path, aq_rate=aq_hz, bias_switch=bias_switch)
        elif self.sensor_type == "all":
            self.reader = MainReader(nidaq_cal1_path, nidaq_cal2_path, labjack_cal_path, aq_hz, bias_switch = bias_switch)

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

        # Initialize spheres
        if self.sensor_type =="nidaq" or self.sensor_type == "all":
            self.sphere1 = pv.Sphere(radius=1, center=(0, 0, 0))
            self.sphere2 = pv.Sphere(radius=1, center=(0, 0, 0))

            self.sphere1_actor = self.plotter.add_mesh(self.sphere1, color="red")
            self.sphere2_actor = self.plotter.add_mesh(self.sphere2, color="green")
        elif self.sensor_type == "labjack" or self.sensor_type == "all":
             self.sphere3 = pv.Sphere(radius=1, center=(0, 0, 0))

             self.sphere3_actor = self.plotter.add_mesh(self.sphere3, color="blue")
        
        # --- Real-time update settings ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_sphere_position)
        self.timer.start(30)  # Update every 30ms (approx 30 FPS)

        self.plotter.show()

    def update_sphere_position(self):
        """Callback function to update sphere position in real-time"""
        if self.sensor_type == "labjack":
            FT_arr = self.reader.read()
            FT_finger3, theta3, h3 = self.contact_estimator.estimate_contact(FT_arr)
        elif self.sensor_type == "nidaq":
            FT_arr = self.reader.read()
            FT_finger1, theta1, h1 = self.contact_estimator.estimate_contact(FT_arr[0:6])  # Sensor 1
            FT_finger2, theta2, h2 = self.contact_estimator.estimate_contact()(FT_arr[6:12])  # Sensor 2
        elif self.sensor_type == "all":
            FT_arr = self.reader.read()
            FT_finger1, theta1, h1 = self.contact_estimator.estimate_contact(FT_arr[0:6])  # Sensor 1
            FT_finger2, theta2, h2 = self.contact_estimator.estimate_contact()(FT_arr[6:12])  # Sensor 2
            FT_finger3, theta3, h3 = self.contact_estimator.estimate_contact(FT_arr[12:18])  # Sensor 3

        if self.sensor_type =="nidaq" or self.sensor_type == "all":
            if theta1 == 0 and h1 == 0:
                pass  # Skip update if no contact detected
            else:
                # Calculate new coordinates (e.g., moving along a helix)
                x = self.mesh_radius * np.cos(theta1)
                y = self.mesh_radius * np.sin(theta1)
                z = h1

                # Core step: directly translate the sphere mesh
                # Either reset the sphere to the origin and then move to the new position, or directly modify coordinates
                new_sphere = pv.Sphere(radius=1, center=(x, y, z))

                # Update Actor's data without re-adding mesh to ensure smoothness
                self.sphere1_actor.mapper.dataset.copy_from(new_sphere)
            if theta2 == 0 and h2 == 0:
                pass  # Skip update if no contact detected
            else:
                theta2 = theta2 + np.pi*2/3
                # Calculate new coordinates (e.g., moving along a helix)
                x = self.mesh_radius * np.cos(theta2)
                y = self.mesh_radius * np.sin(theta2)
                z = h2

                # Core step: directly translate the sphere mesh
                # Either reset the sphere to the origin and then move to the new position, or directly modify coordinates
                new_sphere = pv.Sphere(radius=1, center=(x, y, z))

                # Update Actor's data without re-adding mesh to ensure smoothness
                self.sphere2_actor.mapper.dataset.copy_from(new_sphere)
        elif self.sensor_type == "labjack" or self.sensor_type == "all":
            if theta3 == 0 and h3 == 0:
                pass  # Skip update if no contact detected
            else:
                theta3 = theta3 - np.pi*2/3
                # Calculate new coordinates (e.g., moving along a helix)
                x = self.mesh_radius * np.cos(theta3)
                y = self.mesh_radius * np.sin(theta3)
                z = h3

                # Core step: directly translate the sphere mesh
                # Either reset the sphere to the origin and then move to the new position, or directly modify coordinates
                new_sphere = pv.Sphere(radius=1, center=(x, y, z))

                # Update Actor's data without re-adding mesh to ensure smoothness
                self.sphere3_actor.mapper.dataset.copy_from(new_sphere)
        
        # Refresh rendering
        self.plotter.render()