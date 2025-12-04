"""
ukf_accuracy_demo.py

Full UKF accuracy evaluation for DroneUKFModel:
- Generates synthetic ground-truth trajectory
- Produces IMU + GPS noisy measurements
- Runs the UKF on the noisy data
- Computes estimation errors
- Plots true vs estimated trajectories and error curves
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


class DroneUKFModel:
    """
    UKF model defining the state transition and measurement functions
    for IMU + GPS/VPS fusion.

    State vector (dim_x = 10):
    x = [
        px, py, pz,
        vx, vy, vz,
        qx, qy, qz, qw
    ]

    Measurements:
        IMU: ax, ay, az, gx, gy, gz  (mag optional)
        GPS/VPS: x_mercator, y_mercator, altitude
    """

    def __init__(self, dt=0.01):
        """
        Initialize UKF, sigma points, and noise matrices.
        Parameters
        ----------
        dt : float
            Time step for prediction.
        """

        self.dt = dt
        self.imu_meas = np.zeros(6)
        self.gps_meas = np.zeros(3)
        self.dim_x = 10  # State dimension

        # Sigma points
        self.points = MerweScaledSigmaPoints(
            n=self.dim_x,
            alpha=0.1,
            beta=2.0,
            kappa=1.0
        )

        # Construct the UKF
        # Create UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=6,         # overwritten on each update
            dt=0.01,
            fx=self.fx,
            hx=None,         # we provide measurement functions manually per sensor
            points=self.points
        )

        # Initial state vector
        self.ukf.x = np.zeros(self.dim_x)
        self.ukf.x[9] = 1.0  # quaternion identity

        # Initial Covariance (tune later)
        self.ukf.P = np.eye(self.dim_x) * 0.5
       
        # Process noise
        self.ukf.Q = np.eye(self.dim_x) * 0.01

    # ---------------------------------------------------------
    # 1. STATE TRANSITION MODEL
    # ---------------------------------------------------------
    def fx(self, x, dt):
        """
        Predict next state.
        Quaternion integrated using gyro (simple integration inside update step).

        Translational model:
            p_next = p + v * dt
            v_next = v   (accel added via imu update, not predict)

        Orientation:
            kept constant in predict step (updated via IMU)
        """
        px, py, pz = x[0:3]
        vx, vy, vz = x[3:6]
        qx, qy, qz, qw = x[6:10]

        # Position update
        px += vx * dt
        py += vy * dt
        pz += vz * dt

        # No velocity update (accelerometer handled in update)
        # No quaternion update in predict step

        return np.array([px, py, pz, vx, vy, vz, qx, qy, qz, qw])

   # ----------------------------------------------------------------------
    # 2. IMU MEASUREMENT MODEL
    # ----------------------------------------------------------------------
    def h_imu(self, x):
        """
        IMU measurement function:
            accel_meas = R(q)^T * (v_dot + g)
            gyro_meas  = omega (derived from quaternion rates)

        For simplicity here:
            accel_meas = R(q)^T * g
            gyro_meas  = 0   (gyro update done separately later)

        Output:
            [ax, ay, az, gx, gy, gz]
        """

        # Extract quaternion
        qx, qy, qz, qw = x[6:10]

        # Rotation matrix from quaternion
        R = self.quat_to_rot(qx, qy, qz, qw)

        g = np.array([0, 0, -9.81])  # gravity

        # IMU accel = R^T * g
        accel_body = R.T @ g

        # Gyro predicted as zero here
        gyro_body = np.array([0.0, 0.0, 0.0])

        return np.hstack((accel_body, gyro_body))

    # ----------------------------------------------------------------------
    # 3. GPS / VPS MEASUREMENT MODEL
    # ----------------------------------------------------------------------
    def h_gps(self, x):
        """
        GPS/VPS position measurement in ENU/Mercator frame.
        Output = [px, py, pz]
        """
        return x[0:3]
    
    # ----------------------------------------------------------------------
    # Quaternion to rotation matrix
    # ----------------------------------------------------------------------
    @staticmethod
    def quat_to_rot(qx, qy, qz, qw):
        """Convert quaternion (qx,qy,qz,qw) → 3x3 rotation matrix."""
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2*(qy**2 + qz**2)
        R[0, 1] = 2*(qx*qy - qz*qw)
        R[0, 2] = 2*(qx*qz + qy*qw)

        R[1, 0] = 2*(qx*qy + qz*qw)
        R[1, 1] = 1 - 2*(qx**2 + qz**2)
        R[1, 2] = 2*(qy*qz - qx*qw)

        R[2, 0] = 2*(qx*qz - qy*qw)
        R[2, 1] = 2*(qy*qz + qx*qw)
        R[2, 2] = 1 - 2*(qx**2 + qy**2)

        return R
    
    @staticmethod
    def latlon_to_webmercator(lat, lon):
        """
        Convert latitude/longitude (degrees) to Web Mercator coordinates (meters).

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.

        Returns
        -------
        tuple (x, y)
            Web Mercator coordinates in meters.
        """
        R = 6378137.0  # WGS-84 Earth radius (meters)

        x = math.radians(lon) * R
        y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * R

        return x, y

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def step(self):
        """
        Perform one UKF predict-update cycle with IMU and GPS measurements.

        Parameters
        ----------
        dt : float
            Time step.
        imu_meas : ndarray
            IMU measurement vector [ax, ay, az, gx, gy, gz].
        gps_meas : ndarray
            GPS measurement vector [px, py, pz].

        Returns
        -------
        ndarray
            Updated state estimate.
        ndarray
            Updated covariance matrix.
        """
        self.ukf.predict(dt=self.dt)

        # IMU update
        if self.imu_meas is not None:
            # Measurement noise
            self.ukf.R = np.eye(6) * 0.5
            self.ukf.update(self.imu_meas, hx=self.h_imu)

        # GPS update
        if self.gps_meas is not None:
            # Measurement noise
            self.ukf.R = np.eye(3) * 0.5  
            self.ukf.update(self.gps_meas, hx=self.h_gps)

        return self.ukf.x.copy(), self.ukf.P.copy()


class DroneUKFSimulator:
    """
    Simulates:
        - True trajectory
        - IMU measurements (noisy)
        - GPS measurements (noisy)
    """

    def __init__(self, dt=0.01, total_time=5.0):
        """
        Initialize trajectory generator.

        Parameters
        ----------
        dt : float
            Simulation step size.
        total_time : float
            Duration of simulation.
        """
        self.dt = dt
        self.total_time = total_time
        self.N = int(total_time / dt)

        self.true_states = []
        self.imu_meas = []
        self.gps_meas = []

    def generate(self):
        """
        Create circular trajectory + synthetic measurements.

        Returns
        -------
        tuple
            (true states, imu measurements, gps measurements)
        """
        r = 10.0
        omega = 0.4

        for k in range(self.N):
            t = k * self.dt

            px = r * math.cos(omega * t)
            py = r * math.sin(omega * t)
            pz = 5.0

            vx = -r * omega * math.sin(omega * t)
            vy = r * omega * math.cos(omega * t)
            vz = 0.0

            q = np.array([0, 0, 0, 1])

            state = np.array([px, py, pz, vx, vy, vz, q[0], q[1], q[2], q[3]])
            self.true_states.append(state)

            accel = np.array([0, 0, -9.81]) + np.random.normal(0, 0.15, 3)
            gyro = np.random.normal(0, 0.01, 3)
            imu = np.hstack((accel, gyro))
            self.imu_meas.append(imu)

            gps_noise = np.random.normal(0, 0.5, 3)
            gps = state[0:3] + gps_noise
            self.gps_meas.append(gps)

        return (
            np.array(self.true_states),
            np.array(self.imu_meas),
            np.array(self.gps_meas)
        )


class DroneUKFPlotter:
    """
    Plotting utilities for evaluating UKF accuracy.
    """

    @staticmethod
    def plot_results(true_states, estimates, errors):
        """
        Plot true vs estimated states and estimation errors.

        Parameters
        ----------
        true_states : ndarray
            True trajectory.
        estimates : ndarray
            UKF estimated states.
        errors : ndarray
            Estimation error per timestep.
        """
        t = np.arange(len(true_states))

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        axs[0].plot(t, true_states[:, 0], linestyle='dotted')
        axs[0].plot(t, estimates[:, 0])
        axs[0].set_title("X Position")
        axs[0].legend(["True State", "X Position Estimate"])

        axs[1].plot(t, true_states[:, 1], linestyle='dotted')
        axs[1].plot(t, estimates[:, 1])
        axs[1].set_title("Y Position")
        axs[1].legend(["True State", "Y Position Estimate"])

        axs[2].plot(t, true_states[:, 2], linestyle='dotted')
        axs[2].plot(t, estimates[:, 2])
        axs[2].set_title("Z Position")
        axs[2].legend(["True State", "Z Position Estimate"])

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t, errors[:, 0])
        plt.plot(t, errors[:, 1])
        plt.plot(t, errors[:, 2])
        plt.title("Position Estimation Errors")
        plt.legend(["Ex", "Ey", "Ez"])
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    dt = 0.01
    sim = DroneUKFSimulator(dt=dt, total_time=10.0)
    true_states, imu_meas, gps_meas = sim.generate()

    ukf = DroneUKFModel(dt=dt)

    estimates = []
    errors = []

    for k in range(len(true_states)):
        ukf.imu_meas = imu_meas[k]
        ukf.gps_meas = gps_meas[k]

        x_hat, _ = ukf.step()
        estimates.append(x_hat)

        err = true_states[k, 0:3] - x_hat[0:3]
        errors.append(err)

    estimates = np.array(estimates)
    errors = np.array(errors)

    DroneUKFPlotter.plot_results(true_states, estimates, errors)
