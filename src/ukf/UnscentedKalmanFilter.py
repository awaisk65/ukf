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
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


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

    def __init__(self, dt=0.01) -> None:
        """
        Initialize UKF, sigma points, and noise matrices.
        Parameters
        ----------
        dt : float
            Time step for prediction.
        """

        self.dt = dt
        self.imu_meas = None
        self.gps_meas = None
        self.dim_x = 10  # State dimension

        # Sigma points
        self.points = MerweScaledSigmaPoints(
            n=self.dim_x, alpha=0.1, beta=2.0, kappa=1.0
        )

        # Construct the UKF
        # Create UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=6,  # overwritten on each update
            dt=0.01,
            fx=self.fx,
            hx=None,  # we provide measurement functions manually per sensor
            points=self.points,
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
    def fx(self, x, dt) -> np.ndarray:
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
    def h_imu(self, x) -> np.ndarray:
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
    def h_gps(self, x) -> np.ndarray:
        """
        GPS/VPS position measurement in ENU/Mercator frame.
        Output = [px, py, pz]
        """
        return x[0:3]

    # ----------------------------------------------------------------------
    # Quaternion to rotation matrix
    # ----------------------------------------------------------------------
    @staticmethod
    def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
        """Convert quaternion (qx,qy,qz,qw) → 3x3 rotation matrix."""
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[0, 1] = 2 * (qx * qy - qz * qw)
        R[0, 2] = 2 * (qx * qz + qy * qw)

        R[1, 0] = 2 * (qx * qy + qz * qw)
        R[1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[1, 2] = 2 * (qy * qz - qx * qw)

        R[2, 0] = 2 * (qx * qz - qy * qw)
        R[2, 1] = 2 * (qy * qz + qx * qw)
        R[2, 2] = 1 - 2 * (qx**2 + qy**2)

        return R

    @staticmethod
    def latlon_to_webmercator(lat, lon) -> tuple:
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
    def step(self) -> tuple:
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
