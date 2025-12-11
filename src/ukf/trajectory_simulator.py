import math

import numpy as np
from scipy.spatial.transform import Rotation as R


class TrajectorySimulator:
    """
    Simulator for generating circular drone trajectories and synthetic sensor data.

    This class produces:
        - True states containing position, velocity, and quaternion orientation
        - IMU measurements (noisy accelerometer and gyroscope)
        - GPS position measurements (noisy)
    """

    def __init__(self, dt=0.01, total_time=5.0) -> None:
        """
        Initialize trajectory generator.

        Parameters
        ----------
        dt : float
            Simulation time step.
        total_time : float
            Total simulation duration.
        """
        self.dt = dt
        self.total_time = total_time
        self.N = int(total_time / dt)

        self.true_states = []
        self.imu_meas = []
        self.gps_meas = []

    @staticmethod
    def _compute_heading_quaternion(velocity: np.ndarray) -> np.ndarray:
        """
        Compute quaternion such that the drone heading follows the velocity direction.

        The body frame is assumed standard quadcopter:
            - x-axis: forward
            - y-axis: right
            - z-axis: down

        Parameters
        ----------
        velocity : ndarray, shape (3,)
            Linear velocity vector in world frame.

        Returns
        -------
        ndarray, shape (4,)
            Quaternion [x, y, z, w] representing drone orientation.
        """
        v = velocity.astype(float)
        speed = np.linalg.norm(v)
        if speed < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0])

        x_b = v / speed  # forward axis

        z_b = np.array([0.0, 0.0, -1.0])
        y_b = np.cross(z_b, x_b)
        y_b /= np.linalg.norm(y_b)
        z_b = np.cross(x_b, y_b)

        R_bw = np.vstack((x_b, y_b, z_b)).T
        quat = R.from_matrix(R_bw).as_quat()

        return quat

    @staticmethod
    def _gyro_from_orientation(
        q_prev: np.ndarray, q_curr: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Estimate gyro measurement from consecutive quaternions.

        Parameters
        ----------
        q_prev : ndarray, shape (4,)
            Previous quaternion.
        q_curr : ndarray, shape (4,)
            Current quaternion.
        dt : float
            Time step.

        Returns
        -------
        ndarray, shape (3,)
            Angular velocity vector in body frame.
        """
        r_prev = R.from_quat(q_prev)
        r_curr = R.from_quat(q_curr)
        r_rel = r_curr * r_prev.inv()
        rotvec = r_rel.as_rotvec()
        return rotvec / dt

    def generate(self) -> tuple:
        """
        Create circular trajectory + synthetic measurements.

        Returns
        -------
        tuple of ndarrays
            (true_states, imu_measurements, gps_measurements)

        Notes
        -----
        true_states structure:
            [px, py, pz, vx, vy, vz, qx, qy, qz, qw]

        imu_meas structure:
            [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        """
        r = 10.0
        omega = 0.4
        g = np.array([0.0, 0.0, -9.81])

        q_prev = np.array([0.0, 0.0, 0.0, 1.0])

        for k in range(self.N):
            t = k * self.dt

            px = r * math.cos(omega * t)
            py = r * math.sin(omega * t)
            pz = 5.0

            vx = -r * omega * math.sin(omega * t)
            vy = r * omega * math.cos(omega * t)
            vz = 0.0

            ax = -r * (omega**2) * math.cos(omega * t)
            ay = -r * (omega**2) * math.sin(omega * t)
            az = 0.0

            pos = np.array([px, py, pz], dtype=float)
            vel = np.array([vx, vy, vz], dtype=float)
            acc = np.array([ax, ay, az], dtype=float)

            q_curr = self._compute_heading_quaternion(vel)

            gyro_true = self._gyro_from_orientation(q_prev, q_curr, self.dt)
            q_prev = q_curr

            imu_accel = (acc - g) + np.random.normal(0.0, 0.15, 3)
            imu_gyro = gyro_true + np.random.normal(0.0, 0.01, 3)

            imu_vector = np.hstack((imu_accel, imu_gyro))
            self.imu_meas.append(imu_vector)

            gps = pos + np.random.normal(0.0, 0.5, 3)
            self.gps_meas.append(gps)

            state = np.hstack((pos, vel, q_curr))
            self.true_states.append(state)

        return (
            np.array(self.true_states),
            np.array(self.imu_meas),
            np.array(self.gps_meas),
        )
