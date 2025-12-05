import math

import numpy as np


class TrajectorySimulator:
    """
    Simulates:
        - True trajectory
        - IMU measurements (noisy)
        - GPS measurements (noisy)
    """

    def __init__(self, dt=0.01, total_time=5.0) -> None:
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

    def generate(self) -> tuple:
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
            np.array(self.gps_meas),
        )
