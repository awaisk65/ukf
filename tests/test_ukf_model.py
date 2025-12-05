import numpy as np

from ukf.UnscentedKalmanFilter import DroneUKFModel


class TestDroneUKFModel:
    """Unit tests for the DroneUKFModel class."""

    def test_step_returns_correct_shapes(self):
        dt = 0.01
        ukf = DroneUKFModel(dt=dt)

        # Provide dummy measurements
        ukf.imu_meas = np.zeros(6)
        ukf.gps_meas = np.zeros(3)

        x_hat, P = ukf.step()

        # Check state vector and covariance dimensions
        assert x_hat.ndim == 1
        assert P.shape[0] == P.shape[1] == x_hat.shape[0]

    def test_multiple_steps(self):
        dt = 0.01
        ukf = DroneUKFModel(dt=dt)
        ukf.imu_meas = np.zeros(6)
        ukf.gps_meas = np.zeros(3)

        for _ in range(5):
            x_hat, P = ukf.step()
            assert x_hat.shape[0] == P.shape[0]
