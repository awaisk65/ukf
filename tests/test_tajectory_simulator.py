from src.ukf.trajectory_simulator import TrajectorySimulator


class TestTrajectorySimulator:
    """Unit tests for the TrajectorySimulator class."""

    def test_generate_shapes(self):
        dt = 0.01
        sim = TrajectorySimulator(dt=dt, total_time=1.0)  # short test
        true_states, imu_meas, gps_meas = sim.generate()

        # All arrays should have same length along time axis
        assert true_states.shape[0] == imu_meas.shape[0] == gps_meas.shape[0]

        # true_states expected to have at least 6 elements per timestep (pos+vel)
        assert true_states.shape[1] >= 6
