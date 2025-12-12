import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ukf.trajectory_simulator import TrajectorySimulator
from ukf.UKF_plotter import UKFPlotter
from ukf.UnscentedKalmanFilter import DroneUKFModel

FIG_DIR = Path("tests/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist


class TestIntegrationSimulation:
    """Integration test for UKF workflow."""

    # A fixture that uses the 'request' fixture
    @pytest.fixture
    def test_info(self, request):
        # Access the requesting test function's name
        function_name = request.function.__name__
        # Access the requesting test module's name
        module_name = request.module.__name__
        # You can also access other attributes like class, node, etc.
        return function_name, module_name

    def test_short_simulation(self, test_info):
        function_name, module_name = test_info
        dt = 0.01
        sim = TrajectorySimulator(dt=dt, total_time=5)  # very short simulation
        true_states, imu_meas, gps_meas = sim.generate()

        ukf = DroneUKFModel(dt=dt)

        estimates = []
        errors = []

        for k in range(len(true_states)):
            ukf.imu_meas = imu_meas[k]
            ukf.gps_meas = gps_meas[k]

            x_hat, _ = ukf.step()
            estimates.append(x_hat)
            err = true_states[k, 0:6] - x_hat[0:6]
            errors.append(err)

        estimates = np.array(estimates)
        errors = np.array(errors)

        # Sanity checks
        assert estimates.shape[0] == true_states.shape[0]
        assert errors.shape == (true_states.shape[0], 6)

        figures = []

        # Patch plt.show to capture figure objects and save them
        original_show = plt.show

        def fake_show():
            fig = plt.gcf()
            figures.append(fig)

            # Save figure in the fixed directory
            fig_index = len(figures)
            fig_file = FIG_DIR / f"{module_name}_fig_{fig_index}.png"
            fig.savefig(fig_file)
            plt.close(fig)

        plt.show = fake_show

        # Call the plotting method
        UKFPlotter.plot_results(true_states, estimates, errors)

        # Restore plt.show
        plt.show = original_show

        # Assertions
        assert len(figures) == 3
        for i, fig_file in enumerate(FIG_DIR.glob(f"{module_name}_fig_*.png")):
            assert fig_file.exists()
            assert fig_file.suffix == ".png"

        # Compare estimated vs true positions
        pos_error = np.linalg.norm(true_states[:, 0:3] - estimates[:, 0:3], axis=1)

        # Compare estimated vs true velocities
        vel_error = np.linalg.norm(true_states[:, 3:6] - estimates[:, 3:6], axis=1)

        # Orientation sanity check
        q_norms = np.linalg.norm(estimates[:, 6:10], axis=1)

        # assert np.mean(pos_error) < 1.0  # meters
        # assert np.max(pos_error) < 10.0  # meters
        # assert np.std(pos_error) < 1.0  # meters
        # assert np.mean(vel_error) < 1.5  # m/s
        # assert np.max(vel_error) < 5.5  # m/s
        # assert np.std(vel_error) < 1.0  # m/s
        # assert np.all(q_norms > 0.9)  # quaternions should be normalized
        # assert np.all(q_norms < 1.1)  # quaternions should be normalized

        summary_lines = [
            "### UKF Integration Test Summary",
            f"**Mean position error:** {np.trunc((np.mean(pos_error)) * 1000) / 1000} m",
            f"**Max position error:** {np.trunc((np.max(pos_error)) * 1000) / 1000} m",
            f"**Std position error:** {np.trunc((np.std(pos_error)) * 1000) / 1000} m",
            "",
            f"**Mean velocity error:** {np.trunc((np.mean(vel_error)) * 1000) / 1000} m/s",
            f"**Max velocity error:** {np.trunc((np.max(vel_error)) * 1000) / 1000} m/s",
            f"**Std velocity error:** {np.trunc((np.std(vel_error)) * 1000) / 1000} m/s",
            "",
            f"**Quaternion norm min:** {np.trunc((np.min(q_norms)) * 1000) / 1000}",
            f"**Quaternion norm max:** {np.trunc((np.max(q_norms)) * 1000) / 1000}",
            "",
        ]

        for line in summary_lines:
            self.write_summary(line)

    def write_summary(self, text: str) -> None:
        summary_file = os.getenv("GITHUB_STEP_SUMMARY", None)
        if summary_file:
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
