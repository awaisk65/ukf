import numpy as np

from ukf.UKF_plotter import UKFPlotter


class TestUKFPlotter:
    """Unit tests for UKFPlotter."""

    def test_plot_results_runs_without_error(self):
        # Create dummy arrays
        true_states = np.zeros((10, 6))
        estimates = np.zeros((10, 9))
        errors = np.zeros((10, 3))

        # Just check that plot_results runs without exceptions
        UKFPlotter.plot_results(true_states, estimates, errors)
