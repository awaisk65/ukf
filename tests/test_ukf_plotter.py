from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ukf.UKF_plotter import UKFPlotter

FIG_DIR = Path("tests/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist


def test_plot_results_saves_figures():
    """
    Test that UKFPlotter.plot_results creates figures and saves them as PNG files.
    """
    true_states = np.zeros((10, 6))
    estimates = np.zeros((10, 9))
    errors = np.zeros((10, 3))

    figures = []

    # Patch plt.show to capture figure objects and save them
    original_show = plt.show

    def fake_show():
        fig = plt.gcf()
        figures.append(fig)

        # Save figure in the fixed directory
        fig_index = len(figures)
        fig_file = FIG_DIR / f"fig_{fig_index}.png"
        fig.savefig(fig_file)
        plt.close(fig)

    plt.show = fake_show

    # Call the plotting method
    UKFPlotter.plot_results(true_states, estimates, errors)

    # Restore plt.show
    plt.show = original_show

    # Assertions
    assert len(figures) == 2
    for i, fig_file in enumerate(FIG_DIR.glob("fig_*.png")):
        assert fig_file.exists()
        assert fig_file.suffix == ".png"
