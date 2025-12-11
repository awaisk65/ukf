from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ukf.UKF_plotter import UKFPlotter

FIG_DIR = Path("tests/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist


# A fixture that uses the 'request' fixture
@pytest.fixture
def test_info(request):
    # Access the requesting test function's name
    function_name = request.function.__name__
    # Access the requesting test module's name
    module_name = request.module.__name__
    # You can also access other attributes like class, node, etc.
    return function_name, module_name


def test_plot_results_saves_figures(test_info):
    """
    Test that UKFPlotter.plot_results creates figures and saves them as PNG files.
    """
    function_name, module_name = test_info

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
