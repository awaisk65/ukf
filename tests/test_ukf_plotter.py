import matplotlib.pyplot as plt
import numpy as np

from ukf.UKF_plotter import UKFPlotter


def test_plot_results_saves_figures(tmp_path):
    """
    Test that UKFPlotter.plot_results creates the figures and saves them as files.
    """
    # Dummy data
    true_states = np.zeros((10, 6))
    estimates = np.zeros((10, 9))
    errors = np.zeros((10, 3))

    figures = []

    # Patch plt.show to capture the figure objects and save them
    original_show = plt.show

    def fake_show():
        fig = plt.gcf()  # get the current figure
        figures.append(fig)

        # Save figure as PNG in tmp_path
        fig_index = len(figures)
        fig.savefig(tmp_path / f"fig_{fig_index}.png")
        plt.close(fig)  # close to free memory

    plt.show = fake_show

    # Call the plotting method (which normally calls plt.show() twice)
    UKFPlotter.plot_results(true_states, estimates, errors)

    # Restore plt.show
    plt.show = original_show

    # Assertions
    assert len(figures) == 2  # two figures expected
    for i, fig_file in enumerate(tmp_path.iterdir()):
        assert fig_file.exists()
        print(f"Saved figure: {fig_file}")  # optional, for local debug

    # Optional: check figure properties
    for fig in figures:
        assert fig.get_axes()  # should have at least one axes
