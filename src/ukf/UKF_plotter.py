import matplotlib.pyplot as plt
import numpy as np


class UKFPlotter:
    """
    Plotting utilities for evaluating UKF accuracy.
    """

    @staticmethod
    def plot_results(true_states, estimates, errors) -> None:
        """
        Plot true vs estimated states and estimation errors.

        Parameters
        ----------
        true_states : ndarray
            True trajectory.
        estimates : ndarray
            UKF estimated states.
        errors : ndarray
            Estimation error per timestep.
        """
        t = np.arange(len(true_states))

        # Position plots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        axs[0].plot(t, true_states[:, 0], linestyle="dotted")
        axs[0].plot(t, estimates[:, 0])
        axs[0].set_title("X Position")
        axs[0].legend(["True State", "X Position Estimate"])

        axs[1].plot(t, true_states[:, 1], linestyle="dotted")
        axs[1].plot(t, estimates[:, 1])
        axs[1].set_title("Y Position")
        axs[1].legend(["True State", "Y Position Estimate"])

        axs[2].plot(t, true_states[:, 2], linestyle="dotted")
        axs[2].plot(t, estimates[:, 2])
        axs[2].set_title("Z Position")
        axs[2].legend(["True State", "Z Position Estimate"])

        plt.tight_layout()
        plt.show()

        # Velocity plots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        axs[0].plot(t, true_states[:, 3], linestyle="dotted")
        axs[0].plot(t, estimates[:, 3])
        axs[0].set_title("X Velocity")
        axs[0].legend(["True State", "X Velocity Estimate"])

        axs[1].plot(t, true_states[:, 4], linestyle="dotted")
        axs[1].plot(t, estimates[:, 4])
        axs[1].set_title("Y Velocity")
        axs[1].legend(["True State", "Y Velocity Estimate"])

        axs[2].plot(t, true_states[:, 5], linestyle="dotted")
        axs[2].plot(t, estimates[:, 5])
        axs[2].set_title("Z Velocity")
        axs[2].legend(["True State", "Z Velocity Estimate"])

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t, errors[:, 0])
        plt.plot(t, errors[:, 1])
        plt.plot(t, errors[:, 2])
        plt.title("Position Estimation Errors")
        plt.legend(["Ex", "Ey", "Ez"])
        plt.grid(True)
        plt.show()
