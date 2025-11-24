"""
simple_ukf.py

A minimal class-based example wrapping FilterPy's UKF for a
1D position-velocity system. This serves as the simplest step
before building more advanced UKF architectures.
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


class SimpleUKF(object):
    """
    Minimal Unscented Kalman Filter wrapper for a 1D position-velocity model.

    State:
        x = [position, velocity]

    Measurement:
        z = [position]
    """

    def __init__(self):
        """
        Initialize UKF, sigma points, and noise matrices.
        """

        self.dim_x = 2
        self.dim_z = 1

        # Sigma points
        self.points = MerweScaledSigmaPoints(
            n=self.dim_x,
            alpha=0.1,
            beta=2.0,
            kappa=1.0
        )

        # Construct the UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=1.0,
            fx=self.fx,
            hx=self.hx,
            points=self.points
        )

        # Initial state
        self.ukf.x = np.array([0.0, 1.0])  # position=0, velocity=1

        # Initial covariance
        self.ukf.P = np.eye(2) * 0.5

        # Process noise
        self.ukf.Q = np.eye(2) * 0.01

        # Measurement noise
        self.ukf.R = np.eye(1) * 2.0

    # ---------------------------------------------------------
    # Process model f(x, dt)
    # ---------------------------------------------------------
    def fx(self, x, dt):
        """
        State transition function.

        Parameters
        ----------
        x : ndarray
            Current state [pos, vel].
        dt : float
            Time step.

        Returns
        -------
        ndarray
            Predicted next state.
        """
        pos = x[0] + x[1] * dt
        vel = x[1]
        return np.array([pos, vel])

    # ---------------------------------------------------------
    # Measurement model h(x)
    # ---------------------------------------------------------
    def hx(self, x):
        """
        Measurement function. We only measure position.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        ndarray
            Measurement vector [position].
        """
        return np.array([x[0]])

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def step(self, z):
        """
        Perform one UKF predict-update cycle.

        Parameters
        ----------
        z : float
            Scalar position measurement.

        Returns
        -------
        ndarray
            Updated state estimate.
        ndarray
            Updated covariance matrix.
        """
        self.ukf.predict()
        self.ukf.update(z)
        return self.ukf.x.copy(), self.ukf.P.copy()


# =================================================================
# Example usage (minimal, matches your requested "simple first step")
# =================================================================
if __name__ == "__main__":
    ukf = SimpleUKF()

    measurements = [1.2, 2.1, 3.05, 4.0, 5.1]

    for z in measurements:
        x, P = ukf.step(z)
        print("Updated state:", x)
        print("Covariance:\n", P)
        print("--------------")

    print("Final state estimate:", ukf.ukf.x)
    print("Final covariance:\n", ukf.ukf.P)
