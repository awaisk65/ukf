from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np

# -------------------------------------------------------------
# 1. DEFINE STATE TRANSITION FUNCTION  (SYSTEM MODEL)
# -------------------------------------------------------------
def fx(x, dt):
    """
    State transition function.
    x = [position, velocity]
    """
    pos = x[0] + x[1] * dt
    vel = x[1]
    return np.array([pos, vel])


# -------------------------------------------------------------
# 2. DEFINE MEASUREMENT FUNCTION
# -------------------------------------------------------------
def hx(x):
    """Measurement function: we only measure position."""
    return np.array([x[0]])


# -------------------------------------------------------------
# 3. CREATE SIGMA POINTS
# -------------------------------------------------------------
dim_x = 2   # state: position + velocity
dim_z = 1   # measurement: position

points = MerweScaledSigmaPoints(
    n=dim_x,
    alpha=0.1,
    beta=2.0,
    kappa=1.0
)

# -------------------------------------------------------------
# 4. CONSTRUCT THE UKF
# -------------------------------------------------------------
ukf = UnscentedKalmanFilter(
    dim_x=dim_x,
    dim_z=dim_z,
    dt=1.0,
    hx=hx,
    fx=fx,
    points=points
)

# INITIAL STATE
ukf.x = np.array([0.0, 1.0])   # position=0, velocity=1

# INITIAL COVARIANCE
ukf.P = np.eye(2) * 0.5

# PROCESS NOISE (Q)
ukf.Q = np.eye(2) * 0.01

# MEASUREMENT NOISE (R)
ukf.R = np.eye(1) * 2.0


# -------------------------------------------------------------
# 5. RUN UKF: prediction + update
# -------------------------------------------------------------
measurements = [1.2, 2.1, 3.05, 4.0, 5.1]   # fake position measurements

for z in measurements:
    ukf.predict()     # propagate sigma points through fx()
    ukf.update(z)     # incorporate measurement via hx()

    print("Updated state:", ukf.x)
    print("Covariance:\n", ukf.P)
    print("--------------")

# -------------------------------------------------------------
# FINAL ESTIMATED STATE
# -------------------------------------------------------------
print("\nFinal state estimate:", ukf.x)
print("Final covariance:\n", ukf.P)
