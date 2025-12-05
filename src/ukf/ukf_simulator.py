import numpy as np
from trajectory_simulator import TrajectorySimulator
from UKF_plotter import UKFPlotter
from UnscentedKalmanFilter import DroneUKFModel

if __name__ == "__main__":
    dt = 0.01
    sim = TrajectorySimulator(dt=dt, total_time=10.0)
    true_states, imu_meas, gps_meas = sim.generate()

    ukf = DroneUKFModel(dt=dt)

    estimates = []
    errors = []

    for k in range(len(true_states)):
        ukf.imu_meas = imu_meas[k]
        ukf.gps_meas = gps_meas[k]

        x_hat, _ = ukf.step()
        estimates.append(x_hat)

        err = true_states[k, 0:3] - x_hat[0:3]
        errors.append(err)

    estimates = np.array(estimates)
    errors = np.array(errors)

    UKFPlotter.plot_results(true_states, estimates, errors)
