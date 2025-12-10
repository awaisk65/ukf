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

    # Compare estimated vs true positions
    pos_error = np.linalg.norm(true_states[:, 0:3] - estimates[:, 0:3], axis=1)
    print("Mean position error:", np.mean(pos_error))
    print("Max position error:", np.max(pos_error))
    print("Std of position error:", np.std(pos_error))

    # Compare estimated vs true velocities
    vel_error = np.linalg.norm(true_states[:, 3:6] - estimates[:, 3:6], axis=1)
    print("Mean velocity error:", np.mean(vel_error))
    print("Max velocity error:", np.max(vel_error))
    print("Std of velocity error:", np.std(vel_error))

    # Orientation sanity check
    q_norms = np.linalg.norm(estimates[:, 6:10], axis=1)
    print("Quaternion norms: min =", np.min(q_norms), "max =", np.max(q_norms))

    UKFPlotter.plot_results(true_states, estimates, errors)
