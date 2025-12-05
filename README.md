
# Unscented Kalman Filter (UKF) — Python

This repository contains a small Python implementation and simulator for an Unscented Kalman Filter (UKF) intended for experimentation and plotting of simple state-estimation problems.

**Project**: A compact UKF implementation with a simulator, trajectory generator, and plotting helpers.

**Quick Overview**
- **Purpose**: Demonstrate and experiment with an Unscented Kalman Filter on simulated trajectories.
- **Language**: Python
- **Layout**: Source under `src/ukf/` with simulator and plotting utilities.

**Install**
- **Python**: Use Python 3.8+.
- **Editable install (recommended for development)**:

```
python -m pip install -e .
```

- If the project uses extra/dev dependencies add them to your environment (there is no `requirements.txt` by default in this repo).

**Run the simulator**
- Run the main simulator script directly from the repository root:

```
python src/ukf/ukf_simulator.py
```

If you have installed the package in editable mode, you can also run modules as a package:

```
python -m ukf.ukf_simulator
```

**Repository Structure**
- **`src/ukf/UnscentedKalmanFilter.py`**: Core UKF implementation (filter predict / update steps and sigma point handling).
- **`src/ukf/ukf_simulator.py`**: Example runner that builds a simulated trajectory, adds measurement noise, runs the UKF, and prints or stores results.
- **`src/ukf/trajectory_simulator.py`**: Utilities for generating ground-truth trajectories used by the simulator.
- **`src/ukf/UKF_plotter.py`**: Plotting helpers and quick visualizations for comparing truth, measurements, and filter estimates.
- **`src/ukf/ukf_simulator.py`**: Example entrypoint and demo configurations.

**Testing**
- If you have `pytest` installed, run tests from the project root:

```
pytest
```

If tests aren't present or you get import errors, ensure the package is installed (`pip install -e .`) or add the `src/` folder to `PYTHONPATH` when running tests.

*Note: Currently developing tests. Comprehensive test suite will be included soon.*


**Contributing**
- Open an issue to discuss features/bugs.
- For code changes, fork, create a branch, and open a pull request with a clear description and tests where appropriate.

**License**
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

