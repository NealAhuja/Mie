import numpy as np
from inverse_mie.solver import MieSolver
from inverse_mie.optimizer import Optimizer

def main():
    import sys

    # 1) Ask the user for their medium (default to 1.0)
    try:
        n_med = float(input("Enter medium refractive index (e.g. 1.0 for air, 1.33 for water): ") or 1.0)
    except ValueError:
        print("Invalid number, using n_medium=1.0")
        n_med = 1.0

    # 2) Initialize solver with that medium index
    solver = MieSolver(n_medium=n_med)


    # Initialize the solver
    solver = MieSolver()

    # Define wavelengths to test
    wavelengths = np.array([400e-9, 600e-9, 800e-9])

    # 1) Single-layer sphere test
    Q_sca, Q_abs = solver.single_layer(50e-9, 1.5+0.1j, wavelengths)
    print("single-layer Q_sca:", Q_sca)
    print("single-layer Q_abs:", Q_abs)

    # 2) Core-shell sphere test
    core_sca, core_abs = solver.core_shell(
        radius_core=40e-9,
        radius_shell=60e-9,
        m_core=1.6+0.2j,
        m_shell=1.4+0.05j,
        wavelengths=wavelengths
    )
    print("core-shell Q_sca:", core_sca)
    print("core-shell Q_abs:", core_abs)

    # test optimizer stub (3-gene: core_n, shell_n, shell_thickness_nm)
    opt = Optimizer(solver)
    init_profile = np.array([1.5, 1.4, 20.0])  # start with a 20 nm shell
    best_prof, best_sca, best_abs, history = opt.optimize_shell(
        target_peaks=[650e-9],
        initial_profile=init_profile,
        wavelengths=wavelengths
    )
    print("optimized [n_core, n_shell, shell_thickness_nm]:", best_prof)
    print("Q_sca:", best_sca)
    print("Q_abs:", best_abs)
    print("fitness history:", history)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(history)+1), history, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best Q_sca at 650 nm")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
