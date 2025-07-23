import numpy as np
from inverse_mie.solver import MieSolver
from inverse_mie.optimizer import Optimizer

def main():
    import sys

    # Enter Refractive Index of Medium
    n_med = 1.00
    print("Medium Index:", n_med)

    # 2) Initialize solver with that medium index
    solver = MieSolver(n_medium=n_med)


    # Initialize the solver
    solver = MieSolver()

    #wavelengths = np.array([400e-9, 600e-9, 800e-9])
    #wavelengths = np.array([w1, w2, w3])
    wavelengths = np.linspace(400e-9, 800e-9, 3)

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

    # GA optimizer test (6‐gene: core & two shells + core radius)
    opt = Optimizer(solver)
    init_profile = np.array([
        1.5,  # n_core
        1.4,  # n_shell1
        1.5,  # n_shell2
        40.0,  # core radius (nm)
        20.0,  # shell1 thickness (nm)
        20.0  # shell2 thickness (nm)
    ])
    best, sca, abs_, hist = opt.optimize_shell(
        target_peaks=[650e-9, 900e-9],
        initial_profile=init_profile,
        wavelengths=wavelengths
    )
    print("Optimized for 650 nm & 900 nm:")
    print(" best profile:", best)
    print(" Q_sca:", sca)  # now length-2 array
    print(" Q_abs:", abs_)

    print("optimized [n_core, n_sh1, n_sh2, r_core_nm, t1_nm, t2_nm]:", best)
    print("Q_sca:", sca)
    print("Q_abs:", abs_)
    print("fitness history:", hist)

    # 2-shell test (core + 2 shells)
    Q2_sca, Q2_abs = solver.double_shell(
        radius_core=40e-9,
        radius_shell1=60e-9,
        radius_shell2=80e-9,
        m_core=1.6 + 0.2j,
        m_shell1=1.4 + 0.05j,
        m_shell2=1.5 + 0.1j,
        wavelengths=wavelengths
    )
    print("double-shell Q_sca:", Q2_sca)
    print("double-shell Q_abs:", Q2_abs)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(hist) + 1), hist, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best avg Q_sca (650 & 900 nm)")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- design‐rule sweep: vary first peak, fix second at 900 nm ---
    first_peaks = np.linspace(500e-9, 800e-9, 7)  # 500,550,...,800 nm
    fixed_second = 900e-9
    sweep_results = []

    for peak in first_peaks:
        bp, bq_sca, bq_abs, _ = opt.optimize_shell(
            target_peaks=[peak, fixed_second],
            initial_profile=init_profile,
            wavelengths=wavelengths
        )
        # record [first_peak_nm, core_n, shell1_n, shell2_n, r_core, t1, t2]
        sweep_results.append([
            peak * 1e9,  # in nm
            *bp  # unpack your 6-gene vector
        ])

    sweep_array = np.array(sweep_results)
    # columns: [peak_nm, n_core, n_sh1, n_sh2, r_core_nm, t1_nm, t2_nm]

    print("\nSweep results (first_peak_nm vs. optimized genes):")
    print(sweep_array)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(sweep_array[:, 0], sweep_array[:, 4], 'o-')  # core radius vs. peak
    plt.plot(sweep_array[:, 0], sweep_array[:, 5], 's-')  # shell1 thickness
    plt.plot(sweep_array[:, 0], sweep_array[:, 6], '^-')  # shell2 thickness
    plt.xlabel("Target Peak #1 (nm)")
    plt.ylabel("Optimized thickness (nm)")
    plt.legend(["core radius", "shell1 t", "shell2 t"])
    plt.title("Design Rules: thickness vs. LSPR")
    plt.show()


if __name__ == "__main__":
    main()
