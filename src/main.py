import numpy as np
from inverse_mie.solver import MieSolver
from inverse_mie.optimizer import Optimizer
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # 1) Medium refractive index
    n_med = 1.00
    print("Medium Index:", n_med)

    # 2) Initialize solver
    solver = MieSolver(n_medium=n_med)

    # 3) Wavelength grid (covers 400–1000 nm in 61 steps)
    wavelengths = np.linspace(400e-9, 1000e-9, 7)

    # 4) Baseline tests
    Q_sca, Q_abs = solver.single_layer(50e-9, 1.5+0.1j, wavelengths)
    print("single-layer Q_sca:", Q_sca[:3], "…")
    print("single-layer Q_abs:", Q_abs[:3], "…")

    core_sca, core_abs = solver.core_shell(
        radius_core=40e-9, radius_shell=60e-9,
        m_core=1.6+0.2j, m_shell=1.4+0.05j,
        wavelengths=wavelengths
    )
    print("core-shell Q_sca:", core_sca[:3], "…")
    print("core-shell Q_abs:", core_abs[:3], "…")

    # 5) GA optimization (6-gene) for two peaks at 650 nm & 900 nm
    opt = Optimizer(solver)
    init_profile = np.array([1.5, 1.4, 1.5, 40.0, 20.0, 20.0])
    best, sca_opt, abs_opt, hist = opt.optimize_shell(
        target_peaks=[650e-9, 900e-9],
        initial_profile=init_profile,
        wavelengths=wavelengths
    )
    print("\nOptimized for 650 nm & 900 nm:")
    print(" best profile:", best)
    print(" Q_sca:", sca_opt)
    print(" Q_abs:", abs_opt)

    # 6) Convergence plot
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(hist)+1), hist, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best avg Q_sca (650 & 900 nm)")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7) Automated sweep over first peak 500→800 nm, fix second at 900 nm
    first_peaks = np.linspace(500e-9, 800e-9, 7)
    fixed_second = 900e-9
    sweep_results = []
    for peak in first_peaks:
        bp, _, _, _ = opt.optimize_shell(
            target_peaks=[peak, fixed_second],
            initial_profile=init_profile,
            wavelengths=wavelengths
        )
        sweep_results.append([peak*1e9, *bp])

    # 8) Tabulate sweep
    cols = ["peak_nm", "n_core", "n_sh1", "n_sh2", "r_core_nm", "t1_nm", "t2_nm"]
    df = pd.DataFrame(sweep_results, columns=cols)
    print("\nSweep results:")
    print(df.to_string(index=False))

    # 9) Design‐rule plot (thickness vs. peak)
    plt.figure()
    plt.plot(df["peak_nm"], df["r_core_nm"], "o-", label="core radius")
    plt.plot(df["peak_nm"], df["t1_nm"], "s-", label="shell1 thickness")
    plt.plot(df["peak_nm"], df["t2_nm"], "^-", label="shell2 thickness")
    plt.xlabel("Target Peak #1 (nm)")
    plt.ylabel("Optimized size (nm)")
    plt.legend()
    plt.title("Design Rules: layer thickness vs. LSPR")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
