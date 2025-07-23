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

    # 3) Quick wavelength grid for testing (7 points)
    wavelengths = np.linspace(400e-9, 1000e-9, 7)

    # Baseline checks
    Q_sca_single, Q_abs_single = solver.single_layer(50e-9, 1.5 + 0.1j, wavelengths)
    print("Single-layer Q_sca sample:", Q_sca_single[:5])

    core_sca, core_abs = solver.core_shell(
        radius_core=40e-9, radius_shell=60e-9,
        m_core=1.6 + 0.2j, m_shell=1.4 + 0.05j,
        wavelengths=wavelengths
    )
    print("Core-shell Q_sca sample:", core_sca[:5])

    # 4) GA optimization (6-gene) for two peaks at 650 nm & 900 nm
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

    # 5) Convergence plot
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(hist)+1), hist, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best avg Q_sca (650 & 900 nm)")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6) Automated sweep over first peak 500→800 nm, fix second at 900 nm
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

    cols = ["peak_nm", "n_core", "n_sh1", "n_sh2", "r_core_nm", "t1_nm", "t2_nm"]
    df = pd.DataFrame(sweep_results, columns=cols)
    print("\nSweep results:")
    print(df.to_string(index=False))

    # Design-rule fitting
    x = df["peak_nm"].values
    y_core = df["r_core_nm"].values
    y_t1   = df["t1_nm"].values
    y_t2   = df["t2_nm"].values
    p_core = np.polyfit(x, y_core, deg=1)
    p_t1   = np.polyfit(x, y_t1,   deg=1)
    p_t2   = np.polyfit(x, y_t2,   deg=1)
    print("\nDesign‐rule coefficients (slope, intercept):")
    print(f" Core radius vs. peak:      {p_core}")
    print(f" Shell1 thickness vs. peak: {p_t1}")
    print(f" Shell2 thickness vs. peak: {p_t2}")

    # Plot sweep design rules and linear fits
    plt.figure(figsize=(6,4))
    plt.plot(x, y_core, "o-", label="core radius")
    plt.plot(x, y_t1,   "s-", label="shell1 thickness")
    plt.plot(x, y_t2,   "^-", label="shell2 thickness")
    plt.plot(x, np.polyval(p_core, x), "-",  label=f"r_core fit: {p_core[0]:.3f}·peak + {p_core[1]:.1f}")
    plt.plot(x, np.polyval(p_t1, x),   "--", label=f"t1 fit: {p_t1[0]:.3f}·peak + {p_t1[1]:.1f}")
    plt.plot(x, np.polyval(p_t2, x),   "-.", label=f"t2 fit: {p_t2[0]:.3f}·peak + {p_t2[1]:.1f}")
    plt.xlabel("Target Peak #1 (nm)")
    plt.ylabel("Optimized size (nm)")
    plt.title("Design Rules (linear fits)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Full-spectra plotting
    wavelengths_full = np.linspace(400e-9, 1000e-9, 200)
    Q_sc1, Q_ab1 = solver.single_layer(50e-9, 1.5+0.1j, wavelengths_full)
    Q_sc2, Q_ab2 = solver.core_shell(40e-9, 60e-9, 1.6+0.2j, 1.4+0.05j, wavelengths_full)
    n_c, n1, n2, rc, t1, t2 = best
    rc_m = rc*1e-9; r1_m = rc_m + t1*1e-9; r2_m = r1_m + t2*1e-9
    Q_sc_opt, Q_ab_opt = solver.double_shell(rc_m, r1_m, r2_m, n_c+0j, n1+0j, n2+0j, wavelengths_full)
    plt.figure(figsize=(6,4))
    plt.plot(wavelengths_full*1e9, Q_sc1, label="Single-layer")
    plt.plot(wavelengths_full*1e9, Q_sc2, label="Core-shell")
    plt.plot(wavelengths_full*1e9, Q_sc_opt, label="Optimized double-shell")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("$Q_{sca}$")
    plt.title("Full Spectra Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(wavelengths_full*1e9, Q_ab1, label="Single-layer")
    plt.plot(wavelengths_full*1e9, Q_ab2, label="Core-shell")
    plt.plot(wavelengths_full*1e9, Q_ab_opt, label="Optimized double-shell")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("$Q_{abs}$")
    plt.title("Full Spectra Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
