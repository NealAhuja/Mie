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

    # --- Baseline checks ---
    # Single-layer
    Q_sca_single, Q_abs_single = solver.single_layer(50e-9, 1.5 + 0.1j, wavelengths)
    print("Single-layer Q_sca sample:", Q_sca_single[:5])

    # Core-shell
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

    # 7) Tabulate sweep
    cols = ["peak_nm", "n_core", "n_sh1", "n_sh2", "r_core_nm", "t1_nm", "t2_nm"]
    df = pd.DataFrame(sweep_results, columns=cols)
    print("\nSweep results:")
    print(df.to_string(index=False))

    # 8) Design-rule fitting
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

    # 9) Plot sweep design rules
    plt.figure()
    plt.plot(x, y_core,  "o-", label="core radius")
    plt.plot(x, y_t1,    "s-", label="shell1 thickness")
    plt.plot(x, y_t2,    "^-", label="shell2 thickness")
    plt.xlabel("Target Peak #1 (nm)")
    plt.ylabel("Optimized size (nm)")
    plt.title("Design Rules: layer thickness vs. LSPR")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 10) Full-spectra plotting
    wavelengths = np.linspace(400e-9, 1000e-9, 200)
    # Baseline spectra
    Q_sca_single, Q_abs_single = solver.single_layer(50e-9, 1.5 + 0.1j, wavelengths)
    core_sca, core_abs = solver.core_shell(
        radius_core=40e-9, radius_shell=60e-9,
        m_core=1.6+0.2j,  m_shell=1.4+0.05j,
        wavelengths=wavelengths
    )
    # Optimized double-shell spectra
    n_core, n_sh1, n_sh2, r_core_nm, t1_nm, t2_nm = best
    r_core = r_core_nm * 1e-9
    r_sh1  = r_core    + t1_nm * 1e-9
    r_sh2  = r_sh1     + t2_nm * 1e-9
    opt_sca, opt_abs = solver.double_shell(
        radius_core   = r_core,
        radius_shell1 = r_sh1,
        radius_shell2 = r_sh2,
        m_core        = n_core+0j,
        m_shell1      = n_sh1+0j,
        m_shell2      = n_sh2+0j,
        wavelengths   = wavelengths
    )

    # 11) Combined Q_sca plot
    plt.figure(figsize=(6,4))
    plt.plot(wavelengths*1e9, Q_sca_single, label="Single-layer")
    plt.plot(wavelengths*1e9, core_sca,     label="Core-shell")
    plt.plot(wavelengths*1e9, opt_sca,      label="Optimized double-shell")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering $Q_{sca}$")
    plt.title("Full Spectra Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 12) Standalone optimized Q_sca with targets
    plt.figure(figsize=(6,4))
    plt.plot(wavelengths*1e9, opt_sca, '-o', label='GA optimized')
    plt.axvline(650, color='gray', linestyle='--', label='650 nm target')
    plt.axvline(900, color='red',  linestyle='--', label='900 nm target')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency $Q_{sca}$")
    plt.title("Optimized Particle Full $Q_{sca}$ Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 13) Combined Q_abs plot
    plt.figure(figsize=(6,4))
    plt.plot(wavelengths*1e9, Q_abs_single, label="Single-layer")
    plt.plot(wavelengths*1e9, core_abs,     label="Core-shell")
    plt.plot(wavelengths*1e9, opt_abs,      label="Optimized double-shell")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption $Q_{abs}$")
    plt.title("Full Spectra Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
