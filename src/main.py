import numpy as np
from inverse_mie.solver import MieSolver
from inverse_mie.optimizer import Optimizer
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # === User Settings ===
    # Medium refractive index
    n_med = 1.00
    # Target wavelengths for GA optimization (one or two entries)
    TARGET_PEAKS = [ 650e-9]  # e.g. [650e-9] for single-peak
    # Objective for GA: "sca" to maximize scattering, "abs" to maximize absorption
    OBJECTIVE = "sca"

    print(f"Medium Index: {n_med}")
    print(f"Optimizing for {', '.join(f'{tp*1e9:.0f}nm' for tp in TARGET_PEAKS)} ({OBJECTIVE.upper()})\n")

    # Initialize solver
    solver = MieSolver(n_medium=n_med)

    # Quick wavelength grid for testing
    wavelengths = np.linspace(400e-9, 1000e-9, 7)

    # --- Baseline checks ---
    Q_sca_single, Q_abs_single = solver.single_layer(50e-9, 1.5+0.1j, wavelengths)
    print("Single-layer Q_sca sample:", Q_sca_single[:5])
    core_sca, core_abs = solver.core_shell(
        radius_core=40e-9, radius_shell=60e-9,
        m_core=1.6+0.2j, m_shell=1.4+0.05j,
        wavelengths=wavelengths
    )
    print("Core-shell Q_sca sample:", core_sca[:5], "\n")

    # --- GA Optimization ---
    opt = Optimizer(solver)
    init_profile = np.array([1.5, 1.4, 1.5, 40.0, 20.0, 20.0])  # core n, shell n's, core radius, shell thicknesses (nm)

    best, sca_opt, abs_opt, hist = opt.optimize_shell(
        target_peaks=TARGET_PEAKS,
        initial_profile=init_profile,
        wavelengths=wavelengths,
        objective=OBJECTIVE
    )
    print("Best profile:", best)
    result_metric = sca_opt if OBJECTIVE == "sca" else abs_opt
    print(f"Resulting {OBJECTIVE.upper()} spectrum sample:", result_metric[:5], "\n")

    # --- Convergence plot ---
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(1, len(hist)+1), hist, marker='o')
    plt.xlabel("Generation")
    plt.ylabel(f"Best avg Q_{OBJECTIVE}")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Automated Sweep ---
    fixed_targets = TARGET_PEAKS[1:]
    sweep_results = []
    for peak in np.linspace(500e-9, 800e-9, 7):
        targets = [peak] + fixed_targets
        bp, sca_bp, abs_bp, _ = opt.optimize_shell(
            target_peaks=targets,
            initial_profile=init_profile,
            wavelengths=wavelengths,
            objective=OBJECTIVE
        )
        sweep_results.append([peak*1e9, *bp])

    cols = ["peak_nm","n_core","n_sh1","n_sh2","r_core_nm","t1_nm","t2_nm"]
    df = pd.DataFrame(sweep_results, columns=cols)
    print("Sweep results:")
    print(df.to_string(index=False), "\n")

    # --- Design-Rule Fitting ---
    x = df["peak_nm"].values
    rules = {}
    for field in ["r_core_nm","t1_nm","t2_nm"]:
        poly = np.polyfit(x, df[field].values, 1)
        rules[field] = poly
        print(f"{field} vs. peak: slope={poly[0]:.4e}, intercept={poly[1]:.4f}")
    print()

    plt.figure(figsize=(5,4))
    plt.plot(x, df["r_core_nm"], "o-", label="core radius")
    plt.plot(x, df["t1_nm"],    "s-", label="shell1 thickness")
    plt.plot(x, df["t2_nm"],    "^-", label="shell2 thickness")
    plt.xlabel("Target Peak #1 (nm)")
    plt.ylabel("Optimized size (nm)")
    plt.legend()
    plt.title("Design Rules")
    plt.tight_layout()
    plt.show()

    # --- Full Spectra Comparison ---
    wavelengths_full = np.linspace(400e-9, 1000e-9, 200)
    Qs_single, Qa_single = solver.single_layer(50e-9, 1.5+0.1j, wavelengths_full)
    Qs_core,   Qa_core   = solver.core_shell(
        40e-9,60e-9,1.6+0.2j,1.4+0.05j, wavelengths_full
    )
    n_core,n_sh1,n_sh2,r_core_nm,t1_nm,t2_nm = best
    r_core = r_core_nm*1e-9
    r_sh1  = r_core    + t1_nm*1e-9
    r_sh2  = r_sh1     + t2_nm*1e-9
    Qs_opt,  Qa_opt  = solver.double_shell(
        r_core,r_sh1,r_sh2,
        m_core=n_core+0j,
        m_shell1=n_sh1+0j,
        m_shell2=n_sh2+0j,
        wavelengths=wavelengths_full
    )

    # Combined Q_sca/Q_abs plot
    plt.figure(figsize=(5,4))
    plt.plot(wavelengths_full*1e9, Qs_single, label="Single-layer")
    plt.plot(wavelengths_full*1e9, Qs_core,   label="Core-shell")
    plt.plot(wavelengths_full*1e9, Qs_opt,    label="Optimized")
    plt.xlabel("Wavelength (nm)")
    ylabel = f"{'Scattering' if OBJECTIVE=='sca' else 'Absorption'} Q_{OBJECTIVE}"
    plt.ylabel(ylabel)
    plt.title("Full Spectra Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Standalone optimized spectrum with targets ---
    plt.figure(figsize=(5,4))
    if OBJECTIVE == 'sca':
        metric = Qs_opt
        ylabel2 = "Scattering efficiency $Q_{sca}$"
    else:
        metric = Qa_opt
        ylabel2 = "Absorption efficiency $Q_{abs}$"
    plt.plot(wavelengths_full*1e9, metric, '-o', label='Optimized')
    for tp in TARGET_PEAKS:
        plt.axvline(tp*1e9, color='gray', linestyle='--')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel2)
    plt.title(f"Optimized Particle Full $Q_{{{OBJECTIVE}}}$ Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
