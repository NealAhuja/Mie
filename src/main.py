import numpy as np
from inverse_mie.solver import MieSolver
from inverse_mie.optimizer import Optimizer
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def run_optimization(peak, fixed_targets, init_profile, wavelengths, objective, n_med):
    from inverse_mie.solver import MieSolver
    from inverse_mie.optimizer import Optimizer
    solver = MieSolver(n_medium=n_med)
    opt = Optimizer(solver)
    targets = [peak] + fixed_targets
    bp, sca_bp, abs_bp, _ = opt.optimize_shell(
        target_peaks=targets,
        initial_profile=init_profile,
        wavelengths=wavelengths,
        objective=objective
    )
    return [peak*1e9, *bp]

def main():
    # === User Settings ===
    n_med = 1.00
    TARGET_PEAKS = [650e-9]  # e.g. [650e-9] for single-peak

    # --- Optimization parameter bounds (user editable) ---
    N_LOW = 1.3
    N_HIGH = 1.7
    R_CORE_MIN = 20  # nm
    R_CORE_MAX = 80  # nm
    T1_MIN = 10      # nm
    T1_MAX = 50      # nm
    T2_MIN = 10      # nm
    T2_MAX = 50      # nm

    # --- Genetic Algorithm parameters (user editable) ---
    NUM_GENERATIONS = 60
    SOL_PER_POP = 20
    NUM_PARENTS_MATING = 4
    MUTATION_PERCENT_GENES = 20

    # --- Wavelength grid for optimization (user editable) ---
    WAVELENGTH_MIN = 400e-9
    WAVELENGTH_MAX = 1000e-9
    WAVELENGTH_POINTS = 7

    print(f"Medium Index: {n_med}")
    print(f"Optimizing for {', '.join(f'{tp*1e9:.0f}nm' for tp in TARGET_PEAKS)} (Scattering)\n")

    # Initialize solver
    solver = MieSolver(n_medium=n_med)

    # Wavelength grid for optimization
    wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, WAVELENGTH_POINTS)

    # --- GA Optimization ---
    opt = Optimizer(solver)
    init_profile = np.array([1.5, 1.4, 1.5, 40.0, 20.0, 20.0])  # core n, shell n's, core radius, shell thicknesses (nm)

    bounds = {
        'n_low': N_LOW,
        'n_high': N_HIGH,
        'r_core_min': R_CORE_MIN,
        'r_core_max': R_CORE_MAX,
        't1_min': T1_MIN,
        't1_max': T1_MAX,
        't2_min': T2_MIN,
        't2_max': T2_MAX,
    }

    ga_params = {
        'num_generations': NUM_GENERATIONS,
        'sol_per_pop': SOL_PER_POP,
        'num_parents_mating': NUM_PARENTS_MATING,
        'mutation_percent_genes': MUTATION_PERCENT_GENES,
    }

    best, sca_opt, _, hist = opt.optimize_shell(
        target_peaks=TARGET_PEAKS,
        initial_profile=init_profile,
        wavelengths=wavelengths,
        bounds=bounds,
        ga_params=ga_params
    )
    print("Best profile:", best)
    # Print Q_sca at the target wavelengths
    print("Scattering efficiency at target wavelengths:")
    for tp in TARGET_PEAKS:
        idx = int(np.argmin(np.abs(wavelengths - tp)))
        print(f"  {tp*1e9:.1f} nm: Q_sca = {sca_opt[idx]:.3f}")
    print(f"Resulting scattering spectrum sample:", sca_opt[:5], "\n")

    # --- Standalone optimized spectrum with targets ---
    wavelengths_full = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, 200)
    n_core, n_sh1, n_sh2, r_core_nm, t1_nm, t2_nm = best
    r_core = r_core_nm * 1e-9
    r_sh1 = r_core + t1_nm * 1e-9
    r_sh2 = r_sh1 + t2_nm * 1e-9
    Qs_opt, _ = solver.double_shell(
        r_core, r_sh1, r_sh2,
        m_core=n_core + 0j,
        m_shell1=n_sh1 + 0j,
        m_shell2=n_sh2 + 0j,
        wavelengths=wavelengths_full
    )

    plt.figure(figsize=(5, 4))
    plt.plot(wavelengths_full * 1e9, Qs_opt, '-o', label='Optimized')
    for tp in TARGET_PEAKS:
        plt.axvline(tp * 1e9, color='gray', linestyle='--')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Scattering efficiency $Q_{sca}$")
    plt.title(f"Optimized Particle Full $Q_{{sca}}$ Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
