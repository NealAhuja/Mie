from typing import Tuple, Sequence
import numpy as np
import pygad
from .solver import MieSolver

class Optimizer:
    def __init__(self, solver: MieSolver):
        self.solver = solver

    def optimize_shell(
        self,
        target_peaks: Sequence[float],
        initial_profile: np.ndarray,
        wavelengths: np.ndarray,
        objective: str = "sca",    # new!
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs GA to maximize either scattering (Q_sca) or absorption (Q_abs)
        at the given target wavelengths (averaging over them).
        objective: "sca" or "abs"
        """

        target = target_peaks[0]
        fitness_history = []

        def fitness_func(ga, solution, idx):
            n_core, n_sh1, n_sh2, r_core_nm, t1_nm, t2_nm = solution
            # convert nm → m
            r_core = r_core_nm * 1e-9
            r_sh1 = r_core + t1_nm * 1e-9
            r_sh2 = r_sh1 + t2_nm * 1e-9

            Q_sca, Q_abs = self.solver.double_shell(
                radius_core=r_core,
                radius_shell1=r_sh1,
                radius_shell2=r_sh2,
                m_core=n_core + 0j,
                m_shell1=n_sh1 + 0j,
                m_shell2=n_sh2 + 0j,
                wavelengths=wavelengths,
            )
            # choose which metric to use
            metric = Q_sca if objective == "sca" else Q_abs
            # if two peaks:
            i0 = np.argmin(np.abs(wavelengths - target_peaks[0]))
            i1 = np.argmin(np.abs(wavelengths - target_peaks[1]))
            return 0.5 * (metric[i0] + metric[i1])

        def on_generation(ga):
            fitness_history.append(ga.best_solution()[1])

        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=10,
            num_parents_mating=4,
            fitness_func=fitness_func,
            num_genes=6,
            gene_space=[
                {"low": 1.3, "high": 1.7},  # core index
                {"low": 1.3, "high": 1.7},  # shell1 index
                {"low": 1.3, "high": 1.7},  # shell2 index
                {"low": 20.0, "high": 60.0},  # core radius in nm
                {"low": 5.0, "high": 40.0},  # shell1 thickness in nm
                {"low": 5.0, "high": 40.0},  # shell2 thickness in nm
            ],
            mutation_percent_genes=20,
            on_generation=on_generation
        )

        ga.run()

        best_sol, _, _ = ga.best_solution()
        n_core_opt, n_sh1_opt, n_sh2_opt, r_core_opt, t1_opt, t2_opt = best_sol

        # reconstruct radii and compute final spectra
        r_core = r_core_opt * 1e-9
        r_sh1 = r_core + t1_opt * 1e-9
        r_sh2 = r_sh1 + t2_opt * 1e-9

        best_Q_sca, best_Q_abs = self.solver.double_shell(
            radius_core=r_core,
            radius_shell1=r_sh1,
            radius_shell2=r_sh2,
            m_core=n_core_opt + 0j,
            m_shell1=n_sh1_opt + 0j,
            m_shell2=n_sh2_opt + 0j,
            wavelengths=wavelengths,
        )
        return best_sol, best_Q_sca, best_Q_abs, np.array(fitness_history)

