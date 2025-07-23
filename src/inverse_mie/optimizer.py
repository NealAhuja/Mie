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
        initial_profile: np.ndarray,  # [n_core, n_shell1, n_shell2, t1_nm, t2_nm]
        wavelengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GA‐based optimization for a core + 2-shell structure.
        initial_profile: [n_core, n_shell1, n_shell2, shell1_thickness_nm, shell2_thickness_nm]
        Returns (best_profile, Q_sca, Q_abs, fitness_history).
        """
        target = target_peaks[0]
        fitness_history = []

        def fitness_func(ga, solution, _):
            n_core, n_sh1, n_sh2, t1_nm, t2_nm = solution
            r_core = 40e-9
            r_sh1  = r_core + t1_nm*1e-9
            r_sh2  = r_sh1  + t2_nm*1e-9

            Q_sca, _ = self.solver.double_shell(
                radius_core=r_core,
                radius_shell1=r_sh1,
                radius_shell2=r_sh2,
                m_core=n_core+0j,
                m_shell1=n_sh1+0j,
                m_shell2=n_sh2+0j,
                wavelengths=wavelengths,
            )
            idx = np.argmin(np.abs(wavelengths - target))
            return Q_sca[idx]

        def on_generation(ga):
            fitness_history.append(ga.best_solution()[1])

        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=10,
            num_parents_mating=4,
            fitness_func=fitness_func,
            num_genes=5,
            gene_space=[
                {"low": 1.3, "high": 1.7},   # core index
                {"low": 1.3, "high": 1.7},   # shell1 index
                {"low": 1.3, "high": 1.7},   # shell2 index
                {"low":   5.0, "high": 40.0},# shell1 thickness nm
                {"low":   5.0, "high": 40.0},# shell2 thickness nm
            ],
            mutation_percent_genes=20,
            on_generation=on_generation
        )
        ga.run()

        best_sol, _, _ = ga.best_solution()
        # unpack and recompute full spectra
        n_core_opt, n_sh1_opt, n_sh2_opt, t1_opt, t2_opt = best_sol
        r_core = 40e-9
        r_sh1  = r_core + t1_opt*1e-9
        r_sh2  = r_sh1  + t2_opt*1e-9

        best_Q_sca, best_Q_abs = self.solver.double_shell(
            radius_core=r_core,
            radius_shell1=r_sh1,
            radius_shell2=r_sh2,
            m_core=n_core_opt+0j,
            m_shell1=n_sh1_opt+0j,
            m_shell2=n_sh2_opt+0j,
            wavelengths=wavelengths,
        )
        return best_sol, best_Q_sca, best_Q_abs, np.array(fitness_history)
