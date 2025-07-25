from typing import Tuple, Sequence
import numpy as np
import pygad
from inverse_mie.solver import MieSolver


class Optimizer:
    def __init__(self, solver: MieSolver):
        self.solver = solver

    def optimize_shell(
            self,
            target_peaks: Sequence[float],
            initial_profile: np.ndarray,
            wavelengths: np.ndarray,
            bounds: dict = None,
            ga_params: dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GA‐based shell optimization.

        target_peaks : list of 1 or 2 wavelengths (in meters)
        Always maximizes scattering efficiency (Q_sca)
        bounds       : dict with keys for n_low, n_high, r_core_min, r_core_max, t1_min, t1_max, t2_min, t2_max
        ga_params    : dict with keys for num_generations, sol_per_pop, num_parents_mating, mutation_percent_genes
        """
        fitness_history = []

        def fitness_func(ga, solution, sol_idx):
            n_core = solution[0]
            n_sh1 = solution[1]
            n_sh2 = solution[2]
            r_core_nm = solution[3]
            t1_nm = solution[4]
            t2_nm = solution[5]
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
            inds = [int(np.argmin(np.abs(wavelengths - tp))) for tp in target_peaks]
            return float(np.mean([Q_sca[i] for i in inds]))

        def on_gen(ga_inst):
            fitness_history.append(ga_inst.best_solution()[1])

        # Use bounds if provided, else default
        if bounds is not None:
            n_low = bounds.get('n_low', 1.3)
            n_high = bounds.get('n_high', 1.7)
            r_core_min = bounds.get('r_core_min', 20)
            r_core_max = bounds.get('r_core_max', 80)
            t1_min = bounds.get('t1_min', 10)
            t1_max = bounds.get('t1_max', 50)
            t2_min = bounds.get('t2_min', 10)
            t2_max = bounds.get('t2_max', 50)
        else:
            n_low, n_high = 1.3, 1.7
            r_core_min, r_core_max = 20, 80
            t1_min, t1_max = 10, 50
            t2_min, t2_max = 10, 50

        gene_space = [
            {"low": n_low, "high": n_high},  # core refractive index
            {"low": n_low, "high": n_high},  # shell1 refractive index
            {"low": n_low, "high": n_high},  # shell2 refractive index
            {"low": r_core_min, "high": r_core_max},  # core radius (nm)
            {"low": t1_min, "high": t1_max},  # shell1 thickness (nm)
            {"low": t2_min, "high": t2_max},  # shell2 thickness (nm)
        ]

        # Use GA params if provided, else default
        if ga_params is not None:
            num_generations = ga_params.get('num_generations', 30)
            sol_per_pop = ga_params.get('sol_per_pop', 10)
            num_parents_mating = ga_params.get('num_parents_mating', 4)
            mutation_percent_genes = ga_params.get('mutation_percent_genes', 20)
        else:
            num_generations = 30
            sol_per_pop = 10
            num_parents_mating = 4
            mutation_percent_genes = 20

        ga = pygad.GA(
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            num_genes=initial_profile.size,
            gene_space=gene_space,
            mutation_percent_genes=mutation_percent_genes,
            on_generation=on_gen,
        )
        ga.run()

        # get the final best solution
        best_solution, _, _ = ga.best_solution()

        # unpack all parameters from the best solution
        n_core = best_solution[0]
        n_sh1 = best_solution[1]
        n_sh2 = best_solution[2]
        r_core_nm = best_solution[3]
        t1_nm = best_solution[4]
        t2_nm = best_solution[5]

        # convert to meters
        r_core = r_core_nm * 1e-9
        r_sh1 = r_core + t1_nm * 1e-9
        r_sh2 = r_sh1 + t2_nm * 1e-9

        best_Q_sca, best_Q_abs = self.solver.double_shell(
            radius_core=r_core,
            radius_shell1=r_sh1,
            radius_shell2=r_sh2,
            m_core=n_core + 0j,
            m_shell1=n_sh1 + 0j,
            m_shell2=n_sh2 + 0j,
            wavelengths=wavelengths,
        )

        return best_solution, best_Q_sca, best_Q_abs, np.array(fitness_history)
