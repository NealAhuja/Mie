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
        objective: str = "sca",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GA‐based shell optimization.

        target_peaks : list of 1 or 2 wavelengths (in meters)
        objective    : "sca" to maximize scattering, "abs" to maximize absorption
        """
        fitness_history = []

        # This fitness function now handles 1 or more targets
        def fitness_func(ga, solution, sol_idx):
            # unpack core & shell refractive indices from the GA solution
            m_core_val  = solution[0]
            m_shell_val = solution[-1]

            # compute the full spectra
            Q_sca, Q_abs = self.solver.core_shell(
                radius_core=40e-9,
                radius_shell=60e-9,
                m_core=m_core_val + 0j,
                m_shell=m_shell_val + 0j,
                wavelengths=wavelengths,
            )

            # choose which metric to optimize
            metric = Q_sca if objective == "sca" else Q_abs

            # find the nearest indices for each target peak
            inds = [int(np.argmin(np.abs(wavelengths - tp))) for tp in target_peaks]

            # average the metric over all targets
            return float(np.mean([metric[i] for i in inds]))

        # record the best fitness each generation
        def on_gen(ga_inst):
            fitness_history.append(ga_inst.best_solution()[1])

        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=10,
            num_parents_mating=4,
            fitness_func=fitness_func,
            num_genes=initial_profile.size,
            gene_space=[{"low": 1.3, "high": 1.7}] * initial_profile.size,
            mutation_percent_genes=20,
            on_generation=on_gen,
        )
        ga.run()

        # get the final best solution
        best_solution, _, _ = ga.best_solution()
        best_Q_sca, best_Q_abs = self.solver.core_shell(
            radius_core=40e-9,
            radius_shell=60e-9,
            m_core=best_solution[0] + 0j,
            m_shell=best_solution[-1] + 0j,
            wavelengths=wavelengths,
        )

        return best_solution, best_Q_sca, best_Q_abs, np.array(fitness_history)
