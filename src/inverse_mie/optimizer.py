from typing import Tuple, Sequence
import numpy as np
import pygad
from .solver import MieSolver

class Optimizer:
    def __init__(self, solver: MieSolver):
        """
        solver: an instance of MieSolver to evaluate scattering/absorption
        """
        self.solver = solver

    def optimize_shell(
            self,
            target_peaks: Sequence[float],
            initial_profile: np.ndarray,
            wavelengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GA‐based shell optimization, returning:
          best_profile, best_Q_sca, best_Q_abs, fitness_history
        """
        target = target_peaks[0]
        fitness_history = []

        # 3‐arg fitness function
        def fitness_func(ga, solution, idx):
            m_core_val = solution[0]
            m_shell_val = solution[-1]
            Q_sca, _ = self.solver.core_shell(
                radius_core=40e-9,
                radius_shell=60e-9,
                m_core=m_core_val + 0j,
                m_shell=m_shell_val + 0j,
                wavelengths=wavelengths,
            )
            # fitness = Q_sca at the target λ
            i = np.argmin(np.abs(wavelengths - target))
            return Q_sca[i]

        # record best fitness each generation
        def on_generation(ga):
            best_fit = ga.best_solution()[1]
            fitness_history.append(best_fit)

        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=10,
            num_parents_mating=4,
            fitness_func=fitness_func,
            num_genes=initial_profile.size,
            gene_space=[{"low": 1.3, "high": 1.7}] * initial_profile.size,
            mutation_percent_genes=20,
            on_generation=on_generation
        )
        ga.run()

        best_solution, _, _ = ga.best_solution()
        best_Q_sca, best_Q_abs = self.solver.core_shell(
            radius_core=40e-9,
            radius_shell=60e-9,
            m_core=best_solution[0] + 0j,
            m_shell=best_solution[-1] + 0j,
            wavelengths=wavelengths,
        )

        return best_solution, best_Q_sca, best_Q_abs, np.array(fitness_history)

        """
        Run a simple genetic‐algorithm to optimize a 2‐zone (core+shell) profile
        for maximizing scattering at the first target peak wavelength.
        """

        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=10,
            num_parents_mating=4,
            fitness_func=fitness_func,
            num_genes=initial_profile.size,
            gene_space=[{"low": 1.3, "high": 1.7}] * initial_profile.size,
            mutation_percent_genes=20,
        )

        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()

        best_Q_sca, best_Q_abs = self.solver.core_shell(
            radius_core=40e-9,
            radius_shell=60e-9,
            m_core=best_solution[0] + 0j,
            m_shell=best_solution[-1] + 0j,
            wavelengths=wavelengths,
        )
        return best_solution, best_Q_sca, best_Q_abs
