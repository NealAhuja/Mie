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
        GA‐based optimization of:
          [core_index, shell_index, shell_thickness_nm]
        Returns best_profile, Q_sca, Q_abs, fitness_history
        """
        target = target_peaks[0]
        fitness_history = []

        def fitness_func(ga, solution, idx):
            n_core = solution[0]
            n_shell = solution[1]
            t_shell = solution[2]  # thickness in nm
            r_core = 40e-9  # fixed core radius
            r_shell = r_core + t_shell * 1e-9

            Q_sca, _ = self.solver.core_shell(
                radius_core=r_core,
                radius_shell=r_shell,
                m_core=n_core + 0j,
                m_shell=n_shell + 0j,
                wavelengths=wavelengths,
            )
            i = np.argmin(np.abs(wavelengths - target))
            return Q_sca[i]

        def on_generation(ga):
            fitness_history.append(ga.best_solution()[1])

        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=10,
            num_parents_mating=4,
            fitness_func=fitness_func,
            num_genes=3,
            gene_space=[
                {"low": 1.3, "high": 1.7},  # core index
                {"low": 1.3, "high": 1.7},  # shell index
                {"low": 5.0, "high": 80.0},  # shell thickness in nm
            ],
            mutation_percent_genes=20,
            on_generation=on_generation
        )
        ga.run()

        best_sol, _, _ = ga.best_solution()
        # unpack and compute full spectra
        n_core_opt, n_shell_opt, t_shell_opt = best_sol
        r_core = 40e-9
        r_shell = r_core + t_shell_opt * 1e-9
        best_Q_sca, best_Q_abs = self.solver.core_shell(
            radius_core=r_core,
            radius_shell=r_shell,
            m_core=n_core_opt + 0j,
            m_shell=n_shell_opt + 0j,
            wavelengths=wavelengths,
        )

        return best_sol, best_Q_sca, best_Q_abs, np.array(fitness_history)
