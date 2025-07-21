from typing import Tuple, Sequence
import numpy as np
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stub for shell‐profile optimization.

        target_peaks   : list of desired LSPR peak wavelengths (in meters)
        initial_profile: array of initial refractive‐index values per zone
        wavelengths    : array of λ over which to compute spectra

        Returns:
          best_profile : same shape as initial_profile
          best_Q_sca   : Q_sca spectrum at best_profile
          best_Q_abs   : Q_abs spectrum at best_profile
        """
        # TODO: implement GA / surrogate here.
        # For now, just return the input profile and its spectra:
        best_profile = initial_profile.copy()
        best_Q_sca, best_Q_abs = self.solver.core_shell(
            radius_core=40e-9,       # placeholder radii
            radius_shell=60e-9,
            m_core=best_profile[0] + 0j,
            m_shell=best_profile[-1] + 0j,
            wavelengths=wavelengths,
        )
        return best_profile, best_Q_sca, best_Q_abs
