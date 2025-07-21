import numpy as np
from inverse_mie.solver import MieSolver
from inverse_mie.optimizer import Optimizer

solver = MieSolver()
# test single‐layer solver
wavelengths = np.array([400e-9, 600e-9, 800e-9])
Q_sca, Q_abs = solver.single_layer(50e-9, 1.5+0.1j, wavelengths)
print("single-layer Q_sca:", Q_sca)
print("single-layer Q_abs:", Q_abs)

# test core‐shell solver
core_sca, core_abs = solver.core_shell(
    radius_core=40e-9,
    radius_shell=60e-9,
    m_core=1.6+0.2j,
    m_shell=1.4+0.05j,
    wavelengths=wavelengths
)
print("core-shell Q_sca:", core_sca)
print("core-shell Q_abs:", core_abs)

# test optimizer stub
opt = Optimizer(solver)
init_profile = np.linspace(1.4, 1.6, num=5)
best_prof, best_sca, best_abs = opt.optimize_shell(
    target_peaks=[650e-9, 900e-9],
    initial_profile=init_profile,
    wavelengths=wavelengths,
)
print("optimizer best_profile:", best_prof)
print("optimizer Q_sca:", best_sca)
print("optimizer Q_abs:", best_abs)

opt = Optimizer(solver)
init_profile = np.linspace(1.4, 1.6, num=2)   # 2-zone profile matches GA’s num_genes
best_prof, best_sca, best_abs = opt.optimize_shell(
    target_peaks=[650e-9, 900e-9],
    initial_profile=init_profile,
    wavelengths=wavelengths,
)
print("optimizer best_profile:", best_prof)
print("optimizer Q_sca:", best_sca)
print("optimizer Q_abs:", best_abs)