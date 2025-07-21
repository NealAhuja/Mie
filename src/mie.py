import numpy as np
from scipy import special

def spherical_bessel_jn(n, z):
    """
    Compute the spherical Bessel function of the first kind j_n(z).
    n : order (int)
    z : argument (float or array)
    """
    return special.spherical_jn(n, z)

def spherical_hankel1(n, z):
    """
    Compute the spherical Hankel function of the first kind h_n^(1)(z).
    h_n^(1)(z) = j_n(z) + i y_n(z)
    """
    return special.spherical_jn(n, z) + 1j * special.spherical_yn(n, z)

def psi(n, z):
    """
    Riccati–Bessel function ψₙ(z) = z · jₙ(z)
    """
    return z * spherical_bessel_jn(n, z)

def xi(n, z):
    """
    Riccati–Hankel function ξₙ(z) = z · hₙ⁽¹⁾(z)
    """
    return z * spherical_hankel1(n, z)

def deriv(f, n, z, dz=1e-6):
    """
    Numerical derivative of f(n, z) with respect to z
    using a central difference of step dz.
    """
    return (f(n, z + dz) - f(n, z - dz)) / (2 * dz)


def single_layer_mie(radius, m_rel, wavelengths, n_medium=1.0):
    # Size parameter
    x = 2 * np.pi * radius * n_medium / wavelengths

    # Multipole cutoff per λ, then pick the max
    n_max_array = np.round(x + 4 * x**(1/3) + 2).astype(int)
    n_max = int(n_max_array.max())

    # Prepare storage
    a = np.zeros((n_max, len(wavelengths)), dtype=complex)
    b = np.zeros_like(a)

    # Compute a_n and b_n
    for i, lam in enumerate(wavelengths):
        xi_val = x[i]
        mxi    = m_rel * xi_val

        for n in range(1, n_max+1):
            psi_x    = psi(n, xi_val)
            psi_mx   = psi(n, mxi)
            xi_x     = xi(n, xi_val)

            psi_x_p  = deriv(psi, n, xi_val)
            psi_mx_p = deriv(psi, n, mxi)
            xi_x_p   = deriv(xi, n, xi_val)

            num_a = m_rel*psi_mx*psi_x_p - psi_x*psi_mx_p
            den_a = m_rel*psi_mx*xi_x_p - xi_x*psi_mx_p
            a[n-1, i] = num_a / den_a

            num_b = psi_mx*psi_x_p - m_rel*psi_x*psi_mx_p
            den_b = psi_mx*xi_x_p - m_rel*xi_x*psi_mx_p
            b[n-1, i] = num_b / den_b

    # Sum efficiencies
    orders = np.arange(1, n_max+1)[:, None]
    fact   = (2*orders + 1)

    Q_sca = (2/(x**2)) * np.sum(fact * (np.abs(a)**2 + np.abs(b)**2), axis=0)
    Q_ext = (2/(x**2)) * np.sum(fact * np.real(a + b), axis=0)
    Q_abs = Q_ext - Q_sca

    return Q_sca, Q_abs

def core_shell_mie(radius_core, radius_shell, m_core, m_shell,
                   wavelengths, n_medium=1.0):
    """
    Compute Q_sca and Q_abs for a sphere with:
      • a core of radius_core & index m_core
      • a shell out to radius_shell with index m_shell
      • embedded in medium of index n_medium
    """
    # Part 1: size parameters for core & shell
    x_core  = 2 * np.pi * radius_core * n_medium / wavelengths
    x_shell = 2 * np.pi * radius_shell * n_medium / wavelengths

    # Part 2: multipole cutoff based on the shell size parameter
    #    one cutoff per λ, then pick the largest
    n_max_array = np.round(x_shell + 4 * x_shell**(1/3) + 2).astype(int)
    n_max = int(n_max_array.max())

    # Part 3: pre‐allocate coefficient arrays
    #    a_core/b_core for the inner boundary, a/b for the outer
    a_core = np.zeros((n_max, len(wavelengths)), dtype=complex)
    b_core = np.zeros_like(a_core)
    a      = np.zeros_like(a_core)
    b      = np.zeros_like(a_core)

    # (matching logic will go here in the next step)

    # for now, keep returning zeros so the stub still passes your test
    Q_sca = np.zeros_like(wavelengths, dtype=float)
    Q_abs = np.zeros_like(wavelengths, dtype=float)

    return Q_sca, Q_abs


if __name__ == "__main__":
    radius = 50e-9           # 50 nm
    m_rel = 1.5 + 0.1j       # example index
    wavelengths = np.array([400e-9, 600e-9, 800e-9])


    Q_sca, Q_abs = single_layer_mie(radius, m_rel, wavelengths)
    print("Q_sca:", Q_sca)
    print("Q_abs:", Q_abs)

Q_sca2, Q_abs2 = core_shell_mie(
    radius_core=40e-9,
    radius_shell=60e-9,
    m_core=1.6+0.2j,
    m_shell=1.4+0.05j,
    wavelengths=wavelengths
)
print("core–shell Q_sca:", Q_sca2)
print("core–shell Q_abs:", Q_abs2)




