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


if __name__ == "__main__":
    radius = 50e-9           # 50 nm
    m_rel = 1.5 + 0.1j       # example index
    wavelengths = np.array([400e-9, 600e-9, 800e-9])


    Q_sca, Q_abs = single_layer_mie(radius, m_rel, wavelengths)
    print("Q_sca:", Q_sca)
    print("Q_abs:", Q_abs)




