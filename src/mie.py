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

if __name__ == "__main__":
    z = 2.0
    for n in range(3):
        print(f"j_{n}({z}) =", spherical_bessel_jn(n, z))
        print(f"h1_{n}({z}) =", spherical_hankel1(n, z))

def single_layer_mie(radius, m_rel, wavelengths, n_medium=1.0):
    """
    Compute Mie scattering (Q_sca) and absorption (Q_abs)
    efficiencies for a homogeneous sphere.

    radius      : in the same length units as wavelengths
    m_rel       : complex relative index (n_particle / n_medium)
    wavelengths : 1D numpy array of λ values
    n_medium    : refractive index of the surrounding medium
    """
    # Part 1: size parameter
    x = 2 * np.pi * radius * n_medium / wavelengths

    # Part 2: multipole cutoff
    n_max = np.round(x + 4 * x**(1/3) + 2).astype(int)

    # For now, just return them so we can verify:
    return x, n_max

if __name__ == "__main__":
    radius = 50e-9           # 50 nm
    m_rel = 1.5 + 0.1j       # example index
    wavelengths = np.array([400e-9, 600e-9, 800e-9])

    x, n_max = single_layer_mie(radius, m_rel, wavelengths)
    print("size parameters x:", x)
    print("multipole cutoffs n_max:", n_max)
