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
