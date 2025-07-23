import numpy as np
from scipy import special
from typing import Tuple

class MieSolver:
    def __init__(self, n_medium=1.0):
        """
        n_medium: refractive index of the surrounding medium
        """
        self.n_medium = n_medium

    def spherical_bessel_jn(self, n, z):
        """
        Compute the spherical Bessel function j_n(z).
        """
        return special.spherical_jn(n, z)

    def spherical_hankel1(self, n, z):
        """
        Compute the spherical Hankel h_n^(1)(z) = j_n + i y_n.
        """
        return special.spherical_jn(n, z) + 1j * special.spherical_yn(n, z)

    def psi(self, n, z):
        """
        Riccati–Bessel ψₙ(z) = z · jₙ(z)
        """
        return z * self.spherical_bessel_jn(n, z)

    def xi(self, n, z):
        """
        Riccati–Hankel ξₙ(z) = z · hₙ⁽¹⁾(z)
        """
        return z * self.spherical_hankel1(n, z)

    def deriv(self, f, n, z, dz=1e-6):
        """
        Numerical derivative of f(n, z) w.r.t. z via central difference.
        f should be a bound method like self.psi or self.xi.
        """
        return (f(n, z + dz) - f(n, z - dz)) / (2 * dz)

    def single_layer(self, radius, m_rel, wavelengths):
        """
        Compute Mie Q_sca and Q_abs for a homogeneous sphere.
        """
        # size parameter
        x = 2 * np.pi * radius * self.n_medium / wavelengths

        # multipole cutoff per λ, then pick the max
        n_max_array = np.round(x + 4 * x ** (1 / 3) + 2).astype(int)
        n_max = int(n_max_array.max())

        # prepare storage
        a = np.zeros((n_max, len(wavelengths)), dtype=complex)
        b = np.zeros_like(a)

        # compute a_n and b_n
        for i, lam in enumerate(wavelengths):
            xi_val = x[i]
            mxi = m_rel * xi_val

            for n in range(1, n_max + 1):
                psi_x = self.psi(n, xi_val)
                psi_mx = self.psi(n, mxi)
                xi_x = self.xi(n, xi_val)

                psi_x_p = self.deriv(self.psi, n, xi_val)
                psi_mx_p = self.deriv(self.psi, n, mxi)
                xi_x_p = self.deriv(self.xi, n, xi_val)

                num_a = m_rel * psi_mx * psi_x_p - psi_x * psi_mx_p
                den_a = m_rel * psi_mx * xi_x_p - xi_x * psi_mx_p
                a[n - 1, i] = num_a / den_a

                num_b = psi_mx * psi_x_p - m_rel * psi_x * psi_mx_p
                den_b = psi_mx * xi_x_p - m_rel * xi_x * psi_mx_p
                b[n - 1, i] = num_b / den_b

        # sum efficiencies
        orders = np.arange(1, n_max + 1)[:, None]
        fact = (2 * orders + 1)
        Q_sca = (2 / (x ** 2)) * np.sum(fact * (np.abs(a) ** 2 + np.abs(b) ** 2), axis=0)
        Q_ext = (2 / (x ** 2)) * np.sum(fact * np.real(a + b), axis=0)
        Q_abs = Q_ext - Q_sca

        return Q_sca, Q_abs

    def core_shell(self, radius_core, radius_shell, m_core, m_shell, wavelengths):
        """
        Compute Mie Q_sca and Q_abs for a concentric core+shell sphere.
        """
        # size parameters
        x_core = 2 * np.pi * radius_core * self.n_medium / wavelengths
        x_shell = 2 * np.pi * radius_shell * self.n_medium / wavelengths

        # multipole cutoff
        n_max_array = np.round(x_shell + 4 * x_shell ** (1 / 3) + 2).astype(int)
        n_max = int(n_max_array.max())

        # prepare storage
        a_core = np.zeros((n_max, len(wavelengths)), dtype=complex)
        b_core = np.zeros_like(a_core)
        a = np.zeros_like(a_core)
        b = np.zeros_like(a_core)

        # loop over wavelengths and orders
        for i, lam in enumerate(wavelengths):
            x1 = x_core[i]
            x2 = x_shell[i]
            m1 = m_core / self.n_medium
            m2 = m_shell / self.n_medium

            for n in range(1, n_max + 1):
                # core–shell boundary
                u1 = m1 * x1
                u2 = m2 * x1
                psi_u1 = self.psi(n, u1)
                psi_u1_p = self.deriv(self.psi, n, u1)
                psi_u2 = self.psi(n, u2)
                psi_u2_p = self.deriv(self.psi, n, u2)
                xi_u2 = self.xi(n, u2)
                xi_u2_p = self.deriv(self.xi, n, u2)

                alpha = (m2 * psi_u2 * psi_u1_p - m1 * psi_u1 * psi_u2_p) \
                        / (m2 * xi_u2 * psi_u1_p - m1 * psi_u1 * xi_u2_p)

                # shell–medium boundary
                v = x2
                w = m2 * x2
                psi_v = self.psi(n, v)
                psi_v_p = self.deriv(self.psi, n, v)
                xi_v = self.xi(n, v)
                xi_v_p = self.deriv(self.xi, n, v)
                psi_w = self.psi(n, w)
                psi_w_p = self.deriv(self.psi, n, w)

                num_a = m2 * psi_w * (psi_v_p - alpha * xi_v_p) \
                        - psi_v * (psi_w_p - alpha * xi_u2_p)
                den_a = m2 * psi_w * (xi_v_p - alpha * xi_v_p) \
                        - xi_v * (psi_w_p - alpha * xi_u2_p)
                a[n - 1, i] = num_a / den_a

                num_b = psi_w * (psi_v_p - m2 * alpha * xi_v_p) \
                        - m2 * psi_v * (psi_w_p - alpha * xi_u2_p)
                den_b = psi_w * (xi_v_p - m2 * alpha * xi_v_p) \
                        - m2 * xi_v * (psi_w_p - alpha * xi_u2_p)
                b[n - 1, i] = num_b / den_b

        # sum efficiencies
        orders = np.arange(1, n_max + 1)[:, None]
        fact = (2 * orders + 1)
        Q_sca = (2 / (x_shell ** 2)) * np.sum(fact * (np.abs(a) ** 2 + np.abs(b) ** 2), axis=0)
        Q_ext = (2 / (x_shell ** 2)) * np.sum(fact * np.real(a + b), axis=0)
        Q_abs = Q_ext - Q_sca

        return Q_sca, Q_abs

    def double_shell(
        self,
        radius_core: float,
        radius_shell1: float,
        radius_shell2: float,
        m_core: complex,
        m_shell1: complex,
        m_shell2: complex,
        wavelengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Q_sca and Q_abs for a sphere with
          • core of radius_core & index m_core
          • shell1 out to radius_shell1 with index m_shell1
          • shell2 out to radius_shell2 with index m_shell2
          • embedded in medium of index self.n_medium
        """
        # 1) size parameters
        x1 = 2*np.pi*radius_core   * self.n_medium / wavelengths
        x2 = 2*np.pi*radius_shell1 * self.n_medium / wavelengths
        x3 = 2*np.pi*radius_shell2 * self.n_medium / wavelengths

        # 2) relative indices
        m1 = m_core   / self.n_medium
        m2 = m_shell1 / self.n_medium
        m3 = m_shell2 / self.n_medium

        # 3) multipole cutoff based on outer shell size
        n_max = int(np.round(x3 + 4*x3**(1/3) + 2).max())

        # 4) pre-allocate a/b arrays
        a = np.zeros((n_max, len(wavelengths)), dtype=complex)
        b = np.zeros_like(a)

        # 5) loop over each wavelength & multipole order
        for i, lam in enumerate(wavelengths):
            # size params for this λ
            xi1, xi2, xi3 = x1[i], x2[i], x3[i]

            for n in range(1, n_max+1):
                # --- Core–shell1 interface ---
                u1 = m1*xi1
                u2 = m2*xi1
                ψ_u1   = self.psi(n, u1);   ψ_u1p  = self.deriv(self.psi, n, u1)
                ψ_u2   = self.psi(n, u2);   ψ_u2p  = self.deriv(self.psi, n, u2)
                ξ_u2   = self.xi(n,  u2);   ξ_u2p  = self.deriv(self.xi,  n, u2)

                α = (m2*ψ_u2*ψ_u1p - m1*ψ_u1*ψ_u2p) \
                  / (m2*ξ_u2*ψ_u1p   - m1*ψ_u1*ξ_u2p)

                # --- shell1–shell2 interface ---
                v1 = xi2
                w1 = m2*xi2
                v2 = xi2
                w2 = m3*xi2

                ψ_v1   = self.psi(n, v1);   ψ_v1p  = self.deriv(self.psi, n, v1)
                ξ_v1   = self.xi(n,  v1);   ξ_v1p  = self.deriv(self.xi,  n, v1)
                ψ_w2   = self.psi(n, w2);   ψ_w2p  = self.deriv(self.psi, n, w2)
                ξ_w2   = self.xi(n,  w2);   ξ_w2p  = self.deriv(self.xi,  n, w2)

                β = (m3*ψ_w2*(ψ_v1p - α*ξ_v1p) - ψ_v1*(ψ_w2p - α*ξ_u2p)) \
                  / (m3*ψ_w2*(ξ_v1p - α*ξ_v1p) - ξ_v1*(ψ_w2p - α*ξ_u2p))

                # --- shell2–medium interface (final a_n,b_n) ---
                v = xi3
                w = m3*xi3

                ψ_v    = self.psi(n, v);    ψ_vp   = self.deriv(self.psi, n, v)
                ξ_v    = self.xi(n,  v);    ξ_vp   = self.deriv(self.xi,  n, v)
                ψ_w    = self.psi(n, w);    ψ_wp   = self.deriv(self.psi, n, w)

                num_a = m3*ψ_w*(ψ_vp - β*ξ_vp)   - ψ_v*(ψ_wp - β*ξ_w2p)
                den_a = m3*ψ_w*(ξ_vp - β*ξ_vp)   - ξ_v*(ψ_wp - β*ξ_w2p)
                a[n-1, i] = num_a/den_a

                num_b = ψ_w*(ψ_vp - m3*β*ξ_vp)   - m3*ψ_v*(ψ_wp - β*ξ_w2p)
                den_b = ψ_w*(ξ_vp - m3*β*ξ_vp)   - m3*ξ_v*(ψ_wp - β*ξ_w2p)
                b[n-1, i] = num_b/den_b

        # 6) sum efficiencies (same as single_layer)
        orders = np.arange(1, n_max+1)[:,None]
        fact   = (2*orders + 1)
        Q_sca = (2/(x3**2))*np.sum(fact*(np.abs(a)**2+np.abs(b)**2), axis=0)
        Q_ext = (2/(x3**2))*np.sum(fact*np.real(a+b), axis=0)
        Q_abs = Q_ext - Q_sca

        return Q_sca, Q_abs
