"""Standard or log-polynomial transport models."""

import numpy as np


class Polynomial:
    """Standard or log-polynomial transport fit."""

    def __init__(self, coeffs_mu, coeffs_kappa, poly_type="polynomial"):
        self.coeffs_mu = np.asarray(coeffs_mu, dtype=float)
        self.coeffs_kappa = np.asarray(coeffs_kappa, dtype=float)
        if poly_type not in ("polynomial", "log_polynomial"):
            raise ValueError(
                f"poly_type must be 'polynomial' or 'log_polynomial', got '{poly_type}'"
            )
        self.poly_type = poly_type

    @classmethod
    def fit_polynomial(cls, T, mu, kappa, poly_order=3):
        """Fit standard polynomial transport coefficients.

        Args:
            T: Temperature array
            mu: Viscosity data
            kappa: Thermal conductivity data
            poly_order: Polynomial order (default 3)

        Returns:
            Polynomial instance with poly_type='polynomial'
        """
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        poly_coeffs_mu = np.polyfit(T, mu, poly_order)
        poly_coeffs_kappa = np.polyfit(T, kappa, poly_order)
        return cls(poly_coeffs_mu, poly_coeffs_kappa, poly_type="polynomial")

    @classmethod
    def fit_log_polynomial(cls, T, mu, kappa, poly_order=3):
        """Fit log-polynomial transport coefficients.

        Args:
            T: Temperature array
            mu: Viscosity data
            kappa: Thermal conductivity data
            poly_order: Polynomial order (default 3)

        Returns:
            Polynomial instance with poly_type='log_polynomial'
        """
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        mu_log = np.log(mu)
        kappa_log = np.log(kappa)
        T_log = np.log(T)
        poly_coeffs_mu = np.polyfit(T_log, mu_log, poly_order)
        poly_coeffs_kappa = np.polyfit(T_log, kappa_log, poly_order)
        return cls(poly_coeffs_mu, poly_coeffs_kappa, poly_type="log_polynomial")

    def mu(self, T):
        """Evaluate viscosity from polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_mu)
        if self.poly_type == "log_polynomial":
            return np.exp(f(np.log(T)))
        return f(T)

    def kappa(self, T):
        """Evaluate thermal conductivity from polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_kappa)
        if self.poly_type == "log_polynomial":
            return np.exp(f(np.log(T)))
        return f(T)
