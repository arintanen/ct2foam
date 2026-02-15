"""Sutherland viscosity model with Euken thermal conductivity."""

import numpy as np
from scipy.optimize import curve_fit


class Sutherland:
    """Sutherland viscosity model with Euken thermal conductivity."""

    def __init__(self, As, Ts, std_err=None):
        self.As = float(As)
        self.Ts = float(Ts)
        self.std_err = std_err if std_err is None else np.asarray(std_err, dtype=float)

    @classmethod
    def fit(cls, T, mu, p0=None):
        """Fit Sutherland parameters from viscosity data.

        Args:
            T: Temperature array
            mu: Viscosity data
            p0: Initial guess [As, Ts] (default: [1.0, 1.0])

        Returns:
            Sutherland instance with std_err attribute set
        """
        if p0 is None:
            p0 = np.array([1.0, 1.0])

        def sutherland_func(T, As, Ts):
            """Sutherland viscosity formula."""
            return As * np.sqrt(T) / (1.0 + Ts / T)

        popt, pcov = curve_fit(sutherland_func, T, mu, p0=p0)
        As = popt[0]
        Ts = popt[1]
        std_err = np.sqrt(np.diag(pcov))

        return cls(As, Ts, std_err=std_err)

    def mu(self, T):
        """
        Evaluate viscosity based on Sutherland formulation. Note, that
        while the original formulation is mu = mu0*(T0+C)/(T+C)*(T/T0)^(3/2),
        here a simplified version is used (as in OpenFOAM):
        mu = As*sqrt(T)/(1.0 + Ts/T);
        """
        T = np.asarray(T, dtype=float)
        return self.As * np.sqrt(T) / (1.0 + self.Ts / T)

    def kappa_euken(self, T, cv_mole, W, R):
        """
        Evaluate Euken thermal conductivity based on OpenFOAM formulation:
        mu * Cv * (1.32 + 1.77*R_specific/Cv).
        """
        mu_val = self.mu(T)
        Cv = cv_mole / W
        Rspecific = R / W
        return mu_val * Cv * (1.32 + 1.77 * Rspecific / Cv)
