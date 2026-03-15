"""Sutherland viscosity model with Euken thermal conductivity."""
from typing import Union

import numpy as np
from scipy.optimize import curve_fit
import cantera as ct

class Sutherland:
    """Sutherland viscosity model with Euken thermal conductivity."""

    def __init__(self, As, Ts, std_err=None):
        self.As = float(As)
        self.Ts = float(Ts)
        self.std_err = std_err if std_err is None else np.asarray(std_err, dtype=float)

    @classmethod
    def from_ct(cls, gas: ct.Solution, species: ct.Species, n: int = 128):
        """
        Build from ct. TODO
        Mention here that we made a decision to respect cantera Tmin/Tmax limits for now.
        """
        Tmin = species.thermo.min_temp
        Tmax = species.thermo.max_temp

        reactants = species.name + ":1.0"

        T = np.linspace(Tmin, Tmax, n)
        mu = np.zeros(n)
        for i, Ti in enumerate(T):
            gas.TPX = Ti, ct.one_atm, reactants
            mu[i] = gas.viscosity

        return cls.fit(T, mu)

    @staticmethod
    def sutherland_func(T: Union[float, np.ndarray], As, Ts):
        """Sutherland viscosity formula [Pas]."""
        return As * np.sqrt(T) / (1.0 + Ts / T)

    @classmethod
    def fit(cls, T: Union[float, np.ndarray], mu, p0=None):
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

        popt, pcov = curve_fit(cls.sutherland_func, T, mu, p0=p0)
        As = popt[0]
        Ts = popt[1]
        std_err = np.sqrt(np.diag(pcov))

        return cls(As, Ts, std_err=std_err)

    def mu(self, T: Union[float, np.ndarray]):
        """
        Evaluate viscosity based on Sutherland formulation. Note, that
        while the original formulation is mu = mu0*(T0+C)/(T+C)*(T/T0)^(3/2),
        here a simplified version is used (as in OpenFOAM):
        mu = As*sqrt(T)/(1.0 + Ts/T);
        Return: viscosity [Pas]
        """
        return self.sutherland_func(T, self.As, self.Ts)

    def kappa(self, T: Union[float, np.ndarray], cv_mole, W, R):
        """
        Evaluate Euken thermal conductivity based on OpenFOAM formulation:
        mu * Cv * (1.32 + 1.77*R_specific/Cv).
        Return: conductivity [W/mK]
        """
        mu_val = self.mu(T)
        Cv = cv_mole / W
        Rspecific = R / W
        return mu_val * Cv * (1.32 + 1.77 * Rspecific / Cv)
