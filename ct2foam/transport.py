from typing import Union, Self, List
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import cantera as ct

from typing import Self
from dataclasses import dataclass

import numpy as np
import cantera as ct

from numpy import typing as npt

from matplotlib import pyplot as plt

@dataclass
class TransportData:
    # Universal gas constant [J/kmol/K]
    gas_constant: float
    # Temperature [K]
    temperature: npt.NDArray[np.floating]
    # Molar heat capacity [J/kmol/K]
    cp: npt.NDArray[np.floating]
    # Viscosity [Pa s]
    mu: npt.NDArray[np.floating]
    # Conductivity [W/m/K]
    k: npt.NDArray[np.floating]
    # Specific heat in constant volume [J/kmol/K]
    cv: npt.NDArray[np.floating]
    # Mean molecular weight [kg/kmol]
    W: float

    @classmethod
    def from_ct(cls, gas: ct.Solution, temperature: npt.NDArray[np.floating]) -> Self:
        """
        Evaluate data for fitting based on Cantera species.
        """
        # For mixtures, we retain the existing X
        X = gas.X
        W = gas.mean_molecular_weight

        n = len(temperature)
        mu = np.zeros(n)
        k = np.zeros(n)
        cp = np.zeros(n)
        cv = np.zeros(n)

        for i, Ti in enumerate(temperature):
            gas.TPX = Ti, ct.one_atm, X
            mu[i] = gas.viscosity
            k[i] = gas.thermal_conductivity
            cp[i] = gas.cp_mole
            cv[i] = gas.cv_mole

        return cls(temperature=temperature, gas_constant=ct.gas_constant, mu=mu, k=k, cp=cp, cv=cv, W=W)


class TransportFunction:
    """
    Base class for viscosity/thermal-conductivity transport fits (Sutherland,
    Polynomial, LogPolynomial). Subclasses must implement mu() and kappa();
    kappa() signatures may differ (e.g. Sutherland's Euken formula also needs
    thermodynamic data), so _kappa_from_data() is the common entry point used
    when only a TransportData reference is available.
    """

    name: str = "transport_function"

    def mu(self, T: Union[float, np.ndarray]):
        """Evaluate viscosity. Must be implemented by subclasses."""
        raise NotImplementedError

    def kappa(self, T: Union[float, np.ndarray], *args, **kwargs):
        """Evaluate thermal conductivity. Must be implemented by subclasses."""
        raise NotImplementedError

    def _kappa_from_data(self, T: Union[float, np.ndarray], reference_data: TransportData):
        """
        Evaluate kappa(T) using a TransportData reference for any auxiliary
        arguments. Default just forwards to kappa(T); Sutherland overrides
        this since its Euken formula also requires cv, W and R.
        """
        return self.kappa(T)

    def evaluate_transport_fit_quality(self, reference_data: TransportData) -> dict:
        """
        Evaluate L2 error between this transport fit and reference data.
        Return dictionary for viscosity and kappa error estimates.
        """
        T = reference_data.temperature

        mu_fit = self.mu(T)
        kappa_fit = self._kappa_from_data(T, reference_data)

        err_mu = np.linalg.norm(mu_fit - reference_data.mu) / np.linalg.norm(reference_data.mu)
        err_kappa = np.linalg.norm(kappa_fit - reference_data.k) / np.linalg.norm(reference_data.k)

        return {"mu": err_mu, "kappa": err_kappa}


class Sutherland(TransportFunction):
    """Sutherland viscosity model with Euken thermal conductivity."""
    def __init__(self, As, Ts):
        self.As = float(As)
        self.Ts = float(Ts)
        self.name = "sutherland"

    @classmethod
    def from_ct(
        cls,
        gas: ct.Solution,
        temperature: np.ndarray
    ) -> Self:
        """
        Build accordingly.
        """
        data = TransportData.from_ct(gas, temperature)
        return cls.fit(data.temperature, data.mu)

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
            Sutherland instance
        """
        if p0 is None:
            p0 = np.array([1.0, 1.0])

        popt, pcov = curve_fit(cls.sutherland_func, T, mu, p0=p0)
        As = popt[0]
        Ts = popt[1]
        # For debugging
        # std_err = np.sqrt(np.diag(pcov))
        # print(f"Sutherland std_err={std_err}")
        return cls(As, Ts)

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

    def _kappa_from_data(self, T, reference_data: TransportData):
        """Evaluate Euken kappa using cv, W and R from the reference data."""
        return self.kappa(T, reference_data.cv, reference_data.W, reference_data.gas_constant)




class Polynomial(TransportFunction):
    """Standard or log-polynomial transport fit."""

    def __init__(self, coeffs_mu, coeffs_kappa, poly_type="polynomial"):
        self.coeffs_mu = np.asarray(coeffs_mu, dtype=float)
        self.coeffs_kappa = np.asarray(coeffs_kappa, dtype=float)
        self.name = "polynomial"

    @classmethod
    def from_ct(
        cls,
        gas: ct.Solution,
        temperature: np.ndarray
    ) -> Self:
        """
        Build accordingly.
        """
        data = TransportData.from_ct(gas, temperature)
        return cls.fit_polynomial(data.temperature, data.mu, data.k)

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

    def mu(self, T):
        """Evaluate viscosity from polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_mu)
        return f(T)

    def kappa(self, T):
        """Evaluate thermal conductivity from polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_kappa)
        return f(T)


class LogPolynomial(TransportFunction):
    """Log-polynomial transport fit."""

    def __init__(self, coeffs_mu, coeffs_kappa):
        self.coeffs_mu = np.asarray(coeffs_mu, dtype=float)
        self.coeffs_kappa = np.asarray(coeffs_kappa, dtype=float)
        self.name = "log-polynomial"

    @classmethod
    def from_ct(
        cls,
        gas: ct.Solution,
        temperature: np.ndarray,
        poly_type: str = "polynomial"
    ) -> Self:
        """
        Build accordingly.
        """
        data = TransportData.from_ct(gas, temperature)
        return cls.fit_log_polynomial(data.temperature, data.mu, data.k)

    @classmethod
    def fit_log_polynomial(cls, T, mu, kappa, poly_order=3):
        """Fit log-polynomial transport coefficients.

        Args:
            T: Temperature array
            mu: Viscosity data
            kappa: Thermal conductivity data
            poly_order: Polynomial order (default 3 from CHEMKIN)

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
        return cls(poly_coeffs_mu, poly_coeffs_kappa)

    def mu(self, T):
        """Evaluate viscosity from log-polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_mu)
        return np.exp(f(np.log(T)))

    def kappa(self, T):
        """Evaluate thermal conductivity from log-polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_kappa)
        return np.exp(f(np.log(T)))


def plot_transport_fits(
    data: TransportData,
    fits: List[TransportFunction],
    file_path: Path,
):
    """
    Create comparison plots with reference data.
    """

    import itertools
    linestyles = itertools.cycle(["-", "--", ":"])
    colors = itertools.cycle(["b", "g", "b"])

    T = data.temperature
    fig = plt.figure(num=2, figsize=(7.5, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(T, data.mu, "-", color="r", label="reference")
    ax2.plot(T, data.k, "-", color="r", label="reference")

    for fit in  fits:
        ax1.plot(T, fit.mu(T), linestyle=next(linestyles), color=next(colors), label=fit.name)
        kappa = fit._kappa_from_data(T, data)

        ax2.plot(T, kappa, linestyle=next(linestyles), color=next(colors), label=fit.name)

    ax1.set_ylabel(r"$\mu$")
    ax2.set_ylabel(r"$\kappa$")
    ax2.set_xlabel(r"$T$[K]")
    plt.legend(loc=4)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close()
