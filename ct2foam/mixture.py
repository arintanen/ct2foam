"""
Mixture class - lightweight container for a mixture compunding from multiple species.
"""

from typing import List
from pathlib import Path
from typing import Self

from matplotlib import pyplot as plt
import numpy as np
import cantera as ct

# TODO: fix paths eventually
from .nasa7 import NASA7Polynomial
from .nasa7 import ThermoData

from .sutherland import Sutherland
from .polynomial import Polynomial
import ct2foam.foam_writer as writer


class Mixture:
    """
    Lightweight container for mixture metadata and fitted coefficients.

    This class stores species properties and the results of thermodynamic
    and transport fitting, similar to the Species class.
    """

    def __init__(
        self,
        name,
        X,
        W,
        nasa7=None,
        sutherland=None,
        polynomial=None,
        log_polynomial=None,
    ):
        """
        Initialize mixture with metadata.
        """
        self.name = str(name)
        self.X = X
        self.W = W # molecular weight
        self.nasa7: NASA7Polynomial = nasa7
        self.sutherland: Sutherland = sutherland
        self.polynomial: Polynomial = polynomial
        self.log_polynomial: Polynomial = log_polynomial

    @classmethod
    def from_ct(
        cls,
        mechanism_file: Path,
        mixture_name: str,
        mixture: str,
        Tmin: float = 200,
        Tmax: float = 3000,
        Tmid: float = 1000,
        n: int = 128,
        plot: bool = True,
        fig_dir: Path = Path.cwd(),
        tol: float = 1e-2,
        tol_c0: float = 1e-6,
    ) -> Self:  # TODO: add hint to other from funcs.
        """
        Construct based on cantera Solution. It is assumed the gas object has
        correct mixture definition already.
        """
        gas = ct.Solution(mechanism_file)
        gas.transport_model = "multicomponent"
        gas.TPX = 300, ct.one_atm, mixture

        nasa7 = NASA7Polynomial.from_ct_mixture(gas, Tmin, Tmax, Tmid, n)
        sutherland = Sutherland.from_ct(gas, n=n, Tmin=Tmin, Tmax=Tmax)
        polynomial = Polynomial.from_ct(gas, poly_type="polynomial", n=n, Tmin=Tmin, Tmax=Tmax)
        log_polynomial = Polynomial.from_ct(gas, poly_type="log_polynomial", n=n, Tmin=Tmin, Tmax=Tmax)

        # Check NASA7 quality raise if poor quality
        quality = nasa7.quality
        c0 = quality["c0_continuity"]
        max_c0 = max(c0["cp"], c0["dcpdT"], c0["h"], c0["s"])

        err_cp = quality["consistency"]["cp"]
        err_h = quality["consistency"]["h"]
        err_s = quality["consistency"]["s"]
        max_err = max(err_cp, err_h, err_s)

        if max_c0 > tol_c0 or max_err > tol:
            # TODO: test figure path
            # Save plot to tmp
            import tempfile

            fig_path = Path(tempfile.TemporaryFile().name + ".png")
            cls.plot_nasa7_fit(gas, nasa7, fig_path)
            raise ValueError(
                f"NASA7 polynomial quality failed with tol={tol}."
                f"Overall fit quality is as follows:\n{quality}"
                f"\nPlease see figure: {fig_path}"
            )

        # TODO: We should raise for other fits as well!!!

        X = {name: x for name, x in zip(gas.species_names, gas.X) if x > 0}

        # Plot comparison to the reference data
        if plot:
            print(f"- Saving fit figures under {fig_dir}")
            cls.plot_nasa7_fit(gas, nasa7, Path(fig_dir, f"{mixture_name}_thermo.png"),
            )
            cls.plot_transport_fits(
                gas,
                sutherland=sutherland,
                polynomial=polynomial,
                logpolynomial=log_polynomial,
                file_path=Path(fig_dir, f"{mixture_name}_transport.png"),
            )

        return cls(
            name=mixture_name,
            X=X,
            W=gas.mean_molecular_weight,
            nasa7=nasa7,
            sutherland=sutherland,
            polynomial=polynomial,
            log_polynomial=log_polynomial,
        )

    @staticmethod
    def plot_nasa7_fit(gas: ct.Solution, nasa7: NASA7Polynomial, file_path: Path):
        """
        Create comparison plots with reference data.
        """

        # TODO: below is largely duplicated from species class.

        T = np.linspace(nasa7.Tlow, nasa7.Tmax, 256)
        data = ThermoData.from_ct_mixture(gas, T)
        R = data.gas_constant

        fig = plt.figure(num=1, figsize=(7.5, 10))
        ax1 = plt.subplot(311)
        plt.plot(T, data.cp / R, "-", color="r", label="Orig.")
        plt.plot(T, nasa7.cp_over_R(T), "--", color="b", label="Fit")
        ax1.set_ylabel(r"$cp/R$")
        plt.legend(loc=4)

        ax2 = plt.subplot(312)
        plt.plot(T, data.h / (R * T), "-", color="r")
        plt.plot(T, nasa7.h_over_RT(T), "--", color="b")
        ax2.set_ylabel(r"$h/RT$")

        ax3 = plt.subplot(313)
        plt.plot(T, data.s / R, "-", color="r")
        plt.plot(T, nasa7.s_over_R(T), "--", color="b")
        ax3.set_ylabel(r"$s/R$")
        ax3.set_xlabel(r"$T$[K]")

        fig.savefig(file_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_transport_fits(
        gas: ct.Solution,
        sutherland: Sutherland,
        polynomial: Polynomial,
        logpolynomial: Polynomial,
        file_path: Path,
    ):
        """
        Create comparison plots with reference data.
        """
        n = 128
        T = np.linspace(200, 3000, n)
        mu = np.zeros(n)
        kappa = np.zeros(n)
        cv_mole = np.zeros(n)
        for i, Ti in enumerate(T):
            gas.TPX = Ti, ct.one_atm, gas.X
            mu[i] = gas.viscosity
            kappa[i] = gas.thermal_conductivity
            cv_mole[i] = gas.cv_mole

        W = gas.mean_molecular_weight
        R = ct.gas_constant

        fig = plt.figure(num=2, figsize=(7.5, 10))
        ax1 = plt.subplot(211)
        plt.plot(T, mu, "-", color="r", label="Orig.")
        plt.plot(T, sutherland.mu(T), "--", color="b", label="Sutherland")
        plt.plot(T, polynomial.mu(T), "--", color="k", label="Polynomial")
        plt.plot(T, logpolynomial.mu(T), ":", color="g", label="Log-polynomial")
        ax1.set_ylabel(r"$\mu$")
        plt.legend(loc=4)

        ax2 = plt.subplot(212)
        plt.plot(T, kappa, "-", color="r", label="Orig.")
        plt.plot(
            T,
            sutherland.kappa(T, cv_mole, W, R),
            "--",
            color="b",
            label="Sutherland (Euken)",
        )
        plt.plot(T, polynomial.kappa(T), "--", color="k", label="Polynomial")
        plt.plot(T, logpolynomial.kappa(T), ":", color="k", label="Log-polynomial")
        ax2.set_ylabel(r"$\kappa$")
        ax2.set_xlabel(r"$T$[K]")
        plt.legend(loc=4)

        fig.savefig(file_path, bbox_inches="tight")
        plt.close()

    # TODO: clean this up after everything else is done.
    # And clean as this is a dupplicate
    def write_foam(self, output_dir):
        """Write OpenFOAM output files using foam_writer.

        Args:
            output_dir: Directory to write output files

        Raises:
            RuntimeError: If fitting has not been performed yet
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        thermo_file = output_dir / "thermo.foam"
        reactions_file = output_dir / "reactions.foam"
        species_file = output_dir / "species.foam"

        # Remove existing files
        thermo_file.unlink(missing_ok=True)
        reactions_file.unlink(missing_ok=True)
        species_file.unlink(missing_ok=True)

        writer.write_reactions(reactions_file)

        writer.write_species_list(species_file, self.name)

        poly_mu = self.polynomial.coeffs_mu if self.polynomial else np.zeros(4)
        poly_kappa = self.polynomial.coeffs_kappa if self.polynomial else np.zeros(4)
        logpoly_mu = (
            self.log_polynomial.coeffs_mu if self.log_polynomial else np.zeros(4)
        )
        logpoly_kappa = (
            self.log_polynomial.coeffs_kappa if self.log_polynomial else np.zeros(4)
        )
        As = self.sutherland.As if self.sutherland else 0.0
        Ts = self.sutherland.Ts if self.sutherland else 0.0

        writer.write_thermo_transport(
            thermo_file,
            self.name,
            self.W,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            self.nasa7.Tmid,
            self.nasa7.Tlow,
            self.nasa7.Tmax,
            self.nasa7.coeffs_low,
            self.nasa7.coeffs_high,
            elements=None,
        )
