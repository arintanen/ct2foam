"""Species class - lightweight container for species metadata and fitted coefficients."""

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
import ct2foam.v2.foam_writer as writer


class Species:
    """
    Lightweight container for species metadata and fitted coefficients.

    Virtually similar to Cantera's Species class. Not inherited to allow
    extendability to e.g. experimental data and to avoid sudden API
    changes.

    This class stores species properties and the results of thermodynamic
    and transport fitting. Data arrays (T, cp, h, s, mu, kappa) are NOT
    stored - they are passed as arguments to fitting and quality check methods.
    """

    def __init__(
        self,
        name,
        W,
        elements={},
        nasa7=None,
        sutherland=None,
        polynomial=None,
        log_polynomial=None,
    ):
        """
        Initialize Species with metadata.

        Args:
            name: Species name
            W: Molecular weight (kg/kmol)
            cp0_over_R: Standard-state cp/R at 298.15K
            dhf_over_R: Standard-state enthalpy/R at 298.15K
            s0_over_R: Standard-state entropy/R at 298.15K
            elements: Elemental composition dict (e.g., {'C': 1, 'H': 4})
        """
        self.name = str(name)
        self.W = float(W)
        self.elements = elements
        self.nasa7: NASA7Polynomial = nasa7
        self.sutherland: Sutherland = sutherland
        self.polynomial: Polynomial = polynomial
        self.log_polynomial: Polynomial = log_polynomial

    @classmethod
    def from_ct(
        cls,
        gas: ct.Solution,
        species_name: str,
        Tmin: float = 200,
        Tmax: float = 3000,
        Tmid: float = 1000,
        n: int = 128,
        tol: float = 1e-2,
        tol_c0: float = 1e-6,
    ) -> Self:  # TODO: add hint to other from funcs.
        """
        Construct based on cantera Species object.
        """
        species = gas.species(gas.species_index(species_name))
        W = species.molecular_weight
        elements = species.composition
        nasa7 = NASA7Polynomial.from_ct(species, Tmin, Tmax, Tmid, n, tol_c0)
        sutherland = Sutherland.from_ct(gas, species, n)
        polynomial = Polynomial.from_ct(gas, species, poly_type="polynomial", n=n)
        log_polynomial = Polynomial.from_ct(
            gas, species, poly_type="log_polynomial", n=n
        )

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
            cls.plot_nasa7_fit(species, nasa7, fig_path)
            raise ValueError(
                f"NASA7 polynomial quality failed with tol={tol}."
                f"Overall fit quality is as follows:\n{quality}"
                f"\nPlease see figure: {fig_path}"
            )

        # TODO: We should raise for other fits as well!!!

        return cls(
            name=species_name,
            W=W,
            elements=elements,
            nasa7=nasa7,
            sutherland=sutherland,
            polynomial=polynomial,
            log_polynomial=log_polynomial,
        )

    @staticmethod
    def plot_nasa7_fit(species: ct.Species, nasa7: NASA7Polynomial, file_path: Path):
        """
        Create comparison plots with reference data.
        """

        T = np.linspace(nasa7.Tlow, nasa7.Tmax, 256)
        data = ThermoData.from_ct(species, T)
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
        species: ct.Species,
        sutherland: Sutherland,
        polynomial: Polynomial,
        logpolynomial: Polynomial,
        file_path: Path,
    ):
        """
        Create comparison plots with reference data.
        """
        Tmin = species.thermo.min_temp
        Tmax = species.thermo.max_temp

        reactants = species.name + ":1.0"
        n = 128
        T = np.linspace(Tmin, Tmax, n)
        mu = np.zeros(n)
        kappa = np.zeros(n)
        cv_mole = np.zeros(n)
        for i, Ti in enumerate(T):
            gas.TPX = Ti, ct.one_atm, reactants
            mu[i] = gas.viscosity
            kappa[i] = gas.thermal_conductivity
            cv_mole[i] = gas.cv_mole

        W = species.molecular_weight
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

    # TODO: this is not used!!!
    def to_foam_dict(self, Tlow, Thigh):
        """Convert fitted data to an OpenFOAM-compatible dict.

        Args:
            Tlow: Lower temperature bound of mechanism validity range
            Thigh: Upper temperature bound of mechanism validity range

        Returns:
            Dictionary with OpenFOAM format data

        Raises:
            RuntimeError: If NASA7 coefficients not set
        """
        if self.nasa7 is None:
            raise RuntimeError(
                f"Species {self.name}: NASA7 coefficients not set. "
                "Fitting must be performed before export."
            )

        result = {
            "name": self.name,
            "W": self.W,
            "Tmid": self.nasa7.Tmid,
            "Tlow": Tlow,
            "Thigh": Thigh,
            "nasa7_lo": self.nasa7.coeffs_low.tolist(),
            "nasa7_hi": self.nasa7.coeffs_high.tolist(),
        }

        if self.sutherland is not None:
            result["As"] = self.sutherland.As
            result["Ts"] = self.sutherland.Ts

        if self.polynomial is not None:
            result["poly_mu"] = self.polynomial.coeffs_mu.tolist()
            result["poly_kappa"] = self.polynomial.coeffs_kappa.tolist()

        if self.log_polynomial is not None:
            result["logpoly_mu"] = self.log_polynomial.coeffs_mu.tolist()
            result["logpoly_kappa"] = self.log_polynomial.coeffs_kappa.tolist()

        if self.elements:
            result["elements"] = self.elements

        return result


class SpeciesList:
    """
    Base container class for a list of Species objects with thermo-transport
    fitting functions.

    Here, we can extend to e.g. experimental data by adding new constructors.
    """

    def __init__(self, species: List[Species] = []):
        self.species = species

    @classmethod
    def from_ct_mech(
        cls,
        mechanism_file: str,
        Tmin: float,
        Tmax: float,
        Tmid: float,
        n: int = 128,
        plot: bool = True,
        fig_dir: Path = Path.cwd(),
        tol: float = 1e-2,
        tol_c0: float = 1e-6,
    ) -> Self:
        """
        Build species container based on cantera mechanism file and refit
        any data if found invalid.
        """
        gas = ct.Solution(mechanism_file)
        gas.transport_model = "multicomponent"

        species_list = []
        # Construct Species object and append to a list
        for sp_name in gas.species_names:
            spi = Species.from_ct(gas, sp_name, Tmin, Tmax, Tmid, n, tol, tol_c0)
            species_list.append(spi)

            # Plot comparison to the reference data
            if plot:
                print(f"- Saving {spi.name} fit figures under {fig_dir}")
                ct_spi = gas.species(gas.species_index(spi.name))
                spi.plot_nasa7_fit(
                    species=ct_spi,
                    nasa7=spi.nasa7,
                    file_path=Path(fig_dir, f"{spi.name}_thermo.png")
                )
                spi.plot_transport_fits(
                    gas,
                    species=ct_spi,
                    sutherland=spi.sutherland,
                    polynomial=spi.polynomial,
                    logpolynomial=spi.log_polynomial,
                    file_path=Path(fig_dir, f"{spi.name}_transport.png"),
                )

        return cls(species_list)

    # TODO: clean this up after everything else is done.
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

        names = [sp.name for sp in self.species if sp.nasa7 is not None]
        writer.write_species_list(species_file, names)

        for sp in self.species:
            if sp.nasa7 is None:
                continue

            poly_mu = sp.polynomial.coeffs_mu if sp.polynomial else np.zeros(4)
            poly_kappa = sp.polynomial.coeffs_kappa if sp.polynomial else np.zeros(4)
            logpoly_mu = (
                sp.log_polynomial.coeffs_mu if sp.log_polynomial else np.zeros(4)
            )
            logpoly_kappa = (
                sp.log_polynomial.coeffs_kappa if sp.log_polynomial else np.zeros(4)
            )
            As = sp.sutherland.As if sp.sutherland else 0.0
            Ts = sp.sutherland.Ts if sp.sutherland else 0.0

            writer.write_thermo_transport(
                thermo_file,
                sp.name,
                sp.W,
                As,
                Ts,
                poly_mu,
                poly_kappa,
                logpoly_mu,
                logpoly_kappa,
                sp.nasa7.Tmid,
                sp.nasa7.Tlow,
                sp.nasa7.Tmax,
                sp.nasa7.coeffs_low,
                sp.nasa7.coeffs_high,
                elements=sp.elements,
            )
