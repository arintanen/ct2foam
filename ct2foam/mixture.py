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
from .nasa7 import plot_nasa7_fit

from .transport import TransportData
from .transport import Sutherland
from .transport import Polynomial
from .transport import LogPolynomial
from .transport import plot_transport_fits
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
        self.log_polynomial: LogPolynomial = log_polynomial

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
        tol_nasa7: float = 1e-2,
        tol_nasa7_c0: float = 1e-6,
        tol_transport: float = 1e-1
    ) -> Self:  # TODO: add hint to other from funcs.
        """
        Construct based on cantera Solution. It is assumed the gas object has
        correct mixture definition already.
        """
        gas = ct.Solution(mechanism_file)
        gas.transport_model = "multicomponent"
        gas.TPX = 300, ct.one_atm, mixture

        nasa7 = NASA7Polynomial.from_ct_mixture(gas, Tmin, Tmax, Tmid, n)
        T = np.linspace(Tmin, Tmax, n)
        sutherland = Sutherland.from_ct(gas, T)
        polynomial = Polynomial.from_ct(gas, T)
        log_polynomial = LogPolynomial.from_ct(gas, T)

        # Check NASA7 quality raise if poor quality
        quality = nasa7.quality
        c0 = quality["c0_continuity"]
        max_c0 = max(c0["cp"], c0["dcpdT"], c0["h"], c0["s"])

        err_cp = quality["consistency"]["cp"]
        err_h = quality["consistency"]["h"]
        err_s = quality["consistency"]["s"]
        max_err = max(err_cp, err_h, err_s)

        ref_data_thermo = ThermoData.from_ct_mixture(gas, T)
        ref_data_transport = TransportData.from_ct(gas, np.linspace(Tmin, Tmax, n))


        if max_c0 > tol_nasa7_c0 or max_err > tol_nasa7:
            import tempfile
            fig_path = Path(tempfile.NamedTemporaryFile(suffix=".png").name)
            plot_nasa7_fit(ref_data_thermo, nasa7, fig_path)
            raise ValueError(
                f"NASA7 polynomial quality failed with tol={tol_nasa7}."
                f"Overall fit quality is as follows:\n{quality}"
                f"\nPlease see figure: {fig_path}"
            )

        for tfi in [sutherland, polynomial, log_polynomial]:
            err = tfi.evaluate_transport_fit_quality(ref_data)
            if err["mu"] > tol_transport or err["kappa"] > tol_transport:
                import tempfile
                fig_path = Path(tempfile.NamedTemporaryFile(suffix=".png").name)
                plot_transport_fits(
                    ref_data_transport,
                    [sutherland, polynomial, log_polynomial],
                    fig_path
                )
                raise ValueError(
                    "Transport function fit error is too large:"
                    f"\n    L2 error for viscosity: {err["mu"]}"
                    f"\n    L2 error for conductivity: {err["kappa"]}"
                    f"\nPlease, see figure: {fig_path}"
                )

        X = {name: x for name, x in zip(gas.species_names, gas.X) if x > 0}

        # Plot comparison to the reference data
        if plot:
            print(f"- Saving fit figures under {fig_dir}")
            fig_path = Path(fig_dir, f"{mixture_name}_thermo.png")
            plot_nasa7_fit(ref_data_thermo, nasa7, fig_path)

            plot_transport_fits(
                data=ref_data_transport,
                fits=[sutherland, polynomial, log_polynomial],
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

    # TODO: clean this up after everything else is done.
    # And clean as this is a dupplicate
    # Species not needed to be written but we oculd write for consistency. Now it writes each string character separately. Mayble listify
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
