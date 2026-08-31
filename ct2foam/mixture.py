"""
Mixture class - lightweight container for a mixture compunding from multiple species.
"""

from pathlib import Path
from typing import Optional, Self

import numpy as np
import cantera as ct

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
    """Lightweight container for mixture metadata and fitted coefficients.

    Similar to ``Species`` but represents a gas mixture rather than a single
    species. Does not store data arrays - they are passed to fitting methods.

    Args:
        name: mixture name / label
        X: mole fraction dict (e.g. ``{'O2': 0.21, 'N2': 0.79}``)
        W: mean molecular weight [kg/kmol]
        nasa7: fitted NASA7 polynomial
        sutherland: fitted Sutherland transport model
        polynomial: fitted polynomial transport model
        log_polynomial: fitted log-polynomial transport model
    """

    def __init__(
        self,
        name: str,
        X: dict,
        W: float,
        nasa7: NASA7Polynomial,
        sutherland: Sutherland,
        polynomial: Polynomial,
        log_polynomial: LogPolynomial,
    ):
        self.name = str(name)
        self.X = X
        self.W = W  # molecular weight
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
        tol_transport: float = 1e-1,
    ) -> Self:
        """Construct by fitting thermo and transport models for a gas mixture.

        Args:
            mechanism_file: path to the Cantera mechanism file
            mixture_name: label for the mixture (used in output)
            mixture: Cantera mixture string (e.g. ``"O2:0.21,N2:0.79"``)
            Tmin: lower temperature bound [K]
            Tmax: upper temperature bound [K]
            Tmid: common NASA7 mid-point temperature [K]
            n: number of sample points for fitting (default: 128)
            plot: if True, save fit quality plots to ``fig_dir``
            fig_dir: directory for plot output (default: cwd)
            tol_nasa7: tolerance for NASA7 consistency check
            tol_nasa7_c0: tolerance for C0 continuity check
            tol_transport: tolerance for transport fit quality check
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
            err = tfi.evaluate_transport_fit_quality(ref_data_transport)
            if err["mu"] > tol_transport or err["kappa"] > tol_transport:
                import tempfile

                fig_path = Path(tempfile.NamedTemporaryFile(suffix=".png").name)
                plot_transport_fits(
                    ref_data_transport,
                    [sutherland, polynomial, log_polynomial],
                    fig_path,
                )
                raise ValueError(
                    "Transport function fit error is too large:"
                    f"\n    L2 error for viscosity: {err['mu']}"
                    f"\n    L2 error for conductivity: {err['kappa']}"
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

    def write_foam(self, output_dir: Path):
        """Write thermo transport data into OpenFOAM format."""
        # Create a dummy species list to use common writer function
        from ct2foam.species import Species, SpeciesList

        species = Species(
            self.name,
            W=self.W,
            elements={},
            nasa7=self.nasa7,
            sutherland=self.sutherland,
            polynomial=self.polynomial,
            log_polynomial=self.log_polynomial,
        )
        species_list = SpeciesList(species=[species])
        writer.write_foam(species_list, output_dir)
