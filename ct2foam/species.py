"""Species class - lightweight container for species metadata and fitted coefficients."""

from typing import List
from pathlib import Path
from typing import Self

import numpy as np
import cantera as ct

# TODO: fix paths eventually
from .nasa7 import NASA7Polynomial
from .nasa7 import ThermoData
from .nasa7 import plot_nasa7_fit

from .transport import (
    Sutherland,
    Polynomial,
    LogPolynomial,
    plot_transport_fits,
    TransportData
)
import ct2foam.foam_writer as writer


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
        self.log_polynomial: LogPolynomial = log_polynomial

    @classmethod
    def from_ct(
        cls,
        gas: ct.Solution,
        species_name: str,
        Tmin: float = 200,
        Tmax: float = 3000,
        Tmid: float = 1000,
        n: int = 128,
        tol_nasa7: float = 1e-2,
        tol_nasa7_c0: float = 1e-6,
        tol_transport: float = 1e-1
    ) -> Self:  # TODO: add hint to other from funcs.
        """
        Construct based on cantera Species object.
        """
        species = gas.species(gas.species_index(species_name))
        W = species.molecular_weight
        elements = species.composition
        nasa7 = NASA7Polynomial.from_ct(species, Tmin, Tmax, Tmid, n, tol_nasa7_c0)
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

        if max_c0 > tol_nasa7_c0 or max_err > tol_nasa7:
            # TODO: test figure path
            # Save plot to tmp
            import tempfile

            fig_path = Path(tempfile.NamedTemporaryFile(suffix=".png").name)
            cls.plot_nasa7_fit(species, nasa7, fig_path)
            raise ValueError(
                f"NASA7 polynomial quality failed with tol={tol_nasa7}."
                f" Overall fit quality is as follows:\n{quality}"
                f"\nPlease see figure: {fig_path}"
            )

        ref_data =  TransportData.from_ct(gas, T)
        for tfi in [sutherland, polynomial, log_polynomial]:
            err = tfi.evaluate_transport_fit_quality(ref_data)
            if err["mu"] > tol_transport or err["kappa"] > tol_transport:
                import tempfile
                fig_path = Path(tempfile.NamedTemporaryFile(suffix=".png").name)
                plot_transport_fits(
                    ref_data,
                    [sutherland, polynomial, log_polynomial],
                    fig_path
                )
                raise ValueError(
                    "Transport function fit error is too large:"
                    f"\n    L2 error for viscosity: {err["mu"]}"
                    f"\n    L2 error for conductivity: {err["kappa"]}"
                    f"\nPlease, see figure: {fig_path}"
                )

        return cls(
            name=species_name,
            W=W,
            elements=elements,
            nasa7=nasa7,
            sutherland=sutherland,
            polynomial=polynomial,
            log_polynomial=log_polynomial,
        )


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

    # Make SpeciesList behave like a Python sequence/iterable
    def __iter__(self):
        """Return iterator over contained Species objects."""
        return iter(self.species)

    def __len__(self):
        """Return number of Species in the list."""
        return len(self.species)

    def __getitem__(self, idx):
        """Indexing access, e.g. species_list[0]."""
        return self.species[idx]

    def __contains__(self, item):
        """Membership test. Accepts Species instance or species.name strings."""
        if isinstance(item, str):
            return any(sp.name == item for sp in self.species)
        return item in self.species

    def append(self, item):
        """Append a Species to the list."""
        self.species.append(item)

    def extend(self, items):
        """Extend list with multiple Species."""
        self.species.extend(items)

    def __repr__(self):
        return f"SpeciesList({self.species!r})"

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
        tol_nasa7: float = 1e-2,
        tol_nasa7_c0: float = 1e-6,
        tol_transport: float = 1e-1
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
            spi = Species.from_ct(gas, sp_name, Tmin, Tmax, Tmid, n, tol_nasa7, tol_nasa7_c0, tol_transport)
            species_list.append(spi)

            # Plot comparison to the reference data
            if plot:
                print(f"- Saving {spi.name} fit figures under {fig_dir}")
                T = np.linspace(spi.nasa7.Tlow, spi.nasa7.Tmax, n)
                # Recreate data to avoid polluting the actual species contstructor
                # with plotting related actions. Pretty fast anyway.
                ct_spi = gas.species(gas.species_index(spi.name))
                ref_data = ThermoData.from_ct(ct_spi, T)
                file_path=Path(fig_dir, f"{spi.name}_thermo.png")
                plot_nasa7_fit(ref_data, spi.nasa7, file_path)

                file_path=Path(fig_dir, f"{spi.name}_transport.png")
                ref_data = TransportData.from_ct(gas, T)
                plot_transport_fits(
                    ref_data,
                    [spi.sutherland, spi.polynomial, spi.log_polynomial],
                    file_path
                )

        return cls(species_list)

    def write_foam(self, output_dir):
        """Write thermo transport data into OpenFOAM format."""
        writer.write_foam(self, output_dir)
