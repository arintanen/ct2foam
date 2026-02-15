"""Species class - lightweight container for species metadata and fitted coefficients."""

import numpy as np
import cantera as ct
from .coefficients import NASA7Polynomial, Sutherland, Polynomial


class Species:
    """Lightweight container for species metadata and fitted coefficients.

    This class stores species properties and the results of thermodynamic
    and transport fitting. Data arrays (T, cp, h, s, mu, kappa) are NOT
    stored - they are passed as arguments to fitting and quality check methods.
    """

    def __init__(
        self,
        name,
        W,
        cp0_over_R,
        dhf_over_R,
        s0_over_R,
        elements=None,
    ):
        """Initialize Species with metadata.

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
        self.cp0_over_R = float(cp0_over_R)
        self.dhf_over_R = float(dhf_over_R)
        self.s0_over_R = float(s0_over_R)
        self.elements = elements if elements is not None else {}

        # Fitted coefficients (populated externally)
        self.nasa7 = None
        self.sutherland = None
        self.polynomial = None
        self.log_polynomial = None

        # Quality metrics (populated by check_quality)
        self.quality = None

    def check_quality(
        self,
        T,
        cp,
        h,
        s,
        abs_tol_consistency=0.1,
        cp_tol=1e-6,
        cpdT_tol=0.01,
        h_tol=1e-6,
        s_tol=1e-6,
    ):
        """Check NASA7 fit quality against reference data.

        Args:
            T: Temperature array
            cp: Specific heat (molar) array
            h: Enthalpy (molar) array
            s: Entropy (molar) array
            abs_tol_consistency: Consistency check tolerance
            cp_tol: Continuity cp tolerance
            cpdT_tol: Continuity dcp/dT tolerance
            h_tol: Continuity h tolerance
            s_tol: Continuity s tolerance

        Returns:
            Quality metrics dict

        Raises:
            RuntimeError: If fit_thermo() has not been called
        """
        if self.nasa7 is None:
            raise RuntimeError(
                f"Species {self.name}: NASA7 coefficients not set. "
                "Fitting must be performed before quality check."
            )

        R_gas = ct.gas_constant
        T = np.asarray(T, dtype=float)
        cp = np.asarray(cp, dtype=float)
        h = np.asarray(h, dtype=float)
        s = np.asarray(s, dtype=float)

        cp_over_R = cp / R_gas
        h_over_RT = h / (R_gas * T)
        s_over_R = s / R_gas

        consistency = self.nasa7.check_consistency(
            T, cp_over_R, h_over_RT, s_over_R, abs_tol=abs_tol_consistency
        )
        continuity = self.nasa7.check_continuity(
            cp_tol=cp_tol, cpdT_tol=cpdT_tol, h_tol=h_tol, s_tol=s_tol
        )

        self.quality = {
            "consistency": consistency,
            "continuity": continuity,
        }
        return self.quality

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
