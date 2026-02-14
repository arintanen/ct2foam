"""Per-species dataset with embedded thermo and transport fitting."""

import numpy as np
from scipy.optimize import curve_fit
import cantera as ct
from ct2foam.thermo_transport import lsqlin
from .coefficients import NASA7Polynomial, Sutherland, Polynomial


_T_STD = 298.15


# TODO: potentially rename something like: speciesDataFitter???
class SpeciesDataset:
    """One instance per species/mixture; holds data and fitting results."""

    # TODO: add notes that this can be used with numerical or experimental data.
    def __init__(
        self,
        name,
        T,
        cp,
        h,
        s,
        mu,
        kappa,
        cp0_over_R,
        dhf_over_R,
        s0_over_R,
        Tmid,
        W=None,
        cv_mole=None,
        elements=None,
        cantera_nasa7=None,
        is_nasa7=True,
    ):
        self.name = str(name)
        self.T = np.asarray(T, dtype=float)
        self.cp = np.asarray(cp, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.cp0_over_R = float(cp0_over_R)
        self.dhf_over_R = float(dhf_over_R)
        self.s0_over_R = float(s0_over_R)
        self.Tmid = float(Tmid)
        self.W = float(W) if W is not None else None
        self.cv_mole = np.asarray(cv_mole, dtype=float) if cv_mole is not None else None
        self.elements = elements
        self.cantera_nasa7 = cantera_nasa7 # TODO: rename
        self.is_nasa7 = is_nasa7
        # Fitting results (populated by fit methods)
        self.nasa7 = None
        self.sutherland = None
        self.polynomial = None
        self.log_polynomial = None
        self.quality = None

    # ------------------------------------------------------------------ #
    #  Thermo fitting (embedded from thermo_fitter.py)
    # ------------------------------------------------------------------ #

    def fit_thermo(
        self, strategy="auto", tolerances=None, force_refit=False, verbose=False
    ):
        """Fit NASA7 polynomials with smart reuse of Cantera coefficients.

        Logic:
        1. If cantera_nasa7 available and not force_refit:
           - Check consistency (must pass or raise RuntimeError)
           - Check continuity AND Tmid match → REUSE
           - Check continuity but Tmid differs → cp_only refit
           - Not continuous → full refit
        2. If non-NASA7 format: print warning, use full refit
        3. Otherwise: fit based on strategy

        Args:
            strategy: 'auto', 'cp_only', or 'full'
            tolerances: FittingTolerances instance (defaults to default())
            force_refit: If True, skip Cantera coefficient reuse check
            verbose: If True, print fitting decisions

        Returns:
            NASA7Polynomial object

        Raises:
            RuntimeError: If Cantera coefficients fail consistency check
        """
        if tolerances is None:
            from .fitting_tolerances import FittingTolerances

            tolerances = FittingTolerances.default()

        R_gas = ct.gas_constant
        T = self.T
        Tc_i = int(np.argmin(np.abs(T - self.Tmid)))

        cp_over_R = self.cp / R_gas
        h_over_RT = self.h / (R_gas * T)
        s_over_R = self.s / R_gas

        # ===== STEP 1: Check if can reuse Cantera coefficients =====
        # Only check for reuse if strategy is "auto" (default behavior)
        if (
            strategy == "auto"
            and not force_refit
            and self.is_nasa7
            and self.cantera_nasa7 is not None
        ):
            ct_nasa = self.cantera_nasa7

            # Check 1: Consistency (MUST PASS or raise error)
            consistency = ct_nasa.check_consistency(
                T,
                cp_over_R,
                h_over_RT,
                s_over_R,
                abs_tol=tolerances.consistency_abs_tol,
            )

            if not consistency["is_consistent"]:
                raise RuntimeError(
                    f"Species {self.name}: Cantera NASA7 coefficients are INCONSISTENT "
                    f"with evaluated thermodynamic data. This indicates a problem with the "
                    f"mechanism file. Errors: cp={consistency['cp_error']:.2e}, "
                    f"h={consistency['h_error']:.2e}, s={consistency['s_error']:.2e}"
                )

            # Check 2: Continuity
            continuity = ct_nasa.check_continuity(
                cp_tol=tolerances.continuity_cp_tol,
                cpdT_tol=tolerances.continuity_cpdT_tol,
                h_tol=tolerances.continuity_h_tol,
                s_tol=tolerances.continuity_s_tol,
            )

            is_continuous = continuity["is_continuous"]

            # Check 3: Tmid match
            tmid_matches = (
                abs(ct_nasa.Tmid - self.Tmid) / self.Tmid < tolerances.tmid_rel_tol
            )

            # Decide: reuse, cp_only, or full refit
            if is_continuous and tmid_matches:
                # REUSE: Cantera coefficients are perfect
                if verbose:
                    print(f"{self.name}: Reusing Cantera NASA7 coefficients")
                self.nasa7 = ct_nasa
                return ct_nasa
            elif is_continuous:
                # cp-only refit (continuous but different Tmid)
                if verbose:
                    print(
                        f"{self.name}: Refitting (Tmid differs: {ct_nasa.Tmid:.1f} → {self.Tmid:.1f})"
                    )
                strategy = "cp_only"
            else:
                # Full refit (not continuous)
                if verbose:
                    print(f"{self.name}: Refitting (discontinuous at Tmid)")
                strategy = "full"

        # ===== STEP 2: Handle non-NASA7 formats =====
        if not self.is_nasa7:
            print(
                f"Warning: Species {self.name} has non-NASA7 thermo format. Using full refit."
            )
            strategy = "full"

        # ===== STEP 3: Perform fitting based on strategy =====
        if strategy == "auto":
            # Try cp-only first; fall back to full if consistency fails
            try:
                c_lo, c_hi = self._fit_nasapolys_cp(
                    T, Tc_i, cp_over_R, self.cp0_over_R, self.dhf_over_R, self.s0_over_R
                )
                nasa = NASA7Polynomial(c_lo, c_hi, self.Tmid)
                result = nasa.check_consistency(
                    T, cp_over_R, h_over_RT, s_over_R, abs_tol=0.1
                )
                if result["is_consistent"]:
                    if verbose:
                        print(f"{self.name}: Fitted (cp-only strategy)")
                    self.nasa7 = nasa
                    return nasa
            except Exception:
                pass
            # Full fit
            if verbose:
                print(f"{self.name}: Fitting (full strategy, cp-only failed)")
            c_lo, c_hi = self._fit_nasapolys_full(
                T,
                Tc_i,
                cp_over_R,
                h_over_RT,
                s_over_R,
                self.cp0_over_R,
                self.dhf_over_R,
                self.s0_over_R,
            )
        elif strategy == "cp_only":
            if verbose:
                print(f"{self.name}: Fitting (cp-only strategy)")
            c_lo, c_hi = self._fit_nasapolys_cp(
                T, Tc_i, cp_over_R, self.cp0_over_R, self.dhf_over_R, self.s0_over_R
            )
        elif strategy == "full":
            if verbose:
                print(f"{self.name}: Fitting (full strategy)")
            c_lo, c_hi = self._fit_nasapolys_full(
                T,
                Tc_i,
                cp_over_R,
                h_over_RT,
                s_over_R,
                self.cp0_over_R,
                self.dhf_over_R,
                self.s0_over_R,
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}', use 'auto', 'cp_only', or 'full'"
            )

        nasa = NASA7Polynomial(c_lo, c_hi, self.Tmid)
        self.nasa7 = nasa
        return nasa

    def fit_transport(self, poly_order=3):
        """Fit both Sutherland and Polynomial transport models; returns dict."""
        T = self.T
        mu_data = self.mu
        kappa_data = self.kappa

        # Sutherland
        As, Ts, std_err = self._fit_sutherland(T, mu_data)
        suth = Sutherland(As, Ts)
        self.sutherland = suth

        # Polynomial
        coeffs_mu, coeffs_kappa = self._fit_polynomial(
            T, mu_data, kappa_data, poly_order
        )
        poly = Polynomial(coeffs_mu, coeffs_kappa, poly_type="polynomial")
        self.polynomial = poly

        # Log-polynomial
        logcoeffs_mu, logcoeffs_kappa = self._fit_log_polynomial(
            T, mu_data, kappa_data, poly_order
        )
        logpoly = Polynomial(logcoeffs_mu, logcoeffs_kappa, poly_type="log_polynomial")
        self.log_polynomial = logpoly

        # Error estimates
        mu_suth = suth.mu(T)
        err_mu_suth = float(
            np.linalg.norm(np.abs(mu_data - mu_suth)) / np.linalg.norm(mu_data)
        )

        mu_poly, kappa_poly = poly.mu(T), poly.kappa(T)
        err_mu_poly = float(
            np.linalg.norm(np.abs(mu_data - mu_poly)) / np.linalg.norm(mu_data)
        )
        err_kappa_poly = float(
            np.linalg.norm(np.abs(kappa_data - kappa_poly)) / np.linalg.norm(kappa_data)
        )

        mu_logpoly, kappa_logpoly = logpoly.mu(T), logpoly.kappa(T)
        err_mu_logpoly = float(
            np.linalg.norm(np.abs(mu_data - mu_logpoly)) / np.linalg.norm(mu_data)
        )
        err_kappa_logpoly = float(
            np.linalg.norm(np.abs(kappa_data - kappa_logpoly))
            / np.linalg.norm(kappa_data)
        )

        return {
            "sutherland": suth,
            "polynomial": poly,
            "log_polynomial": logpoly,
            "sutherland_std_err": std_err,
            "err_mu_sutherland": err_mu_suth,
            "err_mu_polynomial": err_mu_poly,
            "err_kappa_polynomial": err_kappa_poly,
            "err_mu_log_polynomial": err_mu_logpoly,
            "err_kappa_log_polynomial": err_kappa_logpoly,
        }

    def check_quality(
        self,
        abs_tol_consistency=0.1,
        cp_tol=1e-6,
        cpdT_tol=0.01,
        h_tol=1e-6,
        s_tol=1e-6,
    ):
        """Check NASA7 fit quality; returns and stores quality dict."""
        if self.nasa7 is None:
            raise RuntimeError("fit_thermo() must be called before check_quality()")
        # R_gas = 8314.46261815324
        R_gas = ct.gas_constant
        T = self.T
        cp_over_R = self.cp / R_gas
        h_over_RT = self.h / (R_gas * T)
        s_over_R = self.s / R_gas

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

    def to_foam_dict(self):
        """Convert fitted data to an OpenFOAM-compatible dict."""
        if self.nasa7 is None:
            raise RuntimeError("fit_thermo() must be called before to_foam_dict()")
        result = {
            "name": self.name,
            "W": self.W,
            "Tmid": self.Tmid,
            "Tlow": float(self.T[0]),
            "Thigh": float(self.T[-1]),
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
        if self.elements is not None:
            result["elements"] = self.elements
        return result

    # ================================================================== #
    #  Private fitting methods (embedded from thermo_fitter.py)
    # ================================================================== #

    @staticmethod
    def _correct_coeffs(coeffs, Tcommon, dhf_over_R, s0_over_R):
        """Solve for integration constants ensuring continuity at Tcommon."""
        T_std = _T_STD

        # coeff[5]: enthalpy at standard conditions
        coeffs[5] = dhf_over_R - (
            coeffs[0] * T_std
            + coeffs[1] * ((1.0 / 2.0) ** (1.0 / 2.0) * T_std) ** 2
            + coeffs[2] * ((1.0 / 3.0) ** (1.0 / 3.0) * T_std) ** 3
            + coeffs[3] * ((1.0 / 4.0) ** (1.0 / 4.0) * T_std) ** 4
            + coeffs[4] * ((1.0 / 5.0) ** (1.0 / 5.0) * T_std) ** 5
        )

        # coeff[5+7]: h continuity at Tcommon
        def h_sum_term_l(T):
            return (
                coeffs[0]
                + coeffs[1] * (1.0 / 2.0) * T
                + coeffs[2] * ((1.0 / 3.0) ** (1.0 / 2.0) * T) ** 2
                + coeffs[3] * ((1.0 / 4.0) ** (1.0 / 3.0) * T) ** 3
                + coeffs[4] * ((1.0 / 5.0) ** (1.0 / 4.0) * T) ** 4
            )

        def h_sum_term_h(T):
            return (
                coeffs[0 + 7]
                + coeffs[1 + 7] * (1.0 / 2.0) * T
                + coeffs[2 + 7] * ((1.0 / 3.0) ** (1.0 / 2.0) * T) ** 2
                + coeffs[3 + 7] * ((1.0 / 4.0) ** (1.0 / 3.0) * T) ** 3
                + coeffs[4 + 7] * ((1.0 / 5.0) ** (1.0 / 4.0) * T) ** 4
            )

        coeffs[5 + 7] = Tcommon * (
            h_sum_term_l(Tcommon) + coeffs[5] / Tcommon - h_sum_term_h(Tcommon)
        )

        # coeff[6]: entropy at standard conditions
        coeffs[6] = s0_over_R - (
            coeffs[0] * np.log(T_std)
            + coeffs[1] * T_std
            + coeffs[2] * ((1.0 / 2.0) ** (1.0 / 2.0) * T_std) ** 2
            + coeffs[3] * ((1.0 / 3.0) ** (1.0 / 3.0) * T_std) ** 3
            + coeffs[4] * ((1.0 / 4.0) ** (1.0 / 4.0) * T_std) ** 4
        )

        # coeff[6+7]: s continuity at Tcommon
        def s_sum_term_l(T):
            return (
                coeffs[0] * np.log(T)
                + coeffs[1] * T
                + coeffs[2] * ((1.0 / 2.0) ** (1.0 / 2.0) * T) ** 2
                + coeffs[3] * ((1.0 / 3.0) ** (1.0 / 3.0) * T) ** 3
                + coeffs[4] * ((1.0 / 4.0) ** (1.0 / 4.0) * T) ** 4
            )

        def s_sum_term_h(T):
            return (
                coeffs[0 + 7] * np.log(T)
                + coeffs[1 + 7] * T
                + coeffs[2 + 7] * ((1.0 / 2.0) ** (1.0 / 2.0) * T) ** 2
                + coeffs[3 + 7] * ((1.0 / 3.0) ** (1.0 / 3.0) * T) ** 3
                + coeffs[4 + 7] * ((1.0 / 4.0) ** (1.0 / 4.0) * T) ** 4
            )

        coeffs[6 + 7] = s_sum_term_l(Tcommon) + coeffs[6] - s_sum_term_h(Tcommon)

        return coeffs

    @staticmethod
    def _fit_nasapolys_cp(T0, Tc_i, cp_over_R, cp0_over_R, dhf_over_R, s0_over_R):
        """Fit cp/R only; derive h and s coefficients analytically."""
        T_std = _T_STD
        Tcommon = T0[Tc_i]
        T_low = T0[0 : Tc_i + 1]
        T_high = T0[Tc_i:]
        T = np.concatenate((T_low, T_high))

        Nl = len(T_low)
        Nh = len(T_high)
        N = len(T)
        M = 5

        C = np.zeros((N, 2 * M))
        d = np.zeros(N)

        for i in range(M):
            C[:Nl, i] = pow(T_low, i)
            C[Nl : Nl + Nh, i + M] = pow(T_high, i)

        d[:Nl] = cp_over_R[0 : Tc_i + 1]
        d[Nl : Nl + Nh] = cp_over_R[Tc_i:]

        Aeq = np.zeros((5, 2 * M))
        beq = np.zeros(5)

        for i in range(M):
            Aeq[0, i] = Tcommon**i
            Aeq[0, i + M] = -(Tcommon**i)
            Aeq[1, i] = Tcommon**i
            Aeq[2, i] = T_std**i
            Aeq[3, i + M] = T[-1] ** i

        beq[0] = 0.0
        beq[1] = cp_over_R[Tc_i]
        beq[2] = cp0_over_R
        beq[3] = cp_over_R[-1]

        Aeq[4, 1] = 1.0
        Aeq[4, 2] = 2.0 * Tcommon
        Aeq[4, 3] = ((3.0 ** (1.0 / 2.0)) * Tcommon) ** 2
        Aeq[4, 4] = ((4.0 ** (1.0 / 3.0)) * Tcommon) ** 3
        Aeq[4, 1 + M] = -1.0
        Aeq[4, 2 + M] = -2.0 * Tcommon
        Aeq[4, 3 + M] = -(((3.0 ** (1.0 / 2.0)) * Tcommon) ** 2)
        Aeq[4, 4 + M] = -(((4.0 ** (1.0 / 3.0)) * Tcommon) ** 3)

        sol = lsqlin.lsqlin(
            C,
            d,
            0,
            None,
            None,
            Aeq,
            beq,
            -1e9,
            1e9,
            None,
            {"show_progress": False, "abstol": 1e-12, "reltol": 1e-8},
        )
        coeffs_tmp = sol["x"]

        coeffs = np.zeros(14)
        for i in range(5):
            coeffs[i] = coeffs_tmp[i]
            coeffs[i + 7] = coeffs_tmp[i + M]

        coeffs_corrected = SpeciesDataset._correct_coeffs(
            coeffs, Tcommon, dhf_over_R, s0_over_R
        )
        return coeffs_corrected[:7], coeffs_corrected[7:]

    @staticmethod
    def _fit_nasapolys_full(
        T0, Tc_i, cp_over_R, h_over_RT, s_over_R, cp0_over_R, dhf_over_R, s0_over_R
    ):
        """Simultaneous least-squares fit of cp, h, and s."""
        T_std = _T_STD
        Tcommon = T0[Tc_i]
        T_low = T0[0 : Tc_i + 1]
        T_high = T0[Tc_i:]
        T = np.concatenate((T_low, T_high))

        cp_over_R_L = cp_over_R[0 : Tc_i + 1]
        cp_over_R_H = cp_over_R[Tc_i:]
        h_over_RT_L = h_over_RT[0 : Tc_i + 1]
        h_over_RT_H = h_over_RT[Tc_i:]
        s_over_R_L = s_over_R[0 : Tc_i + 1]
        s_over_R_H = s_over_R[Tc_i:]

        Nl = len(T_low)
        Nh = len(T_high)
        N = len(T)
        M = 5

        C = np.zeros((3 * N, 2 * M))
        d = np.zeros(3 * N)

        # cp formulation
        for i in range(5):
            C[:Nl, i] = pow(T_low, i)
            C[Nl : Nl + Nh, i + M] = pow(T_high, i)

        # h equation
        c1 = 1.0 / 2.0
        c2 = (1.0 / 3.0) ** (1.0 / 2.0)
        c3 = (1.0 / 4.0) ** (1.0 / 3.0)
        c4 = (1.0 / 5.0) ** (1.0 / 4.0)

        i1 = Nl + Nh
        i2 = i1 + Nl
        C[i1:i2, 0] = np.ones((1, Nl)) - T_std / T_low
        C[i1:i2, 1] = (c1 * T_low) - (T_std / T_low) * (c1 * T_std)
        C[i1:i2, 2] = (c2 * T_low) ** 2 - (T_std / T_low) * (c2 * T_std) ** 2
        C[i1:i2, 3] = (c3 * T_low) ** 3 - (T_std / T_low) * (c3 * T_std) ** 3
        C[i1:i2, 4] = (c4 * T_low) ** 4 - (T_std / T_low) * (c4 * T_std) ** 4

        i3 = i2
        i4 = i2 + Nh
        C[i3:i4, 0 + M] = np.ones((1, Nh)) - T_std / T_high
        C[i3:i4, 1 + M] = (c1 * T_high) - (T_std / T_high) * (c1 * T_std)
        C[i3:i4, 2 + M] = (c2 * T_high) ** 2 - (T_std / T_high) * (c2 * T_std) ** 2
        C[i3:i4, 3 + M] = (c3 * T_high) ** 3 - (T_std / T_high) * (c3 * T_std) ** 3
        C[i3:i4, 4 + M] = (c4 * T_high) ** 4 - (T_std / T_high) * (c4 * T_std) ** 4

        # s equation
        c2 = (1.0 / 2.0) ** (1.0 / 2.0)
        c3 = (1.0 / 3.0) ** (1.0 / 3.0)
        c4 = (1.0 / 4.0) ** (1.0 / 4.0)

        i5 = i4
        i6 = i4 + Nl
        C[i5:i6, 0] = np.log(T_low / T_std)
        C[i5:i6, 1] = T_low - T_std
        C[i5:i6, 2] = (c2 * T_low) ** 2 - (c2 * T_std) ** 2
        C[i5:i6, 3] = (c3 * T_low) ** 3 - (c3 * T_std) ** 3
        C[i5:i6, 4] = (c4 * T_low) ** 4 - (c4 * T_std) ** 4

        i7 = i6
        i8 = i6 + Nh
        C[i7:i8, 0 + M] = np.log(T_high / T_std)
        C[i7:i8, 1 + M] = T_high - T_std
        C[i7:i8, 2 + M] = (c2 * T_high) ** 2 - (c2 * T_std) ** 2
        C[i7:i8, 3 + M] = (c3 * T_high) ** 3 - (c3 * T_std) ** 3
        C[i7:i8, 4 + M] = (c4 * T_high) ** 4 - (c4 * T_std) ** 4

        # RHS
        d[:Nl] = cp_over_R_L
        d[Nl : Nl + Nh] = cp_over_R_H
        d[i1:i2] = h_over_RT_L - dhf_over_R / T_low
        d[i3:i4] = h_over_RT_H - dhf_over_R / T_high
        d[i5:i6] = s_over_R_L - s0_over_R
        d[i7:i8] = s_over_R_H - s0_over_R

        # Constraints
        Aeq = np.zeros((3, 2 * M))
        beq = np.zeros(3)

        for i in range(5):
            Aeq[0, i] = Tcommon**i
            Aeq[0, i + M] = -(Tcommon**i)
            Aeq[2, i] = T_std**i

        Aeq[1, 1] = 1.0
        Aeq[1, 2] = 2.0 * Tcommon
        Aeq[1, 3] = ((3.0 ** (1.0 / 2.0)) * Tcommon) ** 2
        Aeq[1, 4] = ((4.0 ** (1.0 / 3.0)) * Tcommon) ** 3
        Aeq[1, 1 + M] = -1.0
        Aeq[1, 2 + M] = -2.0 * Tcommon
        Aeq[1, 3 + M] = -(((3.0 ** (1.0 / 2.0)) * Tcommon) ** 2)
        Aeq[1, 4 + M] = -(((4.0 ** (1.0 / 3.0)) * Tcommon) ** 3)

        beq[2] = cp0_over_R

        sol = lsqlin.lsqlin(
            C,
            d,
            0,
            None,
            None,
            Aeq,
            beq,
            -1e9,
            1e9,
            None,
            {"show_progress": False, "abstol": 1e-12, "reltol": 1e-8},
        )
        coeffs_tmp = sol["x"]

        coeffs = np.zeros(14)
        for i in range(5):
            coeffs[i] = coeffs_tmp[i]
            coeffs[i + 7] = coeffs_tmp[i + M]

        coeffs_corrected = SpeciesDataset._correct_coeffs(
            coeffs, Tcommon, dhf_over_R, s0_over_R
        )
        return coeffs_corrected[:7], coeffs_corrected[7:]

    # ================================================================== #
    #  Private transport fitting methods (embedded from transport_fitter.py)
    # ================================================================== #

    @staticmethod
    def _sutherland_func(T, As, Ts):
        """Sutherland viscosity formula."""
        return As * np.sqrt(T) / (1.0 + Ts / T)

    @staticmethod
    def _fit_sutherland(T, mu, p0=None):
        """Curve-fit Sutherland model to viscosity data."""
        if p0 is None:
            p0 = np.array([1.0, 1.0])
        popt, pcov = curve_fit(SpeciesDataset._sutherland_func, T, mu, p0=p0)
        As = popt[0]
        Ts = popt[1]
        std_err = np.sqrt(np.diag(pcov))
        return As, Ts, std_err

    @staticmethod
    def _fit_polynomial(T, mu, kappa, poly_order=3):
        """Least-squares polynomial fit for mu and kappa."""
        mu = np.transpose(mu)
        kappa = np.transpose(kappa)
        poly_coeffs_mu = np.polyfit(T, mu, poly_order)
        poly_coeffs_kappa = np.polyfit(T, kappa, poly_order)
        return poly_coeffs_mu, poly_coeffs_kappa

    @staticmethod
    def _fit_log_polynomial(T, mu, kappa, poly_order=3):
        """Least-squares log-polynomial fit for mu and kappa."""
        mu = np.transpose(mu)
        kappa = np.transpose(kappa)
        mu_ = np.log(mu)
        kappa_ = np.log(kappa)
        T_ = np.log(T)
        poly_coeffs_mu = np.polyfit(T_, mu_, poly_order)
        poly_coeffs_kappa = np.polyfit(T_, kappa_, poly_order)
        return poly_coeffs_mu, poly_coeffs_kappa
