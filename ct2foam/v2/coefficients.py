"""Coefficient classes for NASA7, Sutherland and Polynomial transport models."""

import numpy as np
from scipy.optimize import curve_fit
from ct2foam.thermo_transport import lsqlin

_T_STD = 298.15


class NASA7Polynomial:
    """Encapsulates NASA7 polynomial coefficients and evaluation methods."""

    def __init__(self, coeffs_low, coeffs_high, Tmid):
        self.coeffs_low = np.asarray(coeffs_low, dtype=float)
        self.coeffs_high = np.asarray(coeffs_high, dtype=float)
        self.Tmid = float(Tmid)

    # -- private single-range evaluators --

    @staticmethod
    def _cp(c, T):
        return c[0] + c[1] * T + c[2] * pow(T, 2) + c[3] * pow(T, 3) + c[4] * pow(T, 4)

    @staticmethod
    def _h(c, T):
        return (
            c[0]
            + c[1] * T / 2
            + c[2] * pow(T, 2) / 3
            + c[3] * pow(T, 3) / 4
            + c[4] * pow(T, 4) / 5
            + c[5] / T
        )

    @staticmethod
    def _s(c, T):
        return (
            c[0] * np.log(T)
            + c[1] * T
            + c[2] * pow(T, 2) / 2
            + c[3] * pow(T, 3) / 3
            + c[4] * pow(T, 4) / 4
            + c[6]
        )

    @staticmethod
    def _dcpdT(c, T):
        return c[1] + 2.0 * c[2] * T + 3.0 * c[3] * pow(T, 2) + 4.0 * c[4] * pow(T, 3)

    # -- public full-range evaluators --

    def cp_over_R(self, T):
        """Evaluate cp/R over the full temperature range."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        result = np.zeros_like(T)
        lo = T <= self.Tmid
        hi = T > self.Tmid
        result[lo] = self._cp(self.coeffs_low, T[lo])
        result[hi] = self._cp(self.coeffs_high, T[hi])
        return result

    def h_over_RT(self, T):
        """Evaluate h/(RT) over the full temperature range."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        result = np.zeros_like(T)
        lo = T <= self.Tmid
        hi = T > self.Tmid
        result[lo] = self._h(self.coeffs_low, T[lo])
        result[hi] = self._h(self.coeffs_high, T[hi])
        return result

    def s_over_R(self, T):
        """Evaluate s/R over the full temperature range."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        result = np.zeros_like(T)
        lo = T <= self.Tmid
        hi = T > self.Tmid
        result[lo] = self._s(self.coeffs_low, T[lo])
        result[hi] = self._s(self.coeffs_high, T[hi])
        return result

    def dcpdT_over_R(self, T):
        """Evaluate (1/R) * dcp/dT over the full temperature range."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        result = np.zeros_like(T)
        lo = T <= self.Tmid
        hi = T > self.Tmid
        result[lo] = self._dcpdT(self.coeffs_low, T[lo])
        result[hi] = self._dcpdT(self.coeffs_high, T[hi])
        return result

    # -- fitting methods (class methods) --

    @classmethod
    def fit_cp_only(cls, T, cp_over_R, cp0_over_R, dhf_over_R, s0_over_R, Tmid):
        """Fit NASA7 coefficients using cp/R data only.

        Args:
            T: Temperature array
            cp_over_R: Specific heat data (dimensionless)
            cp0_over_R: Standard-state cp/R at 298.15K
            dhf_over_R: Standard-state enthalpy/R at 298.15K
            s0_over_R: Standard-state entropy/R at 298.15K
            Tmid: Midpoint temperature

        Returns:
            NASA7Polynomial instance
        """
        T0 = np.asarray(T, dtype=float)
        cp_over_R = np.asarray(cp_over_R, dtype=float)
        Tmid = float(Tmid)

        # Find index closest to Tmid
        Tc_i = int(np.argmin(np.abs(T0 - Tmid)))
        Tcommon = T0[Tc_i]

        T_low = T0[0 : Tc_i + 1]
        T_high = T0[Tc_i:]
        T_concat = np.concatenate((T_low, T_high))

        Nl = len(T_low)
        Nh = len(T_high)
        N = len(T_concat)
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
            Aeq[2, i] = _T_STD**i
            Aeq[3, i + M] = T_concat[-1] ** i

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

        coeffs_corrected = cls._correct_coeffs(coeffs, Tcommon, dhf_over_R, s0_over_R)
        return cls(coeffs_corrected[:7], coeffs_corrected[7:], Tmid)

    @classmethod
    def fit_full(
        cls, T, cp_over_R, h_over_RT, s_over_R, cp0_over_R, dhf_over_R, s0_over_R, Tmid
    ):
        """Fit NASA7 coefficients using cp, h, and s data simultaneously.

        Args:
            T: Temperature array
            cp_over_R: Specific heat data (dimensionless)
            h_over_RT: Enthalpy data (dimensionless)
            s_over_R: Entropy data (dimensionless)
            cp0_over_R: Standard-state cp/R at 298.15K
            dhf_over_R: Standard-state enthalpy/R at 298.15K
            s0_over_R: Standard-state entropy/R at 298.15K
            Tmid: Midpoint temperature

        Returns:
            NASA7Polynomial instance
        """
        T0 = np.asarray(T, dtype=float)
        cp_over_R = np.asarray(cp_over_R, dtype=float)
        h_over_RT = np.asarray(h_over_RT, dtype=float)
        s_over_R = np.asarray(s_over_R, dtype=float)
        Tmid = float(Tmid)

        # Find index closest to Tmid
        Tc_i = int(np.argmin(np.abs(T0 - Tmid)))
        Tcommon = T0[Tc_i]

        T_low = T0[0 : Tc_i + 1]
        T_high = T0[Tc_i:]
        T_concat = np.concatenate((T_low, T_high))

        cp_over_R_L = cp_over_R[0 : Tc_i + 1]
        cp_over_R_H = cp_over_R[Tc_i:]
        h_over_RT_L = h_over_RT[0 : Tc_i + 1]
        h_over_RT_H = h_over_RT[Tc_i:]
        s_over_R_L = s_over_R[0 : Tc_i + 1]
        s_over_R_H = s_over_R[Tc_i:]

        Nl = len(T_low)
        Nh = len(T_high)
        N = len(T_concat)
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
        C[i1:i2, 0] = np.ones((1, Nl)) - _T_STD / T_low
        C[i1:i2, 1] = (c1 * T_low) - (_T_STD / T_low) * (c1 * _T_STD)
        C[i1:i2, 2] = (c2 * T_low) ** 2 - (_T_STD / T_low) * (c2 * _T_STD) ** 2
        C[i1:i2, 3] = (c3 * T_low) ** 3 - (_T_STD / T_low) * (c3 * _T_STD) ** 3
        C[i1:i2, 4] = (c4 * T_low) ** 4 - (_T_STD / T_low) * (c4 * _T_STD) ** 4

        i3 = i2
        i4 = i2 + Nh
        C[i3:i4, 0 + M] = np.ones((1, Nh)) - _T_STD / T_high
        C[i3:i4, 1 + M] = (c1 * T_high) - (_T_STD / T_high) * (c1 * _T_STD)
        C[i3:i4, 2 + M] = (c2 * T_high) ** 2 - (_T_STD / T_high) * (c2 * _T_STD) ** 2
        C[i3:i4, 3 + M] = (c3 * T_high) ** 3 - (_T_STD / T_high) * (c3 * _T_STD) ** 3
        C[i3:i4, 4 + M] = (c4 * T_high) ** 4 - (_T_STD / T_high) * (c4 * _T_STD) ** 4

        # s equation
        c2 = (1.0 / 2.0) ** (1.0 / 2.0)
        c3 = (1.0 / 3.0) ** (1.0 / 3.0)
        c4 = (1.0 / 4.0) ** (1.0 / 4.0)

        i5 = i4
        i6 = i4 + Nl
        C[i5:i6, 0] = np.log(T_low / _T_STD)
        C[i5:i6, 1] = T_low - _T_STD
        C[i5:i6, 2] = (c2 * T_low) ** 2 - (c2 * _T_STD) ** 2
        C[i5:i6, 3] = (c3 * T_low) ** 3 - (c3 * _T_STD) ** 3
        C[i5:i6, 4] = (c4 * T_low) ** 4 - (c4 * _T_STD) ** 4

        i7 = i6
        i8 = i6 + Nh
        C[i7:i8, 0 + M] = np.log(T_high / _T_STD)
        C[i7:i8, 1 + M] = T_high - _T_STD
        C[i7:i8, 2 + M] = (c2 * T_high) ** 2 - (c2 * _T_STD) ** 2
        C[i7:i8, 3 + M] = (c3 * T_high) ** 3 - (c3 * _T_STD) ** 3
        C[i7:i8, 4 + M] = (c4 * T_high) ** 4 - (c4 * _T_STD) ** 4

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
            Aeq[2, i] = _T_STD**i

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

        coeffs_corrected = cls._correct_coeffs(coeffs, Tcommon, dhf_over_R, s0_over_R)
        return cls(coeffs_corrected[:7], coeffs_corrected[7:], Tmid)

    @classmethod
    def fit_auto(
        cls,
        T,
        cp_over_R,
        h_over_RT,
        s_over_R,
        cp0_over_R,
        dhf_over_R,
        s0_over_R,
        Tmid,
        verbose=False,
    ):
        """Smart fitting: try cp_only first, fall back to full if consistency fails.

        Args:
            T: Temperature array
            cp_over_R: Specific heat data
            h_over_RT: Enthalpy data (dimensionless)
            s_over_R: Entropy data (dimensionless)
            cp0_over_R: Standard-state cp/R at 298.15K
            dhf_over_R: Standard-state enthalpy/R at 298.15K
            s0_over_R: Standard-state entropy/R at 298.15K
            Tmid: Midpoint temperature
            verbose: If True, print fitting decisions

        Returns:
            NASA7Polynomial instance
        """
        # Try cp-only first
        try:
            nasa = cls.fit_cp_only(
                T, cp_over_R, cp0_over_R, dhf_over_R, s0_over_R, Tmid
            )
            result = nasa.check_consistency(
                T, cp_over_R, h_over_RT, s_over_R, abs_tol=0.1
            )
            if result["is_consistent"]:
                if verbose:
                    print("fit_auto: cp-only strategy succeeded")
                return nasa
        except Exception:
            pass

        # Fall back to full fit
        if verbose:
            print("fit_auto: using full strategy (cp-only failed)")
        return cls.fit_full(
            T, cp_over_R, h_over_RT, s_over_R, cp0_over_R, dhf_over_R, s0_over_R, Tmid
        )

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

    # -- quality checks --

    def check_consistency(self, T, cp_over_R, h_over_RT, s_over_R, abs_tol=1e-6):
        """Check L2 error of coefficients against reference data."""
        dcp = np.abs(cp_over_R - self.cp_over_R(T))
        err_cp = np.linalg.norm(dcp)
        dh = np.abs(h_over_RT - self.h_over_RT(T))
        err_h = np.linalg.norm(dh)
        ds = np.abs(s_over_R - self.s_over_R(T))
        err_s = np.linalg.norm(ds)
        max_error = max(err_cp, err_h, err_s)
        is_consistent = max_error < abs_tol
        return {
            "is_consistent": is_consistent,
            "cp_error": float(err_cp),
            "h_error": float(err_h),
            "s_error": float(err_s),
            "max_error": float(max_error),
        }

    def check_continuity(self, cp_tol=1e-6, cpdT_tol=0.01, h_tol=1e-6, s_tol=1e-6):
        """Check C0/C1 continuity at Tmid."""
        Tm = self.Tmid
        cp_jump = float(
            np.abs(self._cp(self.coeffs_low, Tm) - self._cp(self.coeffs_high, Tm))
        )
        cpdT_jump = float(
            np.abs(self._dcpdT(self.coeffs_low, Tm) - self._dcpdT(self.coeffs_high, Tm))
        )
        h_jump = float(
            np.abs(self._h(self.coeffs_low, Tm) - self._h(self.coeffs_high, Tm))
        )
        s_jump = float(
            np.abs(self._s(self.coeffs_low, Tm) - self._s(self.coeffs_high, Tm))
        )
        is_continuous = (
            (cp_jump < cp_tol)
            and (cpdT_jump < cpdT_tol)
            and (h_jump < h_tol)
            and (s_jump < s_tol)
        )
        return {
            "is_continuous": is_continuous,
            "cp_jump": cp_jump,
            "cpdT_jump": cpdT_jump,
            "h_jump": h_jump,
            "s_jump": s_jump,
        }


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


class Polynomial:
    """Standard or log-polynomial transport fit."""

    def __init__(self, coeffs_mu, coeffs_kappa, poly_type="polynomial"):
        self.coeffs_mu = np.asarray(coeffs_mu, dtype=float)
        self.coeffs_kappa = np.asarray(coeffs_kappa, dtype=float)
        if poly_type not in ("polynomial", "log_polynomial"):
            raise ValueError(
                f"poly_type must be 'polynomial' or 'log_polynomial', got '{poly_type}'"
            )
        self.poly_type = poly_type

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

    @classmethod
    def fit_log_polynomial(cls, T, mu, kappa, poly_order=3):
        """Fit log-polynomial transport coefficients.

        Args:
            T: Temperature array
            mu: Viscosity data
            kappa: Thermal conductivity data
            poly_order: Polynomial order (default 3)

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
        return cls(poly_coeffs_mu, poly_coeffs_kappa, poly_type="log_polynomial")

    def mu(self, T):
        """Evaluate viscosity from polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_mu)
        if self.poly_type == "log_polynomial":
            return np.exp(f(np.log(T)))
        return f(T)

    def kappa(self, T):
        """Evaluate thermal conductivity from polynomial coefficients."""
        T = np.asarray(T, dtype=float)
        f = np.poly1d(self.coeffs_kappa)
        if self.poly_type == "log_polynomial":
            return np.exp(f(np.log(T)))
        return f(T)
