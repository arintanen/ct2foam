"""Coefficient classes for NASA7, Sutherland and Polynomial transport models."""

import numpy as np


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

    def __init__(self, As, Ts):
        self.As = float(As)
        self.Ts = float(Ts)

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
