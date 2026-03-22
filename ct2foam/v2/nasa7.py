"""NASA7 polynomial coefficient class for thermodynamic properties."""

from typing import Callable, Self
from numpy import typing as npt
from dataclasses import dataclass

import cantera as ct
import numpy as np

from ct2foam.thermo_transport import lsqlin

_Tstd = 298.15

@dataclass
class ThermoData:
    # Universal gas constant [J/kmol/K]
    gas_constant: float
    # Temperature [K]
    temperature: npt.NDArray[np.floating]
    # Molar heat capacity [J/kmol/K]
    cp: npt.NDArray[np.floating]
    # Molar enthalpy [J/kmol]
    h: npt.NDArray[np.floating]
    # Molar entropy [J/kmol/K]
    s: npt.NDArray[np.floating]
    # Standard-state specific heat [J/kmol/K]: cp(298.15K)
    cp0: float
    # Enthalpy of formation [J/kmol]: h(298.15K)
    dhf: float
    # Standard entropy [J/kmol/K]: s(298.15K) / R
    s0: float

    @classmethod
    def from_ct(
        cls,
        species: ct.Species,
        temperature: npt.NDArray[np.floating]
    ) -> Self:
        """
        Evaluate data for fitting based on Cantera species.
        It is worth noting that infering data via species.thermo.cp() yields
        different floating point value than gas.cp_mole in given mixture temperature.
        Discrepancy is O(1e-9) but enough to influence unit tests compared to old version.
        One needs to be careful when comparin 1-1 results between old and new.
        """
        cp0 = species.thermo.cp(_Tstd)
        dhf = species.thermo.h(_Tstd)
        s0 = species.thermo.s(_Tstd)

        cp = np.zeros_like(temperature)
        h = np.zeros_like(temperature)
        s = np.zeros_like(temperature)
        for i, Ti in enumerate(temperature):
            # Base thermo functions return molar values
            cp[i] = species.thermo.cp(Ti)
            h[i] = species.thermo.h(Ti)
            s[i] = species.thermo.s(Ti)

        return cls(ct.gas_constant, temperature, cp, h, s, cp0, dhf, s0)


class NASA7Polynomial:
    """Encapsulates NASA7 polynomial coefficients and evaluation methods."""

    def __init__(self, coeffs_low, coeffs_high, Tmid, Tmin, Tmax):
        self.coeffs_low = np.asarray(coeffs_low, dtype=float)
        self.coeffs_high = np.asarray(coeffs_high, dtype=float)
        self.Tmid = Tmid
        self.Tlow = Tmin
        self.Tmax = Tmax
        # Fit quality
        self.quality = {
            "c0_continuity": {"cp": 0.0, "dcpdT": 0.0, "h": 0.0, "s": 0.0},
            "consistency": {"cp": 0.0, "h": 0.0, "s": 0.0},
        }

    @classmethod
    def from_ct(
        cls,
        species: ct.Species,
        Tmin: float,
        Tmax: float,
        Tmid: float,
        n: int=128,
        tol_c0: float=1e-6
    ) -> Self:
        """
        Construct from Cantera Species object
        """
        # Use existing NASA7 polynomials if possible
        full_refit_required = False
        cp_refit_required = False

        print(f"\n{species.name} NASA7 polynomial:")

        thermo_type = type(species.thermo).__name__
        coeffs = species.thermo.coeffs

        if thermo_type != "NasaPoly2":
            full_refit_required = True
            print("- Warning: thermo type is not NASA7-Polynomial.")

        if species.thermo.min_temp > Tmin:
            print(f"- Warning: Tmin below limit ({species.thermo.min_temp} > {Tmin})")

        if species.thermo.max_temp < Tmax:
            print(f"- Warning: Tmax above limit ({species.thermo.max_temp} < {Tmin})")

        if np.abs(Tmid - coeffs[0]) / Tmid > tol_c0:
            cp_refit_required = True
            print(f"- Warning: different common temperature: {coeffs[0]} != {Tmid}")

        c_hi = np.array(coeffs[1:8])
        c_lo = np.array(coeffs[8:15])
        nasa7 = cls(c_lo, c_hi, Tmid, Tmin, Tmax)

        continuous = nasa7.is_c0_continuous(tol=tol_c0)
        if not continuous:
            print("- Warning: Existing polynomial not continuous:")
            c0_error = nasa7.continuity_error()
            if c0_error["h"] < tol_c0 and c0_error["s"] < tol_c0:
                print("  - h and s are c0 continuous --> refit cp only.")
                cp_refit_required = True
            else:
                full_refit_required = True

        if not (full_refit_required or cp_refit_required):
            print("- OK: re-using NASA7-polynomials.")
            return nasa7

        _n = int(n / 2)
        _Tl = np.linspace(Tmin, Tmid, _n, endpoint=False)
        _Th = np.linspace(Tmid, Tmax, _n)
        T = np.concatenate((_Tl, _Th))
        thermo_data = ThermoData.from_ct(species, T)

        # Refit cp only
        if not full_refit_required:
            print("- Re-fitting Cp only.")
            nasa7 = cls.fit_cp_only(thermo_data, Tmin, Tmax, Tmid)
            nasa7.fit_quality(thermo_data)
            return nasa7

        # Otherwise carry out full system fit
        print("- Re-fitting full system.")
        nasa7 = cls.fit_full(thermo_data, Tmin, Tmax, Tmid)
        nasa7.fit_quality(thermo_data)
        return nasa7

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


    @classmethod
    def fit_cp_only(cls, data: ThermoData, Tmin, Tmax, Tcommon):
        """Fit NASA7 coefficients using cp/R data only.
        Returns:
            NASA7Polynomial instance
        """
        T = data.temperature
        # Find index closest to Tmid
        Tc_i = np.argmin(np.abs(T - Tcommon))
        T_low = T[0 : Tc_i + 1]
        T_high = T[Tc_i:]
        T_concat = np.concatenate((T_low, T_high))

        cp_over_R = data.cp / data.gas_constant
        cp0_over_R = data.cp0 / data.gas_constant


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
            Aeq[2, i] = _Tstd**i
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

        dhf_over_R = data.dhf / data.gas_constant
        s0_over_R = data.s0 / data.gas_constant
        coeffs_corrected = cls._correct_coeffs(coeffs, Tcommon, dhf_over_R, s0_over_R)

        return cls(coeffs_corrected[:7], coeffs_corrected[7:], Tcommon, Tmin, Tmax)

    @classmethod
    def fit_full(cls, data: ThermoData, Tmin, Tmax, Tcommon):
        """Fit NASA7 coefficients using cp, h, and s data simultaneously.
        Returns:
            NASA7Polynomial instance
        """
        R = data.gas_constant
        T = data.temperature

        # Find index closest to Tmid
        Tc_i = np.argmin(np.abs(T - Tcommon))
        T_low = T[0 : Tc_i + 1]
        T_high = T[Tc_i:]
        T_concat = np.concatenate((T_low, T_high))

        cp_over_R = data.cp / R
        h_over_RT = data.h / (R * T)
        s_over_R = data.s / R

        cp0_over_R = data.cp0 / R
        dhf_over_R = data.dhf / R
        s0_over_R = data.s0 / R

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
        C[i1:i2, 0] = np.ones((1, Nl)) - _Tstd / T_low
        C[i1:i2, 1] = (c1 * T_low) - (_Tstd / T_low) * (c1 * _Tstd)
        C[i1:i2, 2] = (c2 * T_low) ** 2 - (_Tstd / T_low) * (c2 * _Tstd) ** 2
        C[i1:i2, 3] = (c3 * T_low) ** 3 - (_Tstd / T_low) * (c3 * _Tstd) ** 3
        C[i1:i2, 4] = (c4 * T_low) ** 4 - (_Tstd / T_low) * (c4 * _Tstd) ** 4

        i3 = i2
        i4 = i2 + Nh
        C[i3:i4, 0 + M] = np.ones((1, Nh)) - _Tstd / T_high
        C[i3:i4, 1 + M] = (c1 * T_high) - (_Tstd / T_high) * (c1 * _Tstd)
        C[i3:i4, 2 + M] = (c2 * T_high) ** 2 - (_Tstd / T_high) * (c2 * _Tstd) ** 2
        C[i3:i4, 3 + M] = (c3 * T_high) ** 3 - (_Tstd / T_high) * (c3 * _Tstd) ** 3
        C[i3:i4, 4 + M] = (c4 * T_high) ** 4 - (_Tstd / T_high) * (c4 * _Tstd) ** 4

        # s equation
        c2 = (1.0 / 2.0) ** (1.0 / 2.0)
        c3 = (1.0 / 3.0) ** (1.0 / 3.0)
        c4 = (1.0 / 4.0) ** (1.0 / 4.0)

        i5 = i4
        i6 = i4 + Nl
        C[i5:i6, 0] = np.log(T_low / _Tstd)
        C[i5:i6, 1] = T_low - _Tstd
        C[i5:i6, 2] = (c2 * T_low) ** 2 - (c2 * _Tstd) ** 2
        C[i5:i6, 3] = (c3 * T_low) ** 3 - (c3 * _Tstd) ** 3
        C[i5:i6, 4] = (c4 * T_low) ** 4 - (c4 * _Tstd) ** 4

        i7 = i6
        i8 = i6 + Nh
        C[i7:i8, 0 + M] = np.log(T_high / _Tstd)
        C[i7:i8, 1 + M] = T_high - _Tstd
        C[i7:i8, 2 + M] = (c2 * T_high) ** 2 - (c2 * _Tstd) ** 2
        C[i7:i8, 3 + M] = (c3 * T_high) ** 3 - (c3 * _Tstd) ** 3
        C[i7:i8, 4 + M] = (c4 * T_high) ** 4 - (c4 * _Tstd) ** 4

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
            Aeq[2, i] = _Tstd**i

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
        return cls(coeffs_corrected[:7], coeffs_corrected[7:], Tcommon, Tmin, Tmax)

    @staticmethod
    def _correct_coeffs(coeffs, Tcommon, dhf_over_R, s0_over_R):
        """Solve for integration constants ensuring continuity at Tcommon."""
        T_std = _Tstd

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

    def fit_quality(self, data: ThermoData):
        """Check L2 error of coefficients against reference data."""

        # C0 / C1 continuity
        c0 = self.continuity_error()

        # Consistency with reference data
        R = data.gas_constant
        T = data.temperature
        dcp = np.abs(data.cp/R - self.cp_over_R(T))
        err_cp = np.linalg.norm(dcp) / np.linalg.norm(data.cp/R)
        dh = np.abs(data.h/(R*T) - self.h_over_RT(T))
        err_h = np.linalg.norm(dh) / np.linalg.norm(data.h/(R*T))
        ds = np.abs(data.s/R - self.s_over_R(T))
        err_s = np.linalg.norm(ds) / np.linalg.norm(data.s/R)

        quality = {
            "c0_continuity": c0,
            "consistency": {"cp": err_cp, "h": err_h, "s": err_s},
        }
        self.quality = quality
        return quality


    def c0_continuity(self, func: Callable):
        """
        Evaluate C0 continuity for a given function
        """
        val_low = func(self.coeffs_low, self.Tmid)
        val_high = func(self.coeffs_high, self.Tmid)
        return np.abs(val_low - val_high) / max(np.abs(val_low), 1e-12)

    def continuity_error(self) -> dict:
        # C0 / C1 continuity
        cp_c0 = self.c0_continuity(func=self._cp)
        dcpdT_c0 = self.c0_continuity(func=self._dcpdT)
        h_c0 = self.c0_continuity(func=self._h)
        s_c0 = self.c0_continuity(func=self._s)
        quality = {"cp": cp_c0, "dcpdT": dcpdT_c0, "h": h_c0, "s": s_c0}
        return quality

    def is_c0_continuous(self, tol=1e-6) -> bool:
        quality = self.continuity_error()
        return max(quality["cp"], quality["dcpdT"], quality["h"], quality["s"]) < tol
