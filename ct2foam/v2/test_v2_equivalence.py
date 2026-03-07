"""
Equivalence test comparing old and new implementations on GRI-3.0 mechanism.

Note that as the old implementation had the dcpdT continuity bug, we cannot
compare the full pipeline but only function-wise comparison is done here.
"""
from ct2foam.v2 import Sutherland, Polynomial

import unittest
import numpy as np
import cantera as ct

from ct2foam.thermo_transport.ct_properties import ctThermoTransport
from ct2foam.thermo_transport import thermo_fitter as old_fitter
from ct2foam.thermo_transport import transport_fitter as old_transport_fitter

from ct2foam.v2.nasa7 import ThermoData, NASA7Polynomial

MECHANISM = "gri30.yaml"
TMID = 1000.0
TLOW = 300.0
THIGH = 3000.0
T_EVAL = np.linspace(300, 3000, 128)

class TestV2OldEquivalence(unittest.TestCase):
    """Compare old and new implementations on GRI-3.0 mechanism species-by-species."""

    @classmethod
    def setUpClass(cls):
        """Load mechanism with both old and new implementations."""
        print(f"\nLoading {MECHANISM} with old implementation...")
        cls.old_data = ctThermoTransport(
            MECHANISM, T=T_EVAL.copy(), Tmid=TMID, verbose=True
        )
        cls.old_data.evaluate_properties()

        cls.ct_gas = ct.Solution(MECHANISM)


    def test_source_data(self):
        """
        There is a small discrepancy how old and new properties are evaluated.
        In particular, cp, h and s are more accurate in the new system as they
        are based on direct NASA7 evaluation without gas mixture averaging.
        """
        atol = 1e-11
        rtol = 1e-15

        for i, spi in enumerate(self.old_data.names):
            _sp = self.ct_gas.species(self.ct_gas.species_index(spi))
            new_data = ThermoData.from_ct(
                _sp, self.old_data.T
            )
            np.testing.assert_allclose(self.old_data.cp[i,:], new_data.cp, atol=atol, rtol=rtol)
            np.testing.assert_allclose(self.old_data.cp0_over_R[i], new_data.cp0/ct.gas_constant, atol=atol, rtol=rtol)

            np.testing.assert_allclose(self.old_data.h[i,:], new_data.h, atol=atol, rtol=rtol)
            np.testing.assert_allclose(self.old_data.dhf_over_R[i], new_data.dhf/ct.gas_constant, atol=atol, rtol=rtol)

            np.testing.assert_allclose(self.old_data.s[i,:], new_data.s, atol=atol, rtol=rtol)
            np.testing.assert_allclose(self.old_data.s0_over_R[i], new_data.s0/ct.gas_constant, atol=atol, rtol=rtol)


    def test_nasa7_cp_fit(self):
        """NASA7 coefficients should match reasonably between old and new for all species.

        Both implementations now reuse Cantera coefficients when they are continuous
        and have matching Tmid. Since GRI-3.0 has Tmid=1000.0 matching our choice,
        both should reuse the same coefficients. Small differences may occur due to
        different fitting implementations.
        """
        # When everything is right, we should get exactly the same answer
        nasa7_atol = 1e-30
        nasa7_rtol = 1e-30

        for i, sp_name in enumerate(self.old_data.names):

            # CP only fit
            Tci = np.where(self.old_data.T == TMID)[0][0]
            old_c_lo, old_c_hi = old_fitter.fit_nasapolys_cp(
                self.old_data.T,
                Tci,
                self.old_data.cp[i, :] / ct.gas_constant,
                self.old_data.cp0_over_R[i],
                self.old_data.dhf_over_R[i],
                self.old_data.s0_over_R[i],
            )

            _sp = self.ct_gas.species(self.ct_gas.species_index(sp_name))
            new_data = ThermoData.from_ct(
                _sp, self.old_data.T
            )
            # We must ensure cp data is evaluated in exactly the same way
            new_data.cp = self.old_data.cp[i,:]

            nasa7 = NASA7Polynomial.fit_cp_only(new_data, TLOW, THIGH, TMID)

            new_c_lo = nasa7.coeffs_low
            new_c_hi = nasa7.coeffs_high

            np.testing.assert_allclose(
                old_c_lo, new_c_lo, atol=nasa7_atol, rtol=nasa7_rtol
            )
            np.testing.assert_allclose(
                old_c_hi, new_c_hi, atol=nasa7_atol, rtol=nasa7_rtol
            )


    def test_nasa7_full_fit(self):
        """NASA7 coefficients should match reasonably between old and new for all species.

        Both implementations now reuse Cantera coefficients when they are continuous
        and have matching Tmid. Since GRI-3.0 has Tmid=1000.0 matching our choice,
        both should reuse the same coefficients. Small differences may occur due to
        different fitting implementations.
        """
        nasa7_atol = 1e-30
        nasa7_rtol = 1e-30

        for i, sp_name in enumerate(self.old_data.names):

            # CP only fit
            Tci = np.where(self.old_data.T == TMID)[0][0]
            old_c_lo, old_c_hi = old_fitter.fit_nasapolys_full(
                self.old_data.T,
                Tci,
                self.old_data.cp[i, :] / ct.gas_constant,
                self.old_data.h[i, :] / (ct.gas_constant*self.old_data.T),
                self.old_data.s[i, :] / ct.gas_constant,
                self.old_data.cp0_over_R[i],
                self.old_data.dhf_over_R[i],
                self.old_data.s0_over_R[i],
            )

            _sp = self.ct_gas.species(self.ct_gas.species_index(sp_name))
            new_data = ThermoData.from_ct(
                _sp, self.old_data.T
            )
            # We must ensure cp data is evaluated in exactly the same way
            new_data.cp = self.old_data.cp[i,:]
            new_data.h = self.old_data.h[i,:]
            new_data.s = self.old_data.s[i,:]


            nasa7 = NASA7Polynomial.fit_full(new_data, TLOW, THIGH, TMID)

            new_c_lo = nasa7.coeffs_low
            new_c_hi = nasa7.coeffs_high

            np.testing.assert_allclose(
                old_c_lo, new_c_lo, atol=nasa7_atol, rtol=nasa7_rtol
            )
            np.testing.assert_allclose(
                old_c_hi, new_c_hi, atol=nasa7_atol, rtol=nasa7_rtol
            )


    def test_sutherland_coefficients_equivalence(self):
        """Sutherland As and Ts should match reasonably between old and new for all species.

        Small differences in Ts (temperature parameter) up to ~20K are acceptable due to
        fitting implementation differences.
        """
        for i, sp_name in enumerate(self.old_data.names):

            _sp = self.ct_gas.species(self.ct_gas.species_index(sp_name))
            n = 128

            reactants = _sp.name + ":1.0"
            Tmin = _sp.thermo.min_temp
            Tmax = _sp.thermo.max_temp

            T = np.linspace(Tmin, Tmax, n)
            mu = np.zeros(n)
            for i, Ti in enumerate(T):
                self.ct_gas.TPX = Ti, ct.one_atm, reactants
                mu[i] = self.ct_gas.viscosity

            new_fit = Sutherland.from_ct(self.ct_gas, _sp, n=n)

            old_As, old_Ts, _ = old_transport_fitter.fit_sutherland(T, mu)

            new_As = new_fit.As
            new_Ts = new_fit.Ts

            self.assertAlmostEqual(old_As, new_As)
            self.assertAlmostEqual(old_Ts, new_Ts)

    def test_polynomial_mu_evaluation_equivalence(self):

        for i, sp_name in enumerate(self.old_data.names):

            _sp = self.ct_gas.species(self.ct_gas.species_index(sp_name))
            n = 128

            reactants = _sp.name + ":1.0"
            Tmin = _sp.thermo.min_temp
            Tmax = _sp.thermo.max_temp

            T = np.linspace(Tmin, Tmax, n)
            mu = np.zeros(n)
            kappa = np.zeros(n)

            for i, Ti in enumerate(T):
                self.ct_gas.TPX = Ti, ct.one_atm, reactants
                mu[i] = self.ct_gas.viscosity
                kappa[i] = self.ct_gas.thermal_conductivity

            new_fit = Polynomial.from_ct(self.ct_gas, _sp, poly_type="polynomial", n=n)
            old_c_mu, old_c_kappa = old_transport_fitter.fit_polynomial(T,mu,kappa)
            new_c_mu = new_fit.coeffs_mu
            new_c_kappa = new_fit.coeffs_kappa
            np.testing.assert_allclose(old_c_mu, new_c_mu, atol=1e-30)
            np.testing.assert_allclose(old_c_kappa, new_c_kappa, atol=1e-30)

            new_fit_log = Polynomial.from_ct(self.ct_gas, _sp, poly_type="log_polynomial", n=n)
            old_c_mu, old_c_kappa = old_transport_fitter.fit_log_polynomial(T,mu,kappa)
            new_c_mu = new_fit_log.coeffs_mu
            new_c_kappa = new_fit_log.coeffs_kappa
            np.testing.assert_allclose(old_c_mu, new_c_mu, atol=1e-30)
            np.testing.assert_allclose(old_c_kappa, new_c_kappa, atol=1e-30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
