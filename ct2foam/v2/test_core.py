"""Unit tests for v2 implementation - updated for new API.

This file contains 76 tests originally from test_v2.py, updated to work with:
- New Species class (no data array storage)
- Classmethod-based fitting API
- Removal of Tmid/Tlow/Thigh from Species attributes
"""
from multiprocessing import Value

import unittest
import numpy as np

import cantera as ct

from ct2foam.v2.nasa7 import NASA7Polynomial, ThermoData

from ct2foam.v2.sutherland import Sutherland
from ct2foam.v2.polynomial import Polynomial


# Note, OF_reference/Test-thermoMixture.C
# is used as a source of the reference data tested here.

# TODO: think this through
# R = 8314.46261815324
# OF Reguired for sutherland testing
R_OF = 8314.47006650545  # taken from openFoam -- differs slightly from standards

T_STD = 298.15

H2O_TMID = 1000.0
H2O_TMIN = 300.0
H2O_TMAX = 3000.0
H2O_C_LO = np.array(
    [3.38684, 0.00347498, -6.3547e-06, 6.96858e-09, -2.50659e-12, -30208.1, 2.59023]
)
H2O_C_HI = np.array(
    [2.67215, 0.00305629, -8.73026e-07, 1.201e-10, -6.39162e-15, -29899.2, 6.86282]
)


class TestNASA7PolynomialBasics(unittest.TestCase):
    """Test NASA polynomials and related functionalities."""

    def setUp(self):
        self.nasa = NASA7Polynomial(H2O_C_LO, H2O_C_HI, H2O_TMID, H2O_TMIN, H2O_TMAX)

    def test_cp_over_R_below_midpoint(self):
        """cp/R uses low coefficients when T <= Tmid."""
        T = 400.0
        result = self.nasa.cp_over_R(T)
        expected = NASA7Polynomial._cp(H2O_C_LO, T)
        self.assertAlmostEqual(result[0], expected, places=10)

    def test_cp_over_R_above_midpoint(self):
        """cp/R uses high coefficients when T > Tmid."""
        T = 1500.0
        result = self.nasa.cp_over_R(T)
        expected = NASA7Polynomial._cp(H2O_C_HI, T)
        self.assertAlmostEqual(result[0], expected, places=10)

    def test_cp_over_R_at_midpoint(self):
        """cp/R at Tmid uses low coefficients (T = Tmid)."""
        result = self.nasa.cp_over_R(H2O_TMID)
        expected = NASA7Polynomial._cp(H2O_C_LO, H2O_TMID)
        self.assertAlmostEqual(result[0], expected, places=10)

    def test_cp_over_R_array_vectorization(self):
        """cp/R works with array input and segments correctly."""
        T = np.array([300.0, 500.0, 1000.0, 1500.0, 3000.0])
        result = self.nasa.cp_over_R(T)
        self.assertEqual(result.shape, T.shape)
        # Low region
        for i in [0, 1, 2]:
            self.assertAlmostEqual(
                result[i], NASA7Polynomial._cp(H2O_C_LO, T[i]), places=10
            )
        # High region
        for i in [3, 4]:
            self.assertAlmostEqual(
                result[i], NASA7Polynomial._cp(H2O_C_HI, T[i]), places=10
            )

    def test_h_over_RT_segmentation(self):
        """h/(RT) segments correctly across Tmid."""
        T = np.array([400.0, 2000.0])
        result = self.nasa.h_over_RT(T)
        self.assertAlmostEqual(
            result[0], NASA7Polynomial._h(H2O_C_LO, 400.0), places=10
        )
        self.assertAlmostEqual(
            result[1], NASA7Polynomial._h(H2O_C_HI, 2000.0), places=10
        )

    def test_s_over_R_monotonicity(self):
        """Entropy should increase monotonically with T."""
        T = np.linspace(300, 3000, 50)
        s = self.nasa.s_over_R(T)
        diffs = np.diff(s)
        self.assertTrue(np.all(diffs > -1e-10))

    def test_dcpdT_over_R_finite(self):
        """dcp/dT should be finite and physically reasonable."""
        T = 1000.0
        result = self.nasa.dcpdT_over_R(T)
        self.assertTrue(np.isfinite(result[0]))
        self.assertAlmostEqual(np.abs(result[0]), 0.00164496, places=5)

    def test_dcpdT_over_R_matches_numerical_derivative(self):
        """dcp/dT should approximate numerical derivative."""
        T = 800.0
        dT = 0.01
        dcpdT_analytical = self.nasa.dcpdT_over_R(T)[0]
        cp_T_minus = self.nasa.cp_over_R(T - dT / 2)[0]
        cp_T_plus = self.nasa.cp_over_R(T + dT / 2)[0]
        dcpdT_numerical = (cp_T_plus - cp_T_minus) / dT
        self.assertAlmostEqual(dcpdT_analytical, dcpdT_numerical, places=5)

    def test_continuity_cp_at_Tmid(self):
        """cp should be continuous at Tmid."""
        eps = 1e-6
        T_lo = H2O_TMID - eps
        T_hi = H2O_TMID + eps
        cp_lo = self.nasa.cp_over_R(T_lo)[0]
        cp_hi = self.nasa.cp_over_R(T_hi)[0]
        jump = abs(cp_hi - cp_lo)
        self.assertLess(jump, 1e-4)

    def test_continuity_h_at_Tmid(self):
        """h/(RT) should be continuous at Tmid."""
        eps = 1e-6
        T_lo = H2O_TMID - eps
        T_hi = H2O_TMID + eps
        h_lo = self.nasa.h_over_RT(T_lo)[0]
        h_hi = self.nasa.h_over_RT(T_hi)[0]
        jump = abs(h_hi - h_lo)
        self.assertLess(jump, 1e-4)

    def test_continuity_s_at_Tmid(self):
        """s/R should be continuous at Tmid."""
        eps = 1e-6
        T_lo = H2O_TMID - eps
        T_hi = H2O_TMID + eps
        s_lo = self.nasa.s_over_R(T_lo)[0]
        s_hi = self.nasa.s_over_R(T_hi)[0]
        jump = abs(s_hi - s_lo)
        self.assertLess(jump, 1e-4)


class TestNASA7PolynomialCanteraConsistency(unittest.TestCase):

    def setUp(self):
        self.gas = ct.Solution("h2o2.yaml")
        self.R = ct.gas_constant

        # Extract H2 data
        self.gas.TPX = 1000, ct.one_atm, "H2:1.0"
        h2_idx = self.gas.species_index("H2")

        # Generate evaluation data
        self.T = np.linspace(300, 3000, 500)
        self.Tmid = 1000.0

        nT = len(self.T)
        self.cp = np.zeros(nT)
        self.h = np.zeros(nT)
        self.s = np.zeros(nT)

        for i, T_i in enumerate(self.T):
            self.gas.TP = T_i, ct.one_atm
            self.cp[i] = self.gas.cp_mole
            self.h[i] = self.gas.enthalpy_mole
            self.s[i] = self.gas.entropy_mole

        # Standard state values at 298.15K
        self.gas.TP = T_STD, ct.one_atm
        self.cp0_over_R = self.gas.standard_cp_R[h2_idx]
        self.dhf_over_RT = self.gas.standard_enthalpies_RT[h2_idx]
        self.s0_over_R = self.gas.standard_entropies_R[h2_idx]

    def test_thermo_data_class(self):
        h2 = self.gas.species(self.gas.species_index("H2"))
        data = ThermoData.from_ct(h2, self.T, self.R)
        np.testing.assert_array_almost_equal(data.cp, self.cp)
        np.testing.assert_array_almost_equal(data.h, self.h)
        np.testing.assert_array_almost_equal(data.s, self.s)
        np.testing.assert_array_almost_equal(data.cp0 / self.R, self.cp0_over_R)
        np.testing.assert_array_almost_equal(
            data.dhf / (self.R * self.T), self.dhf_over_RT
        )
        np.testing.assert_array_almost_equal(data.s0 / self.R, self.s0_over_R)

    def test_check_continuity_method(self):
        """check_continuity returns expected keys."""
        nasa7 = NASA7Polynomial(H2O_C_LO, H2O_C_HI, H2O_TMID, H2O_TMIN, H2O_TMAX)
        assert nasa7.is_c0_continuous(tol=1e-5)

    def test_consistency(self):
        h2 = self.gas.species(self.gas.species_index("H2"))
        coeffs = h2.thermo.coeffs
        c_hi = np.array(coeffs[1:8])
        c_lo = np.array(coeffs[8:15])
        nasa7 = NASA7Polynomial(c_lo, c_hi, 1000, 200, 3000)
        for Ti in self.T:
            self.assertAlmostEqual(nasa7.cp_over_R(Ti)[0], h2.thermo.cp(Ti) / self.R)
            self.assertAlmostEqual(
                nasa7.h_over_RT(Ti)[0], h2.thermo.h(Ti) / (self.R * Ti)
            )
            self.assertAlmostEqual(nasa7.s_over_R(Ti)[0], h2.thermo.s(Ti) / self.R)

    def test_check_consistency_method(self):
        """check_consistency returns expected structure."""
        h2 = self.gas.species(self.gas.species_index("H2"))
        data = ThermoData.from_ct(h2, self.T, self.R)
        nasa7 = NASA7Polynomial.from_ct(h2, 200, 3500, 1000)
        result = nasa7.fit_quality(data)["consistency"]
        self.assertIn("cp_error", result)
        self.assertIn("h_error", result)
        self.assertIn("s_error", result)

    def test_poly_representation(self):
        """Check if H2 species is correctly presented"""
        h2 = self.gas.species(self.gas.species_index("H2"))
        coeffs = h2.thermo.coeffs
        c_hi = np.array(coeffs[1:8])
        c_lo = np.array(coeffs[8:15])

        nasa7 = NASA7Polynomial(c_lo, c_hi, 1000, 200, 3500)
        np.testing.assert_allclose(c_hi, nasa7.coeffs_high, rtol=1e-8)
        np.testing.assert_allclose(c_lo, nasa7.coeffs_low, rtol=1e-8)


class TestNASA7PolynomialFitting(unittest.TestCase):
    """Test NASA7Polynomial fitting methods."""

    def setUp(self):
        self.R = ct.gas_constant
        self.gas = ct.Solution("h2o2.yaml")
        self.h2 = self.gas.species(self.gas.species_index("H2"))
        self.T = np.linspace(200, 3000, 128)
        self.data = ThermoData.from_ct(self.h2, self.T, self.R)
        self.Tmin = 200
        self.Tmax = 3000
        self.Tcommon = 1000

    def test_fit_nasapolys_cp_synthetic_data(self):
        """Test cp fitting with synthetic quadratic data."""
        T0 = np.linspace(300, 2000, 50)
        R = ct.gas_constant
        cp = (2.5 + 0.001 * T0 + 1e-5 * T0**2) * R
        h = T0**2 * R
        s = T0**2 * R
        cp0 = 2.8 * R
        dhf = 10.0 * R
        s0 = 100.0 * R
        data = ThermoData(ct.gas_constant, temperature=T0, cp=cp, h=h, s=s, cp0=cp0, dhf=dhf, s0=s0)
        nasa7 = NASA7Polynomial.fit_cp_only(data, 300, 2000, 1000)

        self.assertEqual(len(nasa7.coeffs_low), 7)
        self.assertEqual(len(nasa7.coeffs_high), 7)

        cp_fitted = nasa7.cp_over_R(T0) * ct.gas_constant
        eps = np.linalg.norm(cp_fitted - cp) / np.linalg.norm(cp)
        self.assertLessEqual(eps, 8e-3)

    def test_fit_cp_only(self):

        nasa7 = NASA7Polynomial.fit_cp_only(
            self.data, self.Tmin, self.Tmax, self.Tcommon
        )

        self.assertIsNotNone(nasa7)
        self.assertEqual(nasa7.Tmid, self.Tcommon)
        self.assertEqual(len(nasa7.coeffs_low), 7)
        self.assertEqual(len(nasa7.coeffs_high), 7)

        # Evaluate cp and check fit quality
        cp_fit_low = nasa7.cp_over_R(self.T[self.T <= self.Tcommon])
        cp_fit_high = nasa7.cp_over_R(self.T[self.T > self.Tcommon])

        cp_ref_low = self.data.cp[self.T <= self.Tcommon] / self.R
        cp_ref_high = self.data.cp[self.T > self.Tcommon] / self.R

        # Should have good cp fit (relaxed tolerance for numerical precision)
        np.testing.assert_allclose(cp_fit_low, cp_ref_low, rtol=7e-3)
        np.testing.assert_allclose(cp_fit_high, cp_ref_high, rtol=1.5e-3)


    def test_fit_full(self):

        nasa7 = NASA7Polynomial.fit_full(
            self.data, self.Tmin, self.Tmax, self.Tcommon
        )

        self.assertIsNotNone(nasa7)
        self.assertEqual(nasa7.Tmid, self.Tcommon)
        self.assertEqual(len(nasa7.coeffs_low), 7)
        self.assertEqual(len(nasa7.coeffs_high), 7)

        Tlow = self.T[self.T <= self.Tcommon]
        Thigh = self.T[self.T > self.Tcommon]

        cp_fit_low = nasa7.cp_over_R(Tlow)
        cp_fit_high = nasa7.cp_over_R(Thigh)

        cp_ref_low = self.data.cp[self.T <= self.Tcommon] / self.R
        cp_ref_high = self.data.cp[self.T > self.Tcommon] / self.R

        np.testing.assert_allclose(cp_fit_low, cp_ref_low, rtol=5e-3)
        np.testing.assert_allclose(cp_fit_high, cp_ref_high, rtol=3.5e-3)

        h_fit_low = nasa7.h_over_RT(Tlow)
        h_fit_high = nasa7.h_over_RT(Thigh)

        h_ref_low = self.data.h[self.T <= self.Tcommon] / self.R / Tlow
        h_ref_high = self.data.h[self.T > self.Tcommon] / self.R / Thigh

        np.testing.assert_allclose(h_fit_low, h_ref_low, rtol=2e-3)
        np.testing.assert_allclose(h_fit_high, h_ref_high, rtol=5e-4)

        s_fit_low = nasa7.s_over_R(Tlow)
        s_fit_high = nasa7.s_over_R(Thigh)

        s_ref_low = self.data.s[self.T <= self.Tcommon] / self.R
        s_ref_high = self.data.s[self.T > self.Tcommon] / self.R

        np.testing.assert_allclose(s_fit_low, s_ref_low, rtol=2e-4)
        np.testing.assert_allclose(s_fit_high, s_ref_high, rtol=5e-5)


    def test_from_ct(self):
        h2 = self.gas.species(self.gas.species_index("H2"))
        # Will fail if fit quality not ok
        _ = NASA7Polynomial.from_ct(h2, 200, 3500, 1000)

    def test_consistency(self):
        h2 = self.gas.species(self.gas.species_index("H2"))
        nasa7 = NASA7Polynomial.from_ct(h2, 200, 3500, 1000)
        cp_lo = nasa7._cp(nasa7.coeffs_low, nasa7.Tmid)
        cp_hi = nasa7._cp(nasa7.coeffs_high, nasa7.Tmid)
        self.assertAlmostEqual(cp_lo, cp_hi)

    def test_fit_quality(self):
        h2 = self.gas.species(self.gas.species_index("H2"))
        nasa7 = NASA7Polynomial.from_ct(h2, 200, 3500, 1000)
        quality = nasa7.fit_quality(self.data)
        # Perturb the coefficients to make this quality dict go off
        nasa7.coeffs_low *= 1.121212
        with self.assertRaises(ValueError):
            nasa7.fit_quality(self.data, error=True)

    def test_correct_coeffs_integration_constants(self):
        """Test that _correct_coeffs properly sets integration constants."""
        # Use fit_cp_only which sets integration constants internally
        nasa7 = NASA7Polynomial.fit_cp_only(
            self.data, self.Tmin, self.Tmax, self.Tcommon
        )

        # Check that h and s are correct at 298.15K
        h_fit = nasa7.h_over_RT(np.array([T_STD]))[0] * self.R * T_STD
        s_fit = nasa7.s_over_R(np.array([T_STD]))[0] * self.R

        # Check that fit maintains thermodynamic integration constants
        # For H2, dhf_over_R is ~2.5e-9 (nearly zero), so use absolute tolerance
        print(h_fit)
        self.assertAlmostEqual(h_fit, self.data.dhf, places=23)
        self.assertAlmostEqual(s_fit, self.data.s0, places=23)


        nasa7 = NASA7Polynomial.fit_full(
            self.data, self.Tmin, self.Tmax, self.Tcommon
        )

        # Check that h and s are correct at 298.15K
        h_fit = nasa7.h_over_RT(np.array([T_STD]))[0] * self.R * T_STD
        s_fit = nasa7.s_over_R(np.array([T_STD]))[0] * self.R

        # Check that fit maintains thermodynamic integration constants
        # For H2, dhf_over_R is ~2.5e-9 (nearly zero)
        self.assertAlmostEqual(h_fit, self.data.dhf, places=8)
        self.assertAlmostEqual(s_fit, self.data.s0, places=23)




class TestSutherland(unittest.TestCase):
    """Test Sutherland transport model."""

    def test_sutherland_mu_formula(self):
        """Sutherland mu matches expected formula."""
        suth = Sutherland(As=1.5e-6, Ts=100.0)
        T = 500.0
        expected = 1.5e-6 * np.sqrt(T) / (1.0 + 100.0 / T)
        self.assertAlmostEqual(suth.mu(T), expected, places=12)

    def test_sutherland_array_input(self):
        """Sutherland works with array input."""
        suth = Sutherland(As=1.5e-6, Ts=100.0)
        T = np.array([300, 500, 1000])
        result = suth.mu(T)
        self.assertEqual(result.shape, T.shape)

    def test_euken0(self):
        """Reference data retrieved from OpenFOAM"""
        suth = Sutherland(As=1, Ts=1)
        cv_mole = -R_OF + 4
        kappa_foam = 936.697882481863
        kappa = suth.kappa_euken(1, cv_mole, 2, R_OF)
        self.assertTrue(np.abs(kappa - kappa_foam) < 1e-12)

    def test_sutherland1(self):
        """Reference data for H2O from OpenFOAM"""
        suth = Sutherland(As=1.67212e-06, Ts=170.672)
        mu = suth.mu(400)
        mu_foam = 2.34407155073317e-05
        self.assertTrue(np.abs(mu - mu_foam) < 1e-12)

    def test_euken1(self):
        """Reference data for H2O from OpenFOAM"""
        cv_mole = 26123.236960773
        kappa_foam = 0.0640159441308283
        suth = Sutherland(As=1.67212e-06, Ts=170.672)
        kappa = suth.kappa_euken(400, cv_mole, 18.0153, R_OF)
        self.assertTrue(np.abs(kappa - kappa_foam) < 1e-12)

    def test_sutherland_list(self):
        """Reference data for H2O from OpenFOAM"""
        suth = Sutherland(As=1.67212e-06, Ts=170.672)
        mu = suth.mu(np.array([400,400,400]))
        mu_foam = 2.34407155073317e-05
        self.assertTrue(np.linalg.norm(mu - [mu_foam,mu_foam,mu_foam]) < 1e-12)

class TestTransportFitting(unittest.TestCase):
    """Test Sutherland transport fitting."""

    def setUp(self):
        """Set up viscosity data from H2."""
        self.gas = ct.Solution("h2o2.yaml")
        self.gas.transport_model = "multicomponent"

        self.T = np.linspace(300, 3000, 128)
        nT = len(self.T)
        self.mu = np.zeros(nT)
        self.kappa = np.zeros(nT)
        self.cv = np.zeros(nT)

        for i, T_i in enumerate(self.T):
            self.gas.TPX = T_i, ct.one_atm, "H2:1.0"
            self.mu[i] = self.gas.viscosity
            self.kappa[i] = self.gas.thermal_conductivity
            self.cv[i] = self.gas.cv_mole

        self.W = self.gas.molecular_weights[self.gas.species_index("H2")]

    def test_sutherland_fit(self):
        """Test Sutherland.fit() method."""
        sutherland = Sutherland.fit(self.T, self.mu)

        mu_fit = sutherland.mu(self.T)
        err = np.linalg.norm(self.mu - mu_fit) / np.linalg.norm(self.mu)
        assert err < 0.02

        kappa_fit = sutherland.kappa_euken(self.T, self.cv, self.W, ct.gas_constant)
        err = np.linalg.norm(self.kappa - kappa_fit) / np.linalg.norm(self.kappa)
        assert err < 0.02

    def test_polynomial_fit(self):
        poly = Polynomial.fit_polynomial(self.T, self.mu, self.kappa)

        mu_fit=poly.mu(self.T)

        err = np.linalg.norm(self.mu - mu_fit) / np.linalg.norm(self.mu)
        assert err < 5e-3

        kappa_fit = poly.kappa(self.T)
        err = np.linalg.norm(self.kappa - kappa_fit) / np.linalg.norm(self.kappa)
        assert err < 5e-3

    def test_log_polynomial_fit(self):
        poly = Polynomial.fit_log_polynomial(self.T, self.mu, self.kappa)

        mu_fit=poly.mu(self.T)

        err = np.linalg.norm(self.mu - mu_fit) / np.linalg.norm(self.mu)
        assert err < 1e-4

        kappa_fit = poly.kappa(self.T)
        err = np.linalg.norm(self.kappa - kappa_fit) / np.linalg.norm(self.kappa)
        assert err < 5e-3


class TestPolynomial(unittest.TestCase):
     """Test Polynomial transport class."""

     def test_polynomial_mu_evaluation(self):
         """Standard polynomial mu evaluation matches np.poly1d."""
         coeffs = np.array([1e-12, -2e-9, 1e-6, 5e-5])
         poly = Polynomial(coeffs, coeffs, poly_type="polynomial")
         T = 1000.0
         expected = np.poly1d(coeffs)(T)
         self.assertAlmostEqual(float(poly.mu(T)), expected, places=12)

     def test_polynomial_kappa_evaluation(self):
         """Standard polynomial kappa evaluation matches np.poly1d."""
         coeffs_mu = np.array([1e-12, 0, 1e-6, 0])
         coeffs_k = np.array([2e-12, 0, 2e-6, 0])
         poly = Polynomial(coeffs_mu, coeffs_k, poly_type="polynomial")
         T = 500.0
         expected = np.poly1d(coeffs_k)(T)
         self.assertAlmostEqual(float(poly.kappa(T)), expected, places=12)

     def test_log_polynomial_mu_evaluation(self):
         """Log-polynomial mu = exp(P(log(T)))."""
         coeffs = np.array([0.5, -1.0, 2.0, -10.0])
         poly = Polynomial(coeffs, coeffs, poly_type="log_polynomial")
         T = 1000.0
         expected = np.exp(np.poly1d(coeffs)(np.log(T)))
         self.assertAlmostEqual(float(poly.mu(T)), expected, places=10)

     def test_log_polynomial_kappa_evaluation(self):
         """Log-polynomial kappa = exp(P(log(T)))."""
         coeffs = np.array([0.3, -0.5, 1.5, -8.0])
         poly = Polynomial(coeffs, coeffs, poly_type="log_polynomial")
         T = 750.0
         expected = np.exp(np.poly1d(coeffs)(np.log(T)))
         self.assertAlmostEqual(float(poly.kappa(T)), expected, places=10)

     def test_invalid_poly_type_raises(self):
         """Invalid poly_type should raise ValueError."""
         with self.assertRaises(ValueError):
             Polynomial(np.zeros(4), np.zeros(4), poly_type="invalid")

     def test_polynomial_array_input(self):
         """Polynomial evaluation works with array inputs."""
         coeffs = np.array([1e-12, -2e-9, 1e-6, 5e-5])
         poly = Polynomial(coeffs, coeffs, poly_type="polynomial")
         T = np.array([300.0, 500.0, 1000.0])
         result = poly.mu(T)
         self.assertEqual(result.shape, T.shape)

     def test_poly0(self):
         """Reference data for H2O retrieved from OpenFOAM"""
         T = 400
         poly_coeffs_mu = np.flip(np.array([1000, -0.05, 0.003, 0]))
         poly_coeffs_kappa = np.flip(np.array([2000, -0.15, 0.023, 0]))
         mu_foam = 1460.0
         kappa_foam = 5620.0
         poly = Polynomial(poly_coeffs_mu, poly_coeffs_kappa, poly_type="polynomial")
         mu = poly.mu(T)
         kappa = poly.kappa(T)
         self.assertTrue(np.abs(mu - mu_foam) / np.abs(mu_foam) < 1e-12)
         self.assertTrue(np.abs(kappa - kappa_foam) / np.abs(kappa_foam) < 1e-12)

     def test_logpoly0(self):
         """Reference data for H2O retrieved from OpenFOAM"""
         T = 400
         poly_coeffs_mu = np.flip(np.array([0.1, 0.1, 0.1, 0]))
         poly_coeffs_kappa = np.flip(np.array([0.1, 0.1, 0.1, 0]))
         mu_foam = 72.8870655981874
         kappa_foam = 72.8870655981874
         poly = Polynomial(poly_coeffs_mu, poly_coeffs_kappa, poly_type="log_polynomial")
         mu = poly.mu(T)
         kappa = poly.kappa(T)
         self.assertTrue(np.abs(mu - mu_foam) / np.abs(mu_foam) < 1e-12)
         self.assertTrue(np.abs(kappa - kappa_foam) / np.abs(kappa_foam) < 1e-12)



if __name__ == "__main__":
    unittest.main()
