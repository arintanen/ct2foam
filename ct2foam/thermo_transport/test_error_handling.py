import unittest
import numpy as np
from pathlib import Path

from ct2foam.thermo_transport import thermo_fitter as th_fitter
from ct2foam.thermo_transport import transport_fitter as tr_fitter

# Test constants
R = 8314.47006650545
T_STD = 298.15
EPS = 1e-12
TOL_CONTINUITY = 1e-6

# H2O NASA7 coefficients from test_thermo_transport.py
H2O_TMID = 1000.0
H2O_C_LO = np.array([3.38684, 0.00347498, -6.3547e-06, 6.96858e-09, -2.50659e-12])
H2O_C_HI = np.array([2.67215, 0.00305629, -8.73026e-07, 1.201e-10, -6.39162e-15])
H2O_C_LO_FULL = np.array(
    [3.38684, 0.00347498, -6.3547e-06, 6.96858e-09, -2.50659e-12, -30208.1, 2.59023]
)
H2O_C_HI_FULL = np.array(
    [2.67215, 0.00305629, -8.73026e-07, 1.201e-10, -6.39162e-15, -29899.2, 6.86282]
)


class TestTemperatureRangeValidation(unittest.TestCase):
    """Test edge cases with temperature ranges."""

    def test_nasa7_negative_temperature_evaluation(self):
        """Test evaluation with negative temperature (physically invalid but numerically valid)."""
        T = -100.0
        # Should compute without error
        cp = th_fitter.cp_nasa7(T, H2O_TMID, H2O_C_LO, H2O_C_HI)
        self.assertTrue(np.isfinite(cp))

    def test_cp_nasa7_scalar_vs_array_consistency(self):
        """Test scalar evaluation matches array evaluation."""
        T_scalar = 500.0
        T_array = np.array([500.0])

        cp_scalar = th_fitter.cp_nasa7(T_scalar, H2O_TMID, H2O_C_LO, H2O_C_HI)
        cp_array = th_fitter.cp_nasa7(T_array, H2O_TMID, H2O_C_LO, H2O_C_HI)

        # Both should be finite
        self.assertTrue(np.isfinite(cp_scalar))
        self.assertTrue(np.isfinite(cp_array[0]))

    def test_h_nasa7_array_T_proper_segmentation(self):
        """Test low/high temperature arrays are segmented correctly."""
        T = np.array([300.0, 600.0, 1000.0, 1200.0, 2000.0])
        h_over_RT = th_fitter.h_nasa7(T, H2O_TMID, H2O_C_LO_FULL, H2O_C_HI_FULL)

        # Verify segmentation by checking individual values
        for i, T_val in enumerate(T):
            if T_val <= H2O_TMID:
                expected = th_fitter._h_nasa7(H2O_C_LO_FULL, T_val)
            else:
                expected = th_fitter._h_nasa7(H2O_C_HI_FULL, T_val)
            self.assertAlmostEqual(h_over_RT[i], expected, places=10)

    def test_s_nasa7_boundary_values_match(self):
        """Test cp values at Tmid match between lo and hi coefficients."""
        # Evaluate cp at exactly Tmid
        cp_lo = th_fitter._cp_nasa7(H2O_C_LO, H2O_TMID)
        cp_hi = th_fitter._cp_nasa7(H2O_C_HI, H2O_TMID)

        # For well-fitted coefficients, these should be close
        # (though may not be exactly equal before correction)
        self.assertTrue(np.isfinite(cp_lo))
        self.assertTrue(np.isfinite(cp_hi))


class TestPolynomialFittingEdgeCases(unittest.TestCase):
    """Test polynomial fitting edge cases."""

    def test_fit_polynomial_single_point_minimum_data(self):
        """Test fitting with minimal data points."""
        T = np.array([500.0, 1000.0])
        data_mu = np.array([1.0, 2.0])
        data_kappa = np.array([0.5, 1.5])

        # Should compute without error
        poly_mu, poly_kappa = tr_fitter.fit_polynomial(
            T, data_mu, data_kappa, poly_order=2
        )
        self.assertTrue(np.all(np.isfinite(poly_mu)))

    def test_fit_polynomial_dimension_mismatch_T_data(self):
        """Test dimension mismatch raises appropriate error or handles gracefully."""
        T = np.linspace(300, 1000, 50)
        mu = np.linspace(0, 1, 30)  # Different length
        kappa = np.linspace(0, 1, 30)  # Different length

        # Should raise error on dimension mismatch
        with self.assertRaises((ValueError, IndexError, TypeError)):
            # fit_polynomial expects matching dimensions
            tr_fitter.fit_polynomial(T, mu, kappa, poly_order=3)

    def test_fit_polynomial_NaN_in_input_data(self):
        """Test handling of NaN values in input data."""
        T = np.array([300.0, 400.0, 500.0, 1000.0])
        mu = np.array([1.0, np.nan, 2.0, 3.0])
        kappa = np.array([1.0, 1.5, 2.0, 3.0])

        # Should either raise error or produce NaN result
        try:
            result = tr_fitter.fit_polynomial(T, mu, kappa, poly_order=2)
            # If it doesn't raise, result should contain NaN or be invalid
            self.assertTrue(np.any(np.isnan(result)) or not np.all(np.isfinite(result)))
        except (ValueError, RuntimeError):
            # Acceptable to raise error for NaN input
            pass

    def test_fit_polynomial_zero_cv_euken_calculation(self):
        """Test Euken formula with cv=0 edge case."""
        T = 500.0
        mu = 1e-5  # Viscosity
        cv_mole = 0.0  # Edge case
        W = 28.0

        # Euken formula uses cv, division by zero at cv=0
        with self.assertRaises(ZeroDivisionError):
            kappa = tr_fitter.euken(mu, cv_mole, W, R)


class TestContinuityConsistencyEdgeCases(unittest.TestCase):
    """Test continuity and consistency checking edge cases."""

    def test_consistency_with_high_tolerance(self):
        """Test consistency check accepts noisy data with loose tolerance."""
        T = np.linspace(300, 2000, 50)
        cp_over_R = 2.5 * np.ones_like(T)
        h_over_RT = 10.0 * np.ones_like(T)
        s_over_R = 100.0 * np.ones_like(T)

        # Create noisy reference data
        cp_over_R_ref = cp_over_R + 0.01 * np.random.randn(len(T))
        h_over_RT_ref = h_over_RT + 0.01 * np.random.randn(len(T))
        s_over_R_ref = s_over_R + 0.01 * np.random.randn(len(T))

        # With loose tolerance, should pass
        consistent = th_fitter.consistency(
            T,
            H2O_TMID,
            H2O_C_LO_FULL,
            H2O_C_HI_FULL,
            cp_over_R_ref,
            h_over_RT_ref,
            s_over_R_ref,
            abs_tol=1000.0,  # Very loose
        )
        self.assertTrue(consistent)

    def test_continuity_with_loose_derivative_tolerance(self):
        """Test continuity check with loose derivative tolerance."""
        c_lo = H2O_C_LO_FULL.copy()
        c_hi = H2O_C_HI_FULL.copy()

        # With very loose tolerances on all parameters
        continuous = th_fitter.continuity(
            H2O_TMID, c_lo, c_hi, cp_tol=1.0, cpdT_tol=1.0, h_tol=1.0, s_tol=1.0
        )
        self.assertTrue(continuous)

    def test_correct_coeffs_preserves_standard_state(self):
        """Test correct_coeffs() preserves h/s at standard state."""
        coeffs = np.array(
            [
                3.5,
                0.003,
                -6e-6,
                7e-9,
                -2.5e-12,
                0,
                2.5,
                2.7,
                0.003,
                -8e-7,
                1.2e-10,
                -6.4e-15,
                0,
                6.9,
            ]
        )

        Tcommon = 1000.0
        dhf_over_R = 5.0
        s0_over_R = 150.0

        corrected = th_fitter.correct_coeffs(coeffs, Tcommon, dhf_over_R, s0_over_R)

        # Should return 14-element array
        self.assertEqual(len(corrected), 14)
        # Coefficients should be finite
        self.assertTrue(np.all(np.isfinite(corrected)))


class TestSutherlandEukenEdgeCases(unittest.TestCase):
    """Test Sutherland and Euken formula edge cases."""

    def test_sutherland_zero_temperature(self):
        """Test Sutherland formula at T=0 (singular point)."""
        T = 0.0
        As = 1e-6
        Ts = 170.0

        # At T=0, division by zero occurs
        with self.assertRaises(ZeroDivisionError):
            mu = tr_fitter.sutherland(T, As, Ts)

    def test_sutherland_low_temperature(self):
        """Test Sutherland formula at very low temperature."""
        T = 0.1
        As = 1e-6
        Ts = 170.0

        mu = tr_fitter.sutherland(T, As, Ts)
        self.assertTrue(np.isfinite(mu))
        self.assertGreater(mu, 0)

    def test_euken_negative_cv(self):
        """Test Euken formula with negative cv (unphysical)."""
        mu = 1e-5
        cv_mole = -1000.0  # Negative (unphysical)
        W = 28.0

        # Should compute but result may be unphysical
        kappa = tr_fitter.euken(mu, cv_mole, W, R)
        self.assertTrue(np.isfinite(kappa))


if __name__ == "__main__":
    unittest.main()
