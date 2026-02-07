import unittest
import numpy as np
from pathlib import Path

from ct2foam.thermo_transport import thermo_fitter as th_fitter
from ct2foam.thermo_transport import ct_properties

R = 8314.47006650545
T_STD = 298.15
EPS = 1e-12
TOL_CP = 0.001
TOL_H = 0.005
TOL_S = 0.005
TOL_CONTINUITY = 1e-6

# H2O NASA7 coefficients
H2O_TMID = 1000.0
H2O_C_LO = np.array([3.38684, 0.00347498, -6.3547e-06, 6.96858e-09, -2.50659e-12])
H2O_C_HI = np.array([2.67215, 0.00305629, -8.73026e-07, 1.201e-10, -6.39162e-15])
H2O_C_LO_FULL = np.array(
    [3.38684, 0.00347498, -6.3547e-06, 6.96858e-09, -2.50659e-12, -30208.1, 2.59023]
)
H2O_C_HI_FULL = np.array(
    [2.67215, 0.00305629, -8.73026e-07, 1.201e-10, -6.39162e-15, -29899.2, 6.86282]
)

test_data_dir = Path(Path(__file__).parent.parent, "test_data")


class TestThermoFitterEvaluation(unittest.TestCase):

    def test_cp_nasa7_scalar_below_midpoint(self):
        """Test cp_nasa7 evaluates low-temperature region correctly."""
        T = 400.0
        cp_over_R = th_fitter.cp_nasa7(T, H2O_TMID, H2O_C_LO, H2O_C_HI)
        # Should use low-temperature coefficients
        expected = th_fitter._cp_nasa7(H2O_C_LO, T)
        self.assertAlmostEqual(cp_over_R, expected, places=10)

    def test_cp_nasa7_scalar_above_midpoint(self):
        """Test cp_nasa7 evaluates high-temperature region correctly."""
        T = 1500.0
        cp_over_R = th_fitter.cp_nasa7(T, H2O_TMID, H2O_C_LO, H2O_C_HI)
        # Should use high-temperature coefficients
        expected = th_fitter._cp_nasa7(H2O_C_HI, T)
        self.assertAlmostEqual(cp_over_R, expected, places=10)

    def test_cp_nasa7_at_midpoint_boundary(self):
        """Test cp_nasa7 at exactly midpoint temperature."""
        T = H2O_TMID
        cp_over_R = th_fitter.cp_nasa7(T, H2O_TMID, H2O_C_LO, H2O_C_HI)
        # T <= Tmid condition means low coefficients at boundary
        expected = th_fitter._cp_nasa7(H2O_C_LO, T)
        self.assertAlmostEqual(cp_over_R, expected, places=10)

    def test_cp_nasa7_array_input_vectorization(self):
        """Test cp_nasa7 with array input segmentation."""
        T = np.array([200.0, 400.0, 1000.0, 1500.0, 3000.0])
        cp_over_R = th_fitter.cp_nasa7(T, H2O_TMID, H2O_C_LO, H2O_C_HI)

        self.assertEqual(cp_over_R.shape, T.shape)

        for i in [0, 1, 2]:
            expected = th_fitter._cp_nasa7(H2O_C_LO, T[i])
            self.assertAlmostEqual(cp_over_R[i], expected, places=10)

        for i in [3, 4]:
            expected = th_fitter._cp_nasa7(H2O_C_HI, T[i])
            self.assertAlmostEqual(cp_over_R[i], expected, places=10)

    def test_s_nasa7_monotonicity_with_temperature(self):
        """Test entropy increases monotonically with temperature."""
        T = np.linspace(300, 3000, 50)
        s_over_R = th_fitter.s_nasa7(T, H2O_TMID, H2O_C_LO_FULL, H2O_C_HI_FULL)

        # Entropy should increase with T
        diffs = np.diff(s_over_R)
        self.assertTrue(np.all(diffs > -1e-10))

    def test_dcpdT_nasa7_physically_reasonable(self):
        """Test dcp/dT magnitude is physically sensible."""
        T = 1000.0
        dcpdT = th_fitter.dcpdT_nasa7(T, H2O_TMID, H2O_C_LO, H2O_C_HI)

        self.assertTrue(np.isfinite(dcpdT))
        self.assertAlmostEqual(np.abs(dcpdT[0]), 0.00164496)


class TestThermoFitterFitting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.h2o2 = ct_properties.ctThermoTransport("h2o2.yaml", verbose=False)
        cls.h2o2.evaluate_properties()

    def test_fit_nasapolys_cp_synthetic_data(self):
        """Test cp fitting with synthetic quadratic data."""
        T0 = np.linspace(300, 2000, 50)
        Tc_i = 25
        cp_over_R = 2.5 + 0.001 * T0 + 1e-5 * T0**2
        cp0_over_R = 2.8
        dhf_over_R = 10.0
        s0_over_R = 100.0

        c_lo, c_hi = th_fitter.fit_nasapolys_cp(
            T0, Tc_i, cp_over_R, cp0_over_R, dhf_over_R, s0_over_R
        )

        self.assertEqual(len(c_lo), 7)
        self.assertEqual(len(c_hi), 7)

        _cp = th_fitter.cp_nasa7(T0, T0[Tc_i], c_lo, c_hi)
        eps = np.linalg.norm(cp_over_R - _cp) / np.linalg.norm(cp_over_R)
        self.assertLessEqual(eps, 1e-2)

    def test_cp_fit_continuity(self):
        """Test cp fitting with real Cantera H2O data."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_cp(
            T,
            Tc_i,
            cp_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )
        # See default tolerances
        continuous = th_fitter.continuity(self.h2o2.Tmid, c_lo, c_hi)
        self.assertTrue(continuous)

    def test_fit_nasapolys_full_h_s_coupled(self):
        """Test full fitting couples enthalpy and entropy correctly."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = int(np.argmin(np.abs(T - self.h2o2.Tmid)))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        consistent = th_fitter.consistency(
            T,
            self.h2o2.Tmid,
            c_lo,
            c_hi,
            cp_over_R,
            h_over_RT,
            s_over_R,
            abs_tol=0.1,
        )
        self.assertTrue(consistent)

    def test_fit_nasapolys_full_convergence(self):
        """Test full fitting completes without hanging."""
        sp_i = self.h2o2.gas.species_index("H2")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        self.assertIsNotNone(c_lo)
        self.assertIsNotNone(c_hi)

    def test_fit_nasapolys_cp_continuity_at_midpoint(self):
        """Test C0 continuity of cp at midpoint."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_cp(
            T,
            Tc_i,
            cp_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        cp_lo = th_fitter._cp_nasa7(c_lo, self.h2o2.Tmid)
        cp_hi = th_fitter._cp_nasa7(c_hi, self.h2o2.Tmid)
        self.assertLess(np.abs(cp_lo - cp_hi), TOL_CONTINUITY)

    def test_fit_nasapolys_high_temperature_accuracy(self):
        """Test high-temperature region fit accuracy."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        T_high = T[T > 1500]
        cp_ref = cp_over_R[T > 1500]
        cp_fit = th_fitter.cp_nasa7(T_high, self.h2o2.Tmid, c_lo, c_hi)

        rel_err = np.abs((cp_fit - cp_ref) / (cp_ref + 1e-10))
        self.assertLess(np.mean(rel_err), 0.001)

    def test_fit_nasapolys_low_temperature_accuracy(self):
        """Test low-temperature region fit accuracy."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        T_low = T[T < 500]
        cp_ref = cp_over_R[T < 500]
        cp_fit = th_fitter.cp_nasa7(T_low, self.h2o2.Tmid, c_lo, c_hi)

        rel_err = np.abs((cp_fit - cp_ref) / (cp_ref + 1e-10))
        self.assertLess(np.mean(rel_err), 0.001)


class TestThermoFitterQuality(unittest.TestCase):
    """Test NASA7 quality checking functions."""

    @classmethod
    def setUpClass(cls):
        cls.h2o2 = ct_properties.ctThermoTransport("h2o2.yaml", verbose=False)
        cls.h2o2.evaluate_properties()

    def test_consistency_accepts_good_fit(self):
        """Test consistency() returns True for accurate fits."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        consistent = th_fitter.consistency(
            T,
            self.h2o2.Tmid,
            c_lo,
            c_hi,
            cp_over_R,
            h_over_RT,
            s_over_R,
            abs_tol=0.1,
        )
        self.assertTrue(consistent)

    def test_consistency_rejects_bad_fit(self):
        """Test consistency() returns False for perturbed coefficients."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        # Perturb coefficients
        c_lo_bad = c_lo.copy()
        c_lo_bad[0] += 1.0

        consistent = th_fitter.consistency(
            T,
            self.h2o2.Tmid,
            c_lo_bad,
            c_hi,
            cp_over_R,
            h_over_RT,
            s_over_R,
            abs_tol=0.1,  # Same as in the succesfull test
        )
        self.assertFalse(consistent)

    def test_continuity_accepts_smooth_transition(self):
        """Test continuity() returns True for properly fitted coefficients."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp_over_R = self.h2o2.cp[sp_i, :] / R
        h_over_RT = self.h2o2.h[sp_i, :] / (R * T)
        s_over_R = self.h2o2.s[sp_i, :] / R

        Tc_i = np.argmin(np.abs(T - self.h2o2.Tmid))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp_over_R,
            h_over_RT,
            s_over_R,
            self.h2o2.cp0_over_R[sp_i],
            self.h2o2.dhf_over_R[sp_i],
            self.h2o2.s0_over_R[sp_i],
        )

        continuous = th_fitter.continuity(self.h2o2.Tmid, c_lo, c_hi)
        self.assertTrue(continuous)

    def test_continuity_rejects_discontinuous_cp(self):
        """Test continuity() detects jump in cp at Tmid."""
        c_lo = H2O_C_LO_FULL.copy()
        c_hi = H2O_C_HI_FULL.copy()

        # Perturb to create discontinuity
        c_hi[0] += 0.1

        continuous = th_fitter.continuity(H2O_TMID, c_lo, c_hi, cp_tol=1e-6)
        self.assertFalse(continuous)

    def test_continuity_rejects_discontinuous_derivative(self):
        """Test continuity() detects discontinuous dcp/dT at Tmid."""
        c_lo = H2O_C_LO_FULL.copy()
        c_hi = H2O_C_HI_FULL.copy()

        # Perturb derivative coefficient
        c_hi[1] += 0.1

        continuous = th_fitter.continuity(H2O_TMID, c_lo, c_hi, cpdT_tol=1e-6)
        self.assertFalse(continuous)

    def test_error_nasa7_magnitude_scales_with_deviation(self):
        """Test error_nasa7() scales with fit quality."""
        sp_i = self.h2o2.gas.species_index("H2O")
        T = self.h2o2.T
        cp = self.h2o2.cp[sp_i, :]
        h = self.h2o2.h[sp_i, :]
        s = self.h2o2.s[sp_i, :]

        Tc_i = int(np.argmin(np.abs(T - self.h2o2.Tmid)))

        c_lo, c_hi = th_fitter.fit_nasapolys_full(
            T,
            Tc_i,
            cp / R,
            h / (R * T),
            s / R,
            float(self.h2o2.cp0_over_R[sp_i]),
            float(self.h2o2.dhf_over_R[sp_i]),
            float(self.h2o2.s0_over_R[sp_i]),
        )

        err_good = th_fitter.error_nasa7(T, cp, h, s, self.h2o2.Tmid, c_lo, c_hi, R)

        c_lo_bad = c_lo.copy()
        c_lo_bad[0] += 0.5
        err_bad = th_fitter.error_nasa7(T, cp, h, s, self.h2o2.Tmid, c_lo_bad, c_hi, R)

        self.assertLess(err_good, err_bad)


if __name__ == "__main__":
    unittest.main()
