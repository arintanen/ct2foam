"""Equivalence test comparing old and new implementations on GRI-3.0 mechanism."""

import unittest
import numpy as np
import cantera as ct

from ct2foam.thermo_transport.ct_properties import ctThermoTransport
from ct2foam.thermo_transport import ct2foam_utils as old_utils
from ct2foam.v2.mechanism_dataset import MechanismDataset

# Test parameters
MECHANISM = "gri30.yaml"
TMID = 1000.0
TLOW = 300.0
THIGH = 3000.0
T_EVAL = np.linspace(300, 3000, 128)

# Tolerance for numerical equivalence
ABS_TOL = 1e-10  # Absolute tolerance for coefficient comparison
REL_TOL = 1e-8  # Relative tolerance for array comparison


class TestV2OldEquivalence(unittest.TestCase):
    """Compare old and new implementations on GRI-3.0 mechanism species-by-species."""

    @classmethod
    def setUpClass(cls):
        """Load mechanism with both old and new implementations."""
        print(f"\nLoading {MECHANISM} with old implementation...")
        cls.old_data = ctThermoTransport(
            MECHANISM, T=T_EVAL.copy(), Tmid=TMID, verbose=False
        )
        cls.old_data.evaluate_properties()

        print(f"Fitting thermo and transport with old implementation...")
        cls.old_nasa_lo, cls.old_nasa_hi = old_utils.refit_ct_thermo(
            cls.old_data, TMID, output_dir="/tmp"
        )
        cls.old_transport = old_utils.fit_ct_transport(cls.old_data, poly_order=3)
        (
            cls.old_As,
            cls.old_Ts,
            cls.old_std,
            cls.old_poly_mu,
            cls.old_poly_kappa,
            cls.old_logpoly_mu,
            cls.old_logpoly_kappa,
        ) = cls.old_transport

        print(f"Loading {MECHANISM} with new implementation...")
        cls.new_mech = MechanismDataset.from_cantera(
            MECHANISM, Tmid=TMID, Tlow=TLOW, Thigh=THIGH, T_eval=T_EVAL.copy()
        )

        print("Fitting thermo and transport with new implementation...")
        result = cls.new_mech.fit_all(verbose=False)
        cls.fit_result = result

        print(f"Old: {cls.old_data.gas.n_species} species")
        print(f"New: {len(cls.new_mech.species_datasets)} species")
        print(
            f"New fit result: {result['succeeded']} succeeded, {result['failed']} failed"
        )

    def _get_species_by_name(self, name):
        """Get new implementation SpeciesDataset by name."""
        for sd in self.new_mech.species_datasets:
            if sd.name == name:
                return sd
        return None

    def test_species_count_match(self):
        """Both implementations should have the same number of species."""
        self.assertEqual(
            self.old_data.gas.n_species, len(self.new_mech.species_datasets)
        )

    def test_species_names_match(self):
        """Species names should match between implementations."""
        old_names = set(self.old_data.names)
        new_names = set([sd.name for sd in self.new_mech.species_datasets])
        self.assertEqual(old_names, new_names)

    def test_all_species_fitted_successfully(self):
        """All species should fit successfully in new implementation."""
        self.assertEqual(
            self.fit_result["failed"],
            0,
            f"New implementation failed on {self.fit_result['failed']} species",
        )

    def test_nasa7_coefficients_equivalence(self):
        """NASA7 coefficients should match between old and new for all species.

        NOTE: The old implementation reuses original Cantera coefficients when they
        are continuous and have matching Tmid. The new implementation always refits.
        Since GRI-3.0 has Tmid=1000.0 matching our choice, old implementation reuses
        those coefficients. This test checks that both produce good fits, not identical
        coefficients.
        """
        failures = []

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)
            self.assertIsNotNone(
                sd_new, f"Species {sp_name} not found in new implementation"
            )
            self.assertIsNotNone(
                sd_new.nasa7,
                f"Species {sp_name} has no nasa7 fit in new implementation",
            )

            # Instead of comparing coefficients directly, compare fit quality
            # by evaluating at test points
            T_test = np.linspace(400, 2500, 50)

            # Old implementation evaluation
            from ct2foam.thermo_transport import thermo_fitter as th_fitter

            old_cp = th_fitter.cp_nasa7(
                T_test, TMID, self.old_nasa_lo[i, :], self.old_nasa_hi[i, :]
            )
            old_h = th_fitter.h_nasa7(
                T_test, TMID, self.old_nasa_lo[i, :], self.old_nasa_hi[i, :]
            )
            old_s = th_fitter.s_nasa7(
                T_test, TMID, self.old_nasa_lo[i, :], self.old_nasa_hi[i, :]
            )

            # New implementation evaluation
            new_cp = sd_new.nasa7.cp_over_R(T_test)
            new_h = sd_new.nasa7.h_over_RT(T_test)
            new_s = sd_new.nasa7.s_over_R(T_test)

            print(old_cp)
            print(new_cp)
            print(sd_new.cp / ct.gas_constant)

            # Check that both produce similar results (within 10% tolerance for h)
            cp_close = np.allclose(old_cp, new_cp, atol=1e-13, rtol=1e-15)
            h_close = np.allclose(old_h, new_h, atol=1e-3, rtol=0.10)
            s_close = np.allclose(old_s, new_s, atol=1e-4, rtol=0.05)

            if not (cp_close and h_close and s_close):
                cp_err = np.max(np.abs(old_cp - new_cp) / np.abs(old_cp + 1e-10))
                h_err = np.max(np.abs(old_h - new_h) / np.abs(old_h + 1e-10))
                s_err = np.max(np.abs(old_s - new_s) / np.abs(old_s + 1e-10))
                failures.append(
                    f"{sp_name}: cp_rel={cp_err:.2e}, h_rel={h_err:.2e}, s_rel={s_err:.2e}"
                )

        if failures:
            self.fail(
                f"NASA7 fit quality mismatches (cp/s >5%, h >10% rel error):\n"
                + "\n".join(failures[:10])
            )

    def test_sutherland_coefficients_equivalence(self):
        """Sutherland As and Ts should match between old and new for all species."""
        failures = []

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)
            self.assertIsNotNone(
                sd_new.sutherland, f"Species {sp_name} has no sutherland fit"
            )

            old_As = self.old_As[i]
            old_Ts = self.old_Ts[i]
            new_As = sd_new.sutherland.As
            new_Ts = sd_new.sutherland.Ts

            As_close = np.isclose(old_As, new_As, atol=ABS_TOL, rtol=REL_TOL)
            Ts_close = np.isclose(old_Ts, new_Ts, atol=ABS_TOL, rtol=REL_TOL)

            if not (As_close and Ts_close):
                As_diff = abs(old_As - new_As)
                Ts_diff = abs(old_Ts - new_Ts)
                failures.append(
                    f"{sp_name}: As_diff={As_diff:.2e}, Ts_diff={Ts_diff:.2e}"
                )

        if failures:
            self.fail(
                f"Sutherland coefficient mismatches:\n" + "\n".join(failures[:10])
            )

    def test_polynomial_transport_coefficients_equivalence(self):
        """Polynomial transport coefficients should match between old and new."""
        failures = []

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)
            self.assertIsNotNone(
                sd_new.polynomial, f"Species {sp_name} has no polynomial fit"
            )

            old_poly_mu = self.old_poly_mu[i, :]
            old_poly_kappa = self.old_poly_kappa[i, :]
            new_poly_mu = sd_new.polynomial.coeffs_mu
            new_poly_kappa = sd_new.polynomial.coeffs_kappa

            mu_close = np.allclose(old_poly_mu, new_poly_mu, atol=ABS_TOL, rtol=REL_TOL)
            kappa_close = np.allclose(
                old_poly_kappa, new_poly_kappa, atol=ABS_TOL, rtol=REL_TOL
            )

            if not (mu_close and kappa_close):
                mu_diff = np.max(np.abs(old_poly_mu - new_poly_mu))
                kappa_diff = np.max(np.abs(old_poly_kappa - new_poly_kappa))
                failures.append(
                    f"{sp_name}: mu_diff={mu_diff:.2e}, kappa_diff={kappa_diff:.2e}"
                )

        if failures:
            self.fail(
                f"Polynomial transport coefficient mismatches:\n"
                + "\n".join(failures[:10])
            )

    def test_log_polynomial_transport_coefficients_equivalence(self):
        """Log-polynomial transport coefficients should match between old and new."""
        failures = []

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)
            self.assertIsNotNone(
                sd_new.log_polynomial, f"Species {sp_name} has no log_polynomial fit"
            )

            old_logpoly_mu = self.old_logpoly_mu[i, :]
            old_logpoly_kappa = self.old_logpoly_kappa[i, :]
            new_logpoly_mu = sd_new.log_polynomial.coeffs_mu
            new_logpoly_kappa = sd_new.log_polynomial.coeffs_kappa

            mu_close = np.allclose(
                old_logpoly_mu, new_logpoly_mu, atol=ABS_TOL, rtol=REL_TOL
            )
            kappa_close = np.allclose(
                old_logpoly_kappa, new_logpoly_kappa, atol=ABS_TOL, rtol=REL_TOL
            )

            if not (mu_close and kappa_close):
                mu_diff = np.max(np.abs(old_logpoly_mu - new_logpoly_mu))
                kappa_diff = np.max(np.abs(old_logpoly_kappa - new_logpoly_kappa))
                failures.append(
                    f"{sp_name}: mu_diff={mu_diff:.2e}, kappa_diff={kappa_diff:.2e}"
                )

        if failures:
            self.fail(
                f"Log-polynomial transport coefficient mismatches:\n"
                + "\n".join(failures[:10])
            )

    def test_cp_evaluation_equivalence(self):
        """Evaluate cp with both implementations and compare results.

        Note: Since old implementation may reuse Cantera coefficients while new
        always refits, we use 5% relative tolerance.
        """
        failures = []
        T_test = np.linspace(400, 2500, 50)

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)

            # Old implementation evaluation
            from ct2foam.thermo_transport import thermo_fitter as th_fitter

            old_cp = th_fitter.cp_nasa7(
                T_test, TMID, self.old_nasa_lo[i, :], self.old_nasa_hi[i, :]
            )

            # New implementation evaluation
            new_cp = sd_new.nasa7.cp_over_R(T_test)

            if not np.allclose(old_cp, new_cp, atol=1e-3, rtol=0.05):
                max_diff = np.max(np.abs(old_cp - new_cp))
                rel_err = np.max(np.abs(old_cp - new_cp) / np.abs(old_cp + 1e-10))
                failures.append(
                    f"{sp_name}: cp_max_diff={max_diff:.2e}, rel_err={rel_err:.2e}"
                )

        if failures:
            self.fail(
                f"Cp evaluation mismatches (>5% rel error):\n"
                + "\n".join(failures[:10])
            )

    def test_h_evaluation_equivalence(self):
        """Evaluate h/(RT) with both implementations and compare results."""
        failures = []
        T_test = np.linspace(400, 2500, 50)

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)

            # Old implementation evaluation
            from ct2foam.thermo_transport import thermo_fitter as th_fitter

            old_h = th_fitter.h_nasa7(
                T_test, TMID, self.old_nasa_lo[i, :], self.old_nasa_hi[i, :]
            )

            # New implementation evaluation
            new_h = sd_new.nasa7.h_over_RT(T_test)

            if not np.allclose(old_h, new_h, atol=1e-3, rtol=0.10):
                max_diff = np.max(np.abs(old_h - new_h))
                rel_err = np.max(np.abs(old_h - new_h) / np.abs(old_h + 1e-10))
                failures.append(
                    f"{sp_name}: h_max_diff={max_diff:.2e}, rel_err={rel_err:.2e}"
                )

        if failures:
            self.fail(
                f"Enthalpy evaluation mismatches (>10% rel error):\n"
                + "\n".join(failures[:10])
            )

    def test_s_evaluation_equivalence(self):
        """Evaluate s/R with both implementations and compare results."""
        failures = []
        T_test = np.linspace(400, 2500, 50)

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)

            # Old implementation evaluation
            from ct2foam.thermo_transport import thermo_fitter as th_fitter

            old_s = th_fitter.s_nasa7(
                T_test, TMID, self.old_nasa_lo[i, :], self.old_nasa_hi[i, :]
            )

            # New implementation evaluation
            new_s = sd_new.nasa7.s_over_R(T_test)

            if not np.allclose(old_s, new_s, atol=1e-4, rtol=0.05):
                max_diff = np.max(np.abs(old_s - new_s))
                rel_err = np.max(np.abs(old_s - new_s) / np.abs(old_s + 1e-10))
                failures.append(
                    f"{sp_name}: s_max_diff={max_diff:.2e}, rel_err={rel_err:.2e}"
                )

        if failures:
            self.fail(
                f"Entropy evaluation mismatches (>5% rel error):\n"
                + "\n".join(failures[:10])
            )

    def test_viscosity_evaluation_equivalence(self):
        """Evaluate viscosity with Sutherland formula and compare."""
        failures = []
        T_test = np.array([500.0, 1000.0, 1500.0, 2000.0])

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)

            # Old implementation evaluation
            from ct2foam.thermo_transport import transport_fitter as tr_fitter

            old_mu = tr_fitter.sutherland(T_test, self.old_As[i], self.old_Ts[i])

            # New implementation evaluation
            new_mu = sd_new.sutherland.mu(T_test)

            if not np.allclose(old_mu, new_mu, atol=ABS_TOL, rtol=REL_TOL):
                max_diff = np.max(np.abs(old_mu - new_mu))
                failures.append(f"{sp_name}: mu_max_diff={max_diff:.2e}")

        if failures:
            self.fail(f"Viscosity evaluation mismatches:\n" + "\n".join(failures[:10]))

    def test_polynomial_mu_evaluation_equivalence(self):
        """Evaluate polynomial viscosity and compare."""
        failures = []
        T_test = np.array([500.0, 1000.0, 1500.0, 2000.0])

        for i, sp_name in enumerate(self.old_data.names):
            sd_new = self._get_species_by_name(sp_name)

            # Old implementation evaluation
            from ct2foam.thermo_transport import transport_fitter as tr_fitter

            old_mu, _ = tr_fitter.eval_polynomial(
                self.old_poly_mu[i, :], self.old_poly_kappa[i, :], T_test
            )

            # New implementation evaluation
            new_mu = sd_new.polynomial.mu(T_test)

            if not np.allclose(old_mu, new_mu, atol=ABS_TOL, rtol=REL_TOL):
                max_diff = np.max(np.abs(old_mu - new_mu))
                failures.append(f"{sp_name}: poly_mu_max_diff={max_diff:.2e}")

        if failures:
            self.fail(
                f"Polynomial mu evaluation mismatches:\n" + "\n".join(failures[:10])
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
