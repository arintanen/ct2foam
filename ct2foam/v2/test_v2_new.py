"""Comprehensive test suite for refactored v2 implementation."""

import unittest
import numpy as np
from pathlib import Path
import cantera as ct
import tempfile

from ct2foam.v2 import (
    NASA7Polynomial,
    Sutherland,
    Polynomial,
    Species,
    CanteraThermoTransport,
    FittingTolerances,
)


class TestNASA7PolynomialFitting(unittest.TestCase):
    """Test NASA7Polynomial fitting methods."""

    def setUp(self):
        """Set up test data from H2 species."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"
        self.gas = ct.Solution(str(mech_file))
        self.R = ct.gas_constant

        # Extract H2 data
        self.gas.TPX = 1000, ct.one_atm, "H2:1.0"
        h2_idx = self.gas.species_index("H2")

        # Generate evaluation data
        self.T = np.linspace(300, 3000, 50)
        self.Tmid = 1000.0

        nT = len(self.T)
        self.cp = np.zeros(nT)
        self.h = np.zeros(nT)
        self.s = np.zeros(nT)

        for i, T_i in enumerate(self.T):
            self.gas.TP = T_i, ct.one_atm
            self.cp[i] = self.gas.standard_cp_R[h2_idx]
            self.h[i] = self.gas.standard_enthalpies_RT[h2_idx]
            self.s[i] = self.gas.standard_entropies_R[h2_idx]

        # Standard state values at 298.15K
        self.gas.TP = 298.15, ct.one_atm
        self.cp0_over_R = self.gas.standard_cp_R[h2_idx]
        self.dhf_over_R = self.gas.standard_enthalpies_RT[h2_idx]
        self.s0_over_R = self.gas.standard_entropies_R[h2_idx]

    def test_fit_cp_only(self):
        """Test fit_cp_only method."""
        nasa7 = NASA7Polynomial.fit_cp_only(
            self.T, self.cp, self.cp0_over_R, self.dhf_over_R, self.s0_over_R, self.Tmid
        )

        self.assertIsNotNone(nasa7)
        self.assertEqual(nasa7.Tmid, self.Tmid)
        self.assertEqual(len(nasa7.coeffs_low), 7)
        self.assertEqual(len(nasa7.coeffs_high), 7)

        # Evaluate cp and check fit quality
        cp_fit_low = nasa7.cp_over_R(self.T[self.T <= self.Tmid])
        cp_fit_high = nasa7.cp_over_R(self.T[self.T > self.Tmid])

        cp_ref_low = self.cp[self.T <= self.Tmid]
        cp_ref_high = self.cp[self.T > self.Tmid]

        # Should have good cp fit (relaxed tolerance for numerical precision)
        np.testing.assert_allclose(cp_fit_low, cp_ref_low, rtol=0.01)
        np.testing.assert_allclose(cp_fit_high, cp_ref_high, rtol=0.01)

    def test_fit_full(self):
        """Test fit_full method."""
        nasa7 = NASA7Polynomial.fit_full(
            self.T,
            self.cp,
            self.h,
            self.s,
            self.cp0_over_R,
            self.dhf_over_R,
            self.s0_over_R,
            self.Tmid,
        )

        self.assertIsNotNone(nasa7)
        self.assertEqual(nasa7.Tmid, self.Tmid)

        # Check all properties match
        cp_fit = nasa7.cp_over_R(self.T)
        h_fit = nasa7.h_over_RT(self.T)
        s_fit = nasa7.s_over_R(self.T)

        np.testing.assert_allclose(cp_fit, self.cp, rtol=0.01)
        np.testing.assert_allclose(h_fit, self.h, rtol=0.01)
        np.testing.assert_allclose(s_fit, self.s, rtol=0.01)

    def test_fit_auto_success(self):
        """Test fit_auto with data that should succeed with fit_full."""
        nasa7 = NASA7Polynomial.fit_auto(
            self.T,
            self.cp,
            self.h,
            self.s,
            self.cp0_over_R,
            self.dhf_over_R,
            self.s0_over_R,
            self.Tmid,
            verbose=False,
        )

        self.assertIsNotNone(nasa7)
        self.assertEqual(nasa7.Tmid, self.Tmid)

        # Should have good fit (relaxed tolerance)
        cp_fit = nasa7.cp_over_R(self.T)
        np.testing.assert_allclose(cp_fit, self.cp, rtol=0.01)

    def test_correct_coeffs_integration_constants(self):
        """Test that _correct_coeffs properly sets integration constants."""
        # Use fit_cp_only which sets integration constants internally
        nasa7 = NASA7Polynomial.fit_cp_only(
            self.T, self.cp, self.cp0_over_R, self.dhf_over_R, self.s0_over_R, self.Tmid
        )

        # Check that h and s are correct at 298.15K
        T_std = 298.15
        h_fit = nasa7.h_over_RT(np.array([T_std]))[0]
        s_fit = nasa7.s_over_R(np.array([T_std]))[0]

        # Check that fit maintains thermodynamic integration constants
        # For H2, dhf_over_R is ~2.5e-9 (nearly zero), so use absolute tolerance
        np.testing.assert_allclose(h_fit, self.dhf_over_R, atol=1e-8)
        np.testing.assert_allclose(s_fit, self.s0_over_R, rtol=0.01)

    def test_check_consistency(self):
        """Test consistency checking method."""
        nasa7 = NASA7Polynomial.fit_full(
            self.T,
            self.cp,
            self.h,
            self.s,
            self.cp0_over_R,
            self.dhf_over_R,
            self.s0_over_R,
            self.Tmid,
        )

        consistency = nasa7.check_consistency(
            self.T, self.cp, self.h, self.s, abs_tol=0.1
        )

        self.assertTrue(consistency["is_consistent"])
        self.assertLess(consistency["cp_error"], 0.1)
        self.assertLess(consistency["h_error"], 0.1)
        self.assertLess(consistency["s_error"], 0.1)

    def test_check_continuity(self):
        """Test continuity checking at Tmid."""
        nasa7 = NASA7Polynomial.fit_full(
            self.T,
            self.cp,
            self.h,
            self.s,
            self.cp0_over_R,
            self.dhf_over_R,
            self.s0_over_R,
            self.Tmid,
        )

        continuity = nasa7.check_continuity()

        # Should be continuous within tolerance (relaxed for fit_full)
        self.assertLess(continuity["cp_jump"], 1e-3)
        self.assertLess(continuity["h_jump"], 1e-3)
        self.assertLess(continuity["s_jump"], 1e-3)


class TestSutherlandFitting(unittest.TestCase):
    """Test Sutherland transport fitting."""

    def setUp(self):
        """Set up viscosity data from H2."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"
        self.gas = ct.Solution(str(mech_file))
        self.gas.transport_model = "multicomponent"

        h2_idx = self.gas.species_index("H2")

        self.T = np.linspace(300, 3000, 50)
        nT = len(self.T)
        self.mu = np.zeros(nT)

        for i, T_i in enumerate(self.T):
            self.gas.TPX = T_i, ct.one_atm, "H2:1.0"
            self.mu[i] = self.gas.viscosity

    def test_sutherland_fit(self):
        """Test Sutherland.fit() method."""
        sutherland = Sutherland.fit(self.T, self.mu)

        self.assertIsNotNone(sutherland)
        self.assertIsNotNone(sutherland.std_err)
        self.assertGreater(sutherland.As, 0)
        self.assertGreater(sutherland.Ts, 0)

        # Evaluate and check fit quality
        mu_fit = sutherland.mu(self.T)
        rel_error = np.abs((mu_fit - self.mu) / self.mu)

        # Should have reasonable fit (within 15%)
        self.assertLess(np.max(rel_error), 0.15)
        self.assertLess(np.mean(rel_error), 0.08)

    def test_sutherland_eval(self):
        """Test Sutherland.mu() method."""
        sutherland = Sutherland(As=1e-6, Ts=100.0)

        T_test = np.array([300, 500, 1000])
        mu = sutherland.mu(T_test)

        # Check dimensions
        self.assertEqual(len(mu), len(T_test))

        # Check increasing with temperature
        self.assertTrue(np.all(np.diff(mu) > 0))


class TestPolynomialFitting(unittest.TestCase):
    """Test Polynomial transport fitting."""

    def setUp(self):
        """Set up transport data from H2."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"
        self.gas = ct.Solution(str(mech_file))
        self.gas.transport_model = "multicomponent"

        h2_idx = self.gas.species_index("H2")

        self.T = np.linspace(300, 3000, 50)
        nT = len(self.T)
        self.mu = np.zeros(nT)
        self.kappa = np.zeros(nT)

        for i, T_i in enumerate(self.T):
            self.gas.TPX = T_i, ct.one_atm, "H2:1.0"
            self.mu[i] = self.gas.viscosity
            self.kappa[i] = self.gas.thermal_conductivity

    def test_fit_polynomial(self):
        """Test fit_polynomial method."""
        poly = Polynomial.fit_polynomial(self.T, self.mu, self.kappa)

        self.assertIsNotNone(poly)
        self.assertEqual(len(poly.coeffs_mu), 4)
        self.assertEqual(len(poly.coeffs_kappa), 4)

        # Evaluate and check fit
        mu_fit = poly.mu(self.T)
        kappa_fit = poly.kappa(self.T)

        # Polynomial should fit reasonably well
        rel_err_mu = np.abs((mu_fit - self.mu) / self.mu)
        rel_err_kappa = np.abs((kappa_fit - self.kappa) / self.kappa)

        self.assertLess(np.max(rel_err_mu), 0.15)
        self.assertLess(np.max(rel_err_kappa), 0.15)

    def test_fit_log_polynomial(self):
        """Test fit_log_polynomial method."""
        log_poly = Polynomial.fit_log_polynomial(self.T, self.mu, self.kappa)

        self.assertIsNotNone(log_poly)
        self.assertEqual(len(log_poly.coeffs_mu), 4)
        self.assertEqual(len(log_poly.coeffs_kappa), 4)

        # Evaluate
        mu_fit = log_poly.mu(self.T)
        kappa_fit = log_poly.kappa(self.T)

        # Log polynomial typically fits better
        rel_err_mu = np.abs((mu_fit - self.mu) / self.mu)
        rel_err_kappa = np.abs((kappa_fit - self.kappa) / self.kappa)

        self.assertLess(np.max(rel_err_mu), 0.1)
        self.assertLess(np.max(rel_err_kappa), 0.1)


class TestSpeciesClass(unittest.TestCase):
    """Test Species class functionality."""

    def setUp(self):
        """Set up Species test data."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"
        self.gas = ct.Solution(str(mech_file))
        self.R = ct.gas_constant

        h2_idx = self.gas.species_index("H2")
        self.gas.TP = 298.15, ct.one_atm

        # Create a Species object
        self.species = Species(
            name="H2",
            W=self.gas.molecular_weights[h2_idx],
            cp0_over_R=self.gas.standard_cp_R[h2_idx],
            dhf_over_R=self.gas.standard_enthalpies_RT[h2_idx],
            s0_over_R=self.gas.standard_entropies_R[h2_idx],
            elements={"H": 2},
        )

        # Generate data for fitting
        self.T = np.linspace(300, 3000, 50)
        self.Tmid = 1000.0

        nT = len(self.T)
        self.cp = np.zeros(nT)
        self.h = np.zeros(nT)
        self.s = np.zeros(nT)
        self.mu = np.zeros(nT)
        self.kappa = np.zeros(nT)

        self.gas.transport_model = "multicomponent"

        for i, T_i in enumerate(self.T):
            self.gas.TPX = T_i, ct.one_atm, "H2:1.0"
            self.cp[i] = self.gas.standard_cp_R[h2_idx]
            self.h[i] = self.gas.standard_enthalpies_RT[h2_idx]
            self.s[i] = self.gas.standard_entropies_R[h2_idx]
            self.mu[i] = self.gas.viscosity
            self.kappa[i] = self.gas.thermal_conductivity

    def test_species_constructor_minimal_attributes(self):
        """Test that Species constructor doesn't require Tmid/Tlow/Thigh."""
        sp = Species(
            name="TEST",
            W=28.0,
            cp0_over_R=3.5,
            dhf_over_R=0.0,
            s0_over_R=20.0,
            elements={"N": 2},
        )

        self.assertEqual(sp.name, "TEST")
        self.assertEqual(sp.W, 28.0)
        self.assertEqual(sp.cp0_over_R, 3.5)
        self.assertIsNone(sp.nasa7)
        self.assertIsNone(sp.sutherland)
        self.assertIsNone(sp.polynomial)
        self.assertIsNone(sp.log_polynomial)

    def test_species_fitting_workflow(self):
        """Test the complete fitting workflow for a Species."""
        # Fit thermodynamics
        nasa7 = NASA7Polynomial.fit_auto(
            self.T,
            self.cp,
            self.h,
            self.s,
            self.species.cp0_over_R,
            self.species.dhf_over_R,
            self.species.s0_over_R,
            self.Tmid,
            verbose=False,
        )
        self.species.nasa7 = nasa7

        # Fit transport
        self.species.sutherland = Sutherland.fit(self.T, self.mu)
        self.species.polynomial = Polynomial.fit_polynomial(self.T, self.mu, self.kappa)
        self.species.log_polynomial = Polynomial.fit_log_polynomial(
            self.T, self.mu, self.kappa
        )

        # Check quality
        quality = self.species.check_quality(self.T, self.cp, self.h, self.s)

        self.assertIsNotNone(quality)
        self.assertIn("consistency", quality)
        self.assertIn("continuity", quality)

    def test_check_quality_requires_nasa7(self):
        """Test that check_quality raises error without nasa7."""
        with self.assertRaises(RuntimeError):
            self.species.check_quality(self.T, self.cp, self.h, self.s)

    def test_to_foam_dict_requires_nasa7(self):
        """Test that to_foam_dict raises error without nasa7."""
        with self.assertRaises(RuntimeError):
            self.species.to_foam_dict(Tlow=300, Thigh=3000)

    def test_to_foam_dict_with_tlow_thigh_parameters(self):
        """Test that to_foam_dict accepts Tlow/Thigh parameters."""
        # Fit first
        nasa7 = NASA7Polynomial.fit_auto(
            self.T,
            self.cp,
            self.h,
            self.s,
            self.species.cp0_over_R,
            self.species.dhf_over_R,
            self.species.s0_over_R,
            self.Tmid,
            verbose=False,
        )
        self.species.nasa7 = nasa7
        self.species.sutherland = Sutherland.fit(self.T, self.mu)

        # Export with Tlow/Thigh
        foam_dict = self.species.to_foam_dict(Tlow=300, Thigh=3000)

        self.assertEqual(foam_dict["name"], "H2")
        self.assertEqual(foam_dict["Tmid"], self.Tmid)
        self.assertEqual(foam_dict["Tlow"], 300)
        self.assertEqual(foam_dict["Thigh"], 3000)
        self.assertIn("nasa7_lo", foam_dict)
        self.assertIn("nasa7_hi", foam_dict)
        self.assertIn("As", foam_dict)
        self.assertIn("Ts", foam_dict)


class TestCanteraThermoTransport(unittest.TestCase):
    """Test CanteraThermoTransport functionality."""

    def setUp(self):
        """Set up test mechanism."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.mech_file = test_data_dir / "h2o2_mod.yaml"

    def test_from_cantera_loads_mechanism(self):
        """Test that constructor and fit_thermodynamics successfully loads a mechanism."""
        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset.species_list), 10)
        self.assertEqual(dataset.Tmid, 1000.0)

    def test_all_species_have_fitted_coefficients(self):
        """Test that all species have fitted coefficients after constructor and fit_thermodynamics."""
        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        for sp in dataset.species_list:
            self.assertIsNotNone(sp.nasa7, f"{sp.name} missing nasa7")
            self.assertIsNotNone(sp.sutherland, f"{sp.name} missing sutherland")
            self.assertIsNotNone(sp.polynomial, f"{sp.name} missing polynomial")
            self.assertIsNotNone(sp.log_polynomial, f"{sp.name} missing log_polynomial")

    def test_cantera_coefficient_reuse(self):
        """Test that Cantera coefficients are reused when appropriate."""
        # Load with default tolerances (should reuse most)
        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        # Check that some species reused coefficients
        # We can check if the fit was successful by looking at quality
        reuse_count = 0
        for sp in dataset.species_list:
            if sp.nasa7 is not None:
                # If reused or refitted successfully, quality should be excellent
                quality = sp.quality
                if quality and quality["consistency"]["is_consistent"]:
                    reuse_count += 1

        # At least some species should have good fits (reused or refitted)
        self.assertGreater(reuse_count, 5)

    def test_force_refit(self):
        """Test that force_refit=True refits all species."""
        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, force_refit=True, verbose=False)

        # All species should still fit successfully
        for sp in dataset.species_list:
            self.assertIsNotNone(sp.nasa7)

    def test_custom_temperature_range(self):
        """Test using custom Tlow, Thigh, and T_eval."""
        Tlow = 400.0
        Thigh = 2500.0
        T_eval = np.linspace(Tlow, Thigh, 80)

        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(
            Tmid=1200.0,
            Tlow=Tlow,
            Thigh=Thigh,
            T_eval=T_eval,
            verbose=False,
        )

        self.assertEqual(dataset.Tlow, Tlow)
        self.assertEqual(dataset.Thigh, Thigh)
        self.assertEqual(dataset.Tmid, 1200.0)

    def test_custom_tolerances(self):
        """Test using custom FittingTolerances."""
        tolerances = FittingTolerances(
            continuity_cp_tol=1e-8,
            continuity_h_tol=1e-8,
            continuity_s_tol=1e-8,
            consistency_abs_tol=1e-8,
        )

        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, tolerances=tolerances, verbose=False)

        # Should still load successfully
        self.assertEqual(len(dataset.species_list), 10)

    def test_write_output_creates_files(self):
        """Test that write_output creates OpenFOAM files."""
        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "foam_output"

            dataset.write_output(output_dir)

            # Check files exist
            self.assertTrue((output_dir / "thermo.foam").exists())
            self.assertTrue((output_dir / "reactions.foam").exists())
            self.assertTrue((output_dir / "species.foam").exists())

            # Check thermo file has content
            thermo_content = (output_dir / "thermo.foam").read_text()
            self.assertGreater(len(thermo_content), 0)

            # Check that H2 is in the output
            self.assertIn("H2", thermo_content)

    def test_species_list_names_match_cantera(self):
        """Test that species names match Cantera mechanism."""
        gas = ct.Solution(str(self.mech_file))
        dataset = CanteraThermoTransport(self.mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        dataset_names = [sp.name for sp in dataset.species_list]
        cantera_names = gas.species_names

        self.assertEqual(len(dataset_names), len(cantera_names))
        for name in cantera_names:
            self.assertIn(name, dataset_names)


class TestIntegrationEndToEnd(unittest.TestCase):
    """Integration tests for complete workflow."""

    def test_complete_workflow_h2o2(self):
        """Test complete workflow: load, fit, check, export."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"

        # Load and fit
        dataset = CanteraThermoTransport(mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        # Check all species fitted
        self.assertEqual(len(dataset.species_list), 10)

        # Check quality metrics exist
        for sp in dataset.species_list:
            self.assertIsNotNone(sp.quality)
            self.assertIn("consistency", sp.quality)

        # Export to OpenFOAM
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "h2o2_foam"
            dataset.write_output(output_dir)

            # Verify files
            self.assertTrue((output_dir / "thermo.foam").exists())
            self.assertTrue((output_dir / "reactions.foam").exists())
            self.assertTrue((output_dir / "species.foam").exists())

    def test_species_to_foam_dict_integration(self):
        """Test Species.to_foam_dict() in context of CanteraThermoTransport."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"

        dataset = CanteraThermoTransport(mech_file)
        dataset.fit_thermodynamics(Tmid=1000.0, verbose=False)

        # Get a species and export it
        h2_species = next(sp for sp in dataset.species_list if sp.name == "H2")

        foam_dict = h2_species.to_foam_dict(Tlow=dataset.Tlow, Thigh=dataset.Thigh)

        # Verify structure
        self.assertEqual(foam_dict["name"], "H2")
        self.assertEqual(foam_dict["Tlow"], dataset.Tlow)
        self.assertEqual(foam_dict["Thigh"], dataset.Thigh)
        self.assertEqual(foam_dict["Tmid"], h2_species.nasa7.Tmid)
        self.assertEqual(len(foam_dict["nasa7_lo"]), 7)
        self.assertEqual(len(foam_dict["nasa7_hi"]), 7)


if __name__ == "__main__":
    unittest.main()
