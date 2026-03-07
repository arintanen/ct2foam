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
    SpeciesList,
    FittingTolerances,
)


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
    """Test Species class functionality using Species.from_ct."""

    def setUp(self):
        """Set up Species test data."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.mech_file = test_data_dir / "h2o2_mod.yaml"
        self.gas = ct.Solution(str(self.mech_file))

    def test_species_from_ct_creates_fitted_species(self):
        """Test that Species.from_ct creates a fully fitted Species."""
        sp = Species.from_ct(self.gas, "H2", Tmin=300, Tmax=3000, Tmid=1000)

        self.assertEqual(sp.name, "H2")
        self.assertIsNotNone(sp.W)
        self.assertIsNotNone(sp.nasa7)
        self.assertIsNotNone(sp.sutherland)
        self.assertIsNotNone(sp.polynomial)
        self.assertIsNotNone(sp.log_polynomial)

    def test_to_foam_dict_requires_nasa7(self):
        """Test that to_foam_dict raises error without nasa7."""
        sp = Species(name="TEST", W=28.0, elements={"N": 2})
        with self.assertRaises(RuntimeError):
            sp.to_foam_dict(Tlow=300, Thigh=3000)

    def test_to_foam_dict_with_fitted_species(self):
        """Test that to_foam_dict works with fitted Species."""
        sp = Species.from_ct(self.gas, "H2", Tmin=300, Tmax=3000, Tmid=1000)

        # Export with Tlow/Thigh
        foam_dict = sp.to_foam_dict(Tlow=300, Thigh=3000)

        self.assertEqual(foam_dict["name"], "H2")
        self.assertEqual(foam_dict["Tmid"], 1000)
        self.assertEqual(foam_dict["Tlow"], 300)
        self.assertEqual(foam_dict["Thigh"], 3000)
        self.assertIn("nasa7_lo", foam_dict)
        self.assertIn("nasa7_hi", foam_dict)
        self.assertIn("As", foam_dict)
        self.assertIn("Ts", foam_dict)


class TestSpeciesList(unittest.TestCase):
    """Test SpeciesList functionality."""

    def setUp(self):
        """Set up test mechanism."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.mech_file = test_data_dir / "h2o2_mod.yaml"

    def test_from_ct_mech_loads_mechanism(self):
        """Test that from_ct_mech successfully loads and fits a mechanism."""
        species_list = SpeciesList.from_ct_mech(
            str(self.mech_file), Tmin=300, Tmax=3000, Tmid=1000
        )

        self.assertIsNotNone(species_list)
        self.assertEqual(len(species_list.species), 10)
        # Check that all species have Tmid=1000
        for sp in species_list.species:
            self.assertIsNotNone(sp.nasa7)
            self.assertEqual(sp.nasa7.Tmid, 1000.0)

    def test_all_species_have_fitted_coefficients(self):
        """Test that all species have fitted coefficients after from_ct_mech."""
        species_list = SpeciesList.from_ct_mech(
            str(self.mech_file), Tmin=300, Tmax=3000, Tmid=1000
        )

        for sp in species_list.species:
            self.assertIsNotNone(sp.nasa7, f"{sp.name} missing nasa7")
            self.assertIsNotNone(sp.sutherland, f"{sp.name} missing sutherland")
            self.assertIsNotNone(sp.polynomial, f"{sp.name} missing polynomial")
            self.assertIsNotNone(sp.log_polynomial, f"{sp.name} missing log_polynomial")

    def test_custom_temperature_range(self):
        """Test using custom Tmin, Tmax, Tmid."""
        Tmin = 400.0
        Tmax = 2500.0
        Tmid = 1200.0

        species_list = SpeciesList.from_ct_mech(
            str(self.mech_file), Tmin=Tmin, Tmax=Tmax, Tmid=Tmid
        )

        # Check that all species have the correct temperature bounds
        for sp in species_list.species:
            self.assertIsNotNone(sp.nasa7)
            self.assertEqual(sp.nasa7.Tlow, Tmin)
            self.assertEqual(sp.nasa7.Tmax, Tmax)
            self.assertEqual(sp.nasa7.Tmid, Tmid)

    def test_write_foam_creates_files(self):
        """Test that write_foam creates OpenFOAM files."""
        species_list = SpeciesList.from_ct_mech(
            str(self.mech_file), Tmin=300, Tmax=3000, Tmid=1000
        )

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "foam_output"

            species_list.write_foam(output_dir)

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
        species_list = SpeciesList.from_ct_mech(
            str(self.mech_file), Tmin=300, Tmax=3000, Tmid=1000
        )

        dataset_names = [sp.name for sp in species_list.species]
        cantera_names = gas.species_names

        self.assertEqual(len(dataset_names), len(cantera_names))
        for name in cantera_names:
            self.assertIn(name, dataset_names)


class TestIntegrationEndToEnd(unittest.TestCase):
    """Integration tests for complete workflow."""

    def test_complete_workflow_h2o2(self):
        """Test complete workflow: load, fit, export."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"

        # Load and fit (happens in constructor)
        species_list = SpeciesList.from_ct_mech(
            str(mech_file), Tmin=300, Tmax=3000, Tmid=1000
        )

        # Check all species fitted
        self.assertEqual(len(species_list.species), 10)

        # Export to OpenFOAM
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "h2o2_foam"
            species_list.write_foam(output_dir)

            # Verify files
            self.assertTrue((output_dir / "thermo.foam").exists())
            self.assertTrue((output_dir / "reactions.foam").exists())
            self.assertTrue((output_dir / "species.foam").exists())

    def test_species_to_foam_dict_integration(self):
        """Test Species.to_foam_dict() in context of SpeciesList."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        mech_file = test_data_dir / "h2o2_mod.yaml"

        species_list = SpeciesList.from_ct_mech(
            str(mech_file), Tmin=300, Tmax=3000, Tmid=1000
        )

        # Get a species and export it
        h2_species = next(sp for sp in species_list.species if sp.name == "H2")

        # Use temperature bounds from the species itself
        foam_dict = h2_species.to_foam_dict(
            Tlow=h2_species.nasa7.Tlow, Thigh=h2_species.nasa7.Tmax
        )

        # Verify structure
        self.assertEqual(foam_dict["name"], "H2")
        self.assertEqual(foam_dict["Tlow"], 300)
        self.assertEqual(foam_dict["Thigh"], 3000)
        self.assertEqual(foam_dict["Tmid"], h2_species.nasa7.Tmid)
        self.assertEqual(len(foam_dict["nasa7_lo"]), 7)
        self.assertEqual(len(foam_dict["nasa7_hi"]), 7)


if __name__ == "__main__":
    unittest.main()
