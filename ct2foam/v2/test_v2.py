import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from ct2foam.v2.coefficients import NASA7Polynomial, Sutherland, Polynomial
from ct2foam.v2.species_dataset import SpeciesDataset
from ct2foam.v2.mechanism_dataset import MechanismDataset
from ct2foam.v2.fitting_tolerances import FittingTolerances

# Note, OF_reference/Test-thermoMixture.C
# is used as a source of the reference data tested here.

# R = 8314.46261815324
R = 8314.47006650545  # taken from openFoam -- differs slightly from standards

T_STD = 298.15

H2O_TMID = 1000.0
H2O_C_LO = np.array(
    [3.38684, 0.00347498, -6.3547e-06, 6.96858e-09, -2.50659e-12, -30208.1, 2.59023]
)
H2O_C_HI = np.array(
    [2.67215, 0.00305629, -8.73026e-07, 1.201e-10, -6.39162e-15, -29899.2, 6.86282]
)


class TestNASA7Polynomial(unittest.TestCase):
    """Test NASA polynomials and related functionalities."""

    def setUp(self):
        self.nasa = NASA7Polynomial(H2O_C_LO, H2O_C_HI, H2O_TMID)

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
        self.assertAlmostEqual(np.abs(result[0]), 0.00164496)

    def test_dcpdT_over_R_matches_numerical_derivative(self):
        """dcp/dT should approximate numerical derivative."""
        T = 800.0
        dT = 0.01
        dcpdT_analytical = self.nasa.dcpdT_over_R(T)[0]
        cp_plus = self.nasa.cp_over_R(T + dT)[0]
        cp_minus = self.nasa.cp_over_R(T - dT)[0]
        dcpdT_numerical = (cp_plus - cp_minus) / (2 * dT)
        self.assertAlmostEqual(dcpdT_analytical, dcpdT_numerical, places=4)

    def test_check_consistency_returns_dict_keys(self):
        """check_consistency returns expected dict keys."""
        T = np.linspace(300, 3000, 50)
        cp = self.nasa.cp_over_R(T)
        h = self.nasa.h_over_RT(T)
        s = self.nasa.s_over_R(T)
        result = self.nasa.check_consistency(T, cp, h, s)
        for key in ("is_consistent", "cp_error", "h_error", "s_error", "max_error"):
            self.assertIn(key, result)

    def test_check_consistency_perfect_match(self):
        """Evaluating from own coefficients should be perfectly consistent."""
        T = np.linspace(300, 3000, 50)
        cp = self.nasa.cp_over_R(T)
        h = self.nasa.h_over_RT(T)
        s = self.nasa.s_over_R(T)
        result = self.nasa.check_consistency(T, cp, h, s)
        self.assertTrue(result["is_consistent"])
        self.assertLess(result["max_error"], 1e-10)

    def test_check_continuity_returns_dict_keys(self):
        """Check_continuity returns expected dict keys."""
        result = self.nasa.check_continuity()
        for key in ("is_continuous", "cp_jump", "cpdT_jump", "h_jump", "s_jump"):
            self.assertIn(key, result)

    def test_check_continuity_h2o_reference(self):
        """
        H2O reference coefficients should have reasonable continuity. Note,
        this checks that the continuity method works and is not considering
        the fit quality per'se yet.
        """
        result = self.nasa.check_continuity(
            cp_tol=1e-4, cpdT_tol=1e-4, h_tol=1e-4, s_tol=1e-4
        )
        # With loose tolerances the reference data should pass
        self.assertTrue(result["is_continuous"])


class TestSutherland(unittest.TestCase):
    """Test Sutherland viscosity model."""

    def setUp(self):
        self.suth = Sutherland(As=1.67212e-06, Ts=170.672)

    def test_mu_positive(self):
        """Viscosity must be positive for T > 0."""
        T = np.array([300.0, 500.0, 1000.0, 2000.0])
        mu = self.suth.mu(T)
        self.assertTrue(np.all(mu > 0))

    def test_mu_formula_manual(self):
        """Check viscosity formula against manual calculation."""
        T = 500.0
        expected = 1.67212e-06 * np.sqrt(500.0) / (1.0 + 170.672 / 500.0)
        result = self.suth.mu(T)
        self.assertAlmostEqual(float(result), expected, places=12)

    def test_kappa_euken_positive(self):
        """Euken thermal conductivity must be positive for valid inputs."""
        T = 1000.0
        cv_mole = 20000.0  # J/mol/K
        W = 18.015
        kappa = self.suth.kappa_euken(T, cv_mole, W, R)
        self.assertGreater(float(kappa), 0)

    def test_kappa_euken_formula(self):
        """Check Euken formula against manual calculation."""
        T = 500.0
        cv_mole = 25000.0
        W = 28.0
        mu = self.suth.mu(T)
        Cv = cv_mole / W
        Rspecific = R / W
        expected = float(mu) * Cv * (1.32 + 1.77 * Rspecific / Cv)
        result = self.suth.kappa_euken(T, cv_mole, W, R)
        self.assertAlmostEqual(float(result), expected, places=10)

    def test_mu_scalar_and_array_agree(self):
        """Scalar and array evaluation should give the same result."""
        T_val = 750.0
        scalar_result = float(self.suth.mu(T_val))
        array_result = float(self.suth.mu(np.array([T_val]))[0])
        self.assertAlmostEqual(scalar_result, array_result, places=12)

    # Tests against OpenFOAM reference values
    def test_sutherland0(self):
        suth = Sutherland(As=1, Ts=1)
        mu = suth.mu(1)
        mu_foam = 0.5
        self.assertTrue(np.abs(mu - mu_foam) < 1e-12)

    def test_euken0(self):
        """Euken reference data from OF"""
        suth = Sutherland(As=1, Ts=1)
        cv_mole = -R + 4
        kappa_foam = 936.697882481863
        kappa = suth.kappa_euken(1, cv_mole, 2, R)
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
        kappa = suth.kappa_euken(400, cv_mole, 18.0153, R)
        self.assertTrue(np.abs(kappa - kappa_foam) < 1e-12)


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


class TestSpeciesDatasetFitting(unittest.TestCase):
    """Test SpeciesDataset thermo and transport fitting with real data."""

    @classmethod
    def setUpClass(cls):
        """Load h2o2 mechanism via Cantera once."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        p0 = ct.one_atm
        gas.transport_model = "multicomponent"

        T = np.linspace(280, 3000, 128)
        T = np.sort(T)
        Tmid = 1000.0
        idx = T.searchsorted(Tmid)
        T = np.concatenate((T[:idx], [Tmid], T[idx:]))

        cls.T = T
        cls.Tmid = Tmid
        cls.R = ct.gas_constant
        cls.species_data = {}

        for sp_name in ["H2O", "H2", "O2", "AR"]:
            i = gas.species_index(sp_name)
            reactants = sp_name + ":1.0"
            nT = len(T)

            mu_arr = np.zeros(nT)
            kappa_arr = np.zeros(nT)
            cp_arr = np.zeros(nT)
            h_arr = np.zeros(nT)
            s_arr = np.zeros(nT)
            cv_arr = np.zeros(nT)

            for j in range(nT):
                gas.TPX = T[j], p0, reactants
                mu_arr[j] = gas.viscosity
                kappa_arr[j] = gas.thermal_conductivity
                cp_arr[j] = gas.cp_mole
                h_arr[j] = gas.enthalpy_mole
                s_arr[j] = gas.entropy_mole
                cv_arr[j] = gas.cv_mole

            cp0_over_R = gas.species(i).thermo.cp(298.15) / cls.R
            dhf_over_R = gas.species(i).thermo.h(298.15) / cls.R
            s0_over_R = gas.species(i).thermo.s(298.15) / cls.R

            sd = SpeciesDataset(
                name=sp_name,
                T=T,
                cp=cp_arr,
                h=h_arr,
                s=s_arr,
                mu=mu_arr,
                kappa=kappa_arr,
                cp0_over_R=cp0_over_R,
                dhf_over_R=dhf_over_R,
                s0_over_R=s0_over_R,
                Tmid=Tmid,
                W=gas.molecular_weights[i],
                cv_mole=cv_arr,
            )
            cls.species_data[sp_name] = sd

    def _make_fitted_species(self, name):
        """Create a deep copy and run fits."""
        import copy

        sd = copy.deepcopy(self.species_data[name])
        sd.fit_thermo()
        sd.fit_transport()
        return sd

    def test_fit_thermo_returns_nasa7(self):
        """fit_thermo returns a NASA7Polynomial object."""
        sd = self._make_fitted_species("H2O")
        self.assertIsInstance(sd.nasa7, NASA7Polynomial)

    def test_fit_thermo_coeffs_shape(self):
        """Fitted coefficients should each have 7 elements."""
        sd = self._make_fitted_species("H2O")
        self.assertEqual(len(sd.nasa7.coeffs_low), 7)
        self.assertEqual(len(sd.nasa7.coeffs_high), 7)

    def test_fit_thermo_cp_accuracy(self):
        """Fitted cp should match reference data within 1%."""
        sd = self._make_fitted_species("H2O")
        cp_fit = sd.nasa7.cp_over_R(sd.T) * R
        rel_err = np.linalg.norm(sd.cp - cp_fit) / np.linalg.norm(sd.cp)
        self.assertLess(rel_err, 1e-5)

    def test_fit_thermo_continuity(self):
        """Fitted NASA7 should be continuous at Tmid."""
        sd = self._make_fitted_species("H2O")
        result = sd.nasa7.check_continuity(
            cp_tol=1e-15, cpdT_tol=1e-15, h_tol=1e-15, s_tol=1e-15
        )
        self.assertTrue(result["is_continuous"])

    def test_fit_thermo_h2_species(self):
        """Fitting works for H2 species."""
        sd = self._make_fitted_species("H2")
        self.assertIsNotNone(sd.nasa7)
        cp_fit = sd.nasa7.cp_over_R(sd.T) * R
        rel_err = np.linalg.norm(sd.cp - cp_fit) / np.linalg.norm(sd.cp)
        self.assertLess(rel_err, 2e-3)

    def test_fit_thermo_o2_species(self):
        """Fitting works for O2 species."""
        sd = self._make_fitted_species("O2")
        self.assertIsNotNone(sd.nasa7)
        cp_fit = sd.nasa7.cp_over_R(sd.T) * R
        rel_err = np.linalg.norm(sd.cp - cp_fit) / np.linalg.norm(sd.cp)
        self.assertLess(rel_err, 5e-4)

    def test_fit_thermo_ar_monatomic(self):
        """Fitting works for monatomic Ar species."""
        sd = self._make_fitted_species("AR")
        self.assertIsNotNone(sd.nasa7)
        cp_fit = sd.nasa7.cp_over_R(sd.T) * R
        rel_err = np.linalg.norm(sd.cp - cp_fit) / np.linalg.norm(sd.cp)
        self.assertLess(rel_err, 1e-6)

    def test_fit_thermo_strategy_full(self):
        """Explicit 'full' strategy produces valid results."""
        import copy

        sd = copy.deepcopy(self.species_data["H2O"])
        sd.fit_thermo(strategy="full")
        self.assertIsNotNone(sd.nasa7)
        cp_fit = sd.nasa7.cp_over_R(sd.T) * R
        rel_err = np.linalg.norm(sd.cp - cp_fit) / np.linalg.norm(sd.cp)
        self.assertLess(rel_err, 1e-3)

    def test_fit_thermo_strategy_cp_only(self):
        """Explicit 'cp_only' strategy produces valid results."""
        import copy

        sd = copy.deepcopy(self.species_data["H2O"])
        sd.fit_thermo(strategy="cp_only")
        self.assertIsNotNone(sd.nasa7)
        cp_fit = sd.nasa7.cp_over_R(sd.T) * R
        rel_err = np.linalg.norm(sd.cp - cp_fit) / np.linalg.norm(sd.cp)
        self.assertLess(rel_err, 1e-5)

    def test_fit_thermo_invalid_strategy_raises(self):
        """Invalid strategy raises ValueError."""
        import copy

        sd = copy.deepcopy(self.species_data["H2O"])
        with self.assertRaises(ValueError):
            sd.fit_thermo(strategy="bogus")

    def test_fit_transport_returns_dict(self):
        """fit_transport returns a dict with expected keys."""
        sd = self._make_fitted_species("H2O")
        import copy

        sd2 = copy.deepcopy(self.species_data["H2O"])
        result = sd2.fit_transport()
        self.assertIn("sutherland", result)
        self.assertIn("polynomial", result)
        self.assertIn("log_polynomial", result)

    def test_fit_transport_sutherland_positive_As(self):
        """Sutherland As should be positive."""
        sd = self._make_fitted_species("H2O")
        self.assertGreater(sd.sutherland.As, 0)

    def test_fit_transport_polynomial_mu_accuracy(self):
        """Polynomial mu fit error should be small."""
        import copy

        sd = copy.deepcopy(self.species_data["H2O"])
        result = sd.fit_transport()
        self.assertLess(result["err_mu_polynomial"], 5e-3)

    def test_fit_transport_log_polynomial_stored(self):
        """Log-polynomial should be stored on the dataset."""
        sd = self._make_fitted_species("H2O")
        self.assertIsNotNone(sd.log_polynomial)
        self.assertEqual(sd.log_polynomial.poly_type, "log_polynomial")

    def test_check_quality_returns_dict(self):
        """check_quality returns dict with consistency and continuity."""
        sd = self._make_fitted_species("H2O")
        quality = sd.check_quality()
        self.assertIn("consistency", quality)
        self.assertIn("continuity", quality)


class TestSpeciesDatasetOutput(unittest.TestCase):
    """Test SpeciesDataset to_foam_dict output."""

    @classmethod
    def setUpClass(cls):
        """Load and fit H2O once."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        p0 = ct.one_atm
        gas.transport_model = "multicomponent"

        T = np.linspace(280, 3000, 64)
        Tmid = 1000.0
        idx = T.searchsorted(Tmid)
        T = np.concatenate((T[:idx], [Tmid], T[idx:]))

        sp_name = "H2O"
        i = gas.species_index(sp_name)
        reactants = sp_name + ":1.0"
        nT = len(T)

        arrs = {k: np.zeros(nT) for k in ("mu", "kappa", "cp", "h", "s", "cv")}
        for j in range(nT):
            gas.TPX = T[j], p0, reactants
            arrs["mu"][j] = gas.viscosity
            arrs["kappa"][j] = gas.thermal_conductivity
            arrs["cp"][j] = gas.cp_mole
            arrs["h"][j] = gas.enthalpy_mole
            arrs["s"][j] = gas.entropy_mole
            arrs["cv"][j] = gas.cv_mole

        sd = SpeciesDataset(
            name=sp_name,
            T=T,
            cp=arrs["cp"],
            h=arrs["h"],
            s=arrs["s"],
            mu=arrs["mu"],
            kappa=arrs["kappa"],
            cp0_over_R=gas.species(i).thermo.cp(298.15) / ct.gas_constant,
            dhf_over_R=gas.species(i).thermo.h(298.15) / ct.gas_constant,
            s0_over_R=gas.species(i).thermo.s(298.15) / ct.gas_constant,
            Tmid=Tmid,
            W=gas.molecular_weights[i],
            cv_mole=arrs["cv"],
            elements={"H": 2, "O": 1},
        )
        sd.fit_thermo()
        sd.fit_transport()
        cls.sd = sd

    def test_to_foam_dict_has_name(self):
        """Foam dict has the species name."""
        d = self.sd.to_foam_dict()
        self.assertEqual(d["name"], "H2O")

    def test_to_foam_dict_has_nasa7_coefficients(self):
        """Foam dict has low and high NASA7 coefficient arrays."""
        d = self.sd.to_foam_dict()
        self.assertEqual(len(d["nasa7_lo"]), 7)
        self.assertEqual(len(d["nasa7_hi"]), 7)

    def test_to_foam_dict_has_sutherland(self):
        """Foam dict has Sutherland As and Ts."""
        d = self.sd.to_foam_dict()
        self.assertIn("As", d)
        self.assertIn("Ts", d)

    def test_to_foam_dict_has_polynomial(self):
        """Foam dict has polynomial coefficients."""
        d = self.sd.to_foam_dict()
        self.assertIn("poly_mu", d)
        self.assertIn("poly_kappa", d)

    def test_to_foam_dict_has_log_polynomial(self):
        """Foam dict has log-polynomial coefficients."""
        d = self.sd.to_foam_dict()
        self.assertIn("logpoly_mu", d)
        self.assertIn("logpoly_kappa", d)

    def test_to_foam_dict_has_elements(self):
        """Foam dict has elements when provided."""
        d = self.sd.to_foam_dict()
        self.assertEqual(d["elements"], {"H": 2, "O": 1})


class TestMechanismDatasetWorkflow(unittest.TestCase):
    """Test MechanismDataset end-to-end workflow."""

    @classmethod
    def setUpClass(cls):
        """Load h2o2 mechanism via from_cantera."""
        cls.mech = MechanismDataset.from_cantera(
            "h2o2.yaml",
            Tmid=1000.0,
            Tlow=280.0,
            Thigh=3000.0,
            T_eval=np.linspace(280, 3000, 64),
        )
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_from_cantera_species_count(self):
        """h2o2 mechanism should have 10 species."""
        self.assertEqual(len(self.mech.species_datasets), 10)

    def test_from_cantera_species_names(self):
        """Species names should match Cantera output."""
        names = [sd.name for sd in self.mech.species_datasets]
        self.assertIn("H2O", names)
        self.assertIn("H2", names)
        self.assertIn("O2", names)

    def test_from_cantera_tmid_stored(self):
        """Tmid should be stored correctly."""
        self.assertEqual(self.mech.Tmid, 1000.0)

    def test_from_cantera_T_contains_tmid(self):
        """Temperature array should contain Tmid."""
        for sd in self.mech.species_datasets:
            self.assertIn(1000.0, sd.T)
            break

    def test_from_cantera_species_has_W(self):
        """Each species should have molecular weight."""
        for sd in self.mech.species_datasets:
            self.assertIsNotNone(sd.W)
            self.assertGreater(sd.W, 0)

    def test_from_cantera_species_has_elements(self):
        """Species should have element composition."""
        h2o = [sd for sd in self.mech.species_datasets if sd.name == "H2O"][0]
        self.assertIn("H", h2o.elements)
        self.assertIn("O", h2o.elements)

    def test_fit_all_succeeds(self):
        """fit_all should succeed for all h2o2 species."""
        import copy

        mech = copy.deepcopy(self.mech)
        result = mech.fit_all(verbose=False)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["succeeded"], 10)

    def test_fit_all_details_per_species(self):
        """fit_all result has details for each species."""
        import copy

        mech = copy.deepcopy(self.mech)
        result = mech.fit_all(verbose=False)
        self.assertEqual(len(result["details"]), 10)

    def test_write_output_creates_files(self):
        """write_output should create thermo.foam, species.foam, reactions.foam."""
        import copy

        mech = copy.deepcopy(self.mech)
        mech.fit_all(verbose=False)
        out_dir = Path(self.tmpdir) / "write_test"
        mech.write_output(out_dir)
        print(out_dir)
        self.assertTrue((out_dir / "thermo.foam").exists())
        self.assertTrue((out_dir / "species.foam").exists())
        self.assertTrue((out_dir / "reactions.foam").exists())

    def test_write_output_thermo_nonempty(self):
        """thermo.foam should have content."""
        import copy

        mech = copy.deepcopy(self.mech)
        mech.fit_all(verbose=False)
        out_dir = Path(self.tmpdir) / "write_content_test"
        mech.write_output(out_dir)
        content = (out_dir / "thermo.foam").read_text()
        self.assertGreater(len(content), 100)
        self.assertIn("H2O", content)

    def test_print_fit_summary_runs(self):
        """print_fit_summary should execute without error."""
        import copy, io, sys

        mech = copy.deepcopy(self.mech)
        mech.fit_all(verbose=False)
        captured = io.StringIO()
        sys.stdout = captured
        mech.print_fit_summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Fit summary", output)

    def test_fit_all_with_gri30(self):
        """fit_all should succeed for the larger GRI-3.0 mechanism."""
        mech = MechanismDataset.from_cantera(
            "gri30.yaml",
            Tmid=1000.0,
            Tlow=300.0,
            Thigh=3000.0,
            T_eval=np.linspace(300, 3000, 50),
        )
        result = mech.fit_all(verbose=False)
        self.assertTrue(result["succeeded"], 53)


class TestMechanismDatasetErrorHandling(unittest.TestCase):
    """Test MechanismDataset error logging and recovery."""

    def test_fit_all_continues_on_failure(self):
        """fit_all should continue to next species on failure."""
        # Create a dataset with one broken species (all-zero data)
        T = np.linspace(300, 3000, 50)
        idx = T.searchsorted(1000.0)
        T = np.concatenate((T[:idx], [1000.0], T[idx:]))

        good = SpeciesDataset(
            name="GOOD",
            T=T,
            cp=3.5 * R * np.ones_like(T),
            h=3.5 * R * T,
            s=3.5 * R * np.log(T),
            mu=1e-5 * np.ones_like(T),
            kappa=0.02 * np.ones_like(T),
            cp0_over_R=3.5,
            dhf_over_R=0.0,
            s0_over_R=150.0,
            Tmid=1000.0,
            W=28.0,
        )
        bad = SpeciesDataset(
            name="BAD",
            T=T,
            cp=np.zeros_like(T),
            h=np.zeros_like(T),
            s=np.zeros_like(T),
            mu=np.zeros_like(T),
            kappa=np.zeros_like(T),
            cp0_over_R=0.0,
            dhf_over_R=0.0,
            s0_over_R=0.0,
            Tmid=1000.0,
            W=28.0,
        )
        mech = MechanismDataset([good, bad], "test", 1000.0, 300.0, 3000.0)
        result = mech.fit_all(verbose=False)
        # Should not crash and fitting of a zero array should be successfull
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["succeeded"], 2)

    def test_failed_species_logged(self):
        """Failed species should appear in failed_species dict."""
        T = np.linspace(300, 3000, 50)
        idx = T.searchsorted(1000.0)
        T = np.concatenate((T[:idx], [1000.0], T[idx:]))

        bad = SpeciesDataset(
            name="BAD_SP",
            T=T,
            cp=np.full_like(T, np.nan),
            h=np.full_like(T, np.nan),
            s=np.full_like(T, np.nan),
            mu=np.full_like(T, np.nan),
            kappa=np.full_like(T, np.nan),
            cp0_over_R=np.nan,
            dhf_over_R=np.nan,
            s0_over_R=np.nan,
            Tmid=1000.0,
            W=28.0,
        )
        mech = MechanismDataset([bad], "test", 1000.0, 300.0, 3000.0)
        result = mech.fit_all(verbose=False)
        self.assertIn("BAD_SP", mech.failed_species)
        self.assertEqual(result["failed"], 1)

    def test_error_message_contains_tune_tolerances(self):
        """Error message should contain 'tune your tolerances'."""
        T = np.linspace(300, 3000, 50)
        idx = T.searchsorted(1000.0)
        T = np.concatenate((T[:idx], [1000.0], T[idx:]))

        bad = SpeciesDataset(
            name="BAD2",
            T=T,
            cp=np.full_like(T, np.nan),
            h=np.full_like(T, np.nan),
            s=np.full_like(T, np.nan),
            mu=np.full_like(T, np.nan),
            kappa=np.full_like(T, np.nan),
            cp0_over_R=np.nan,
            dhf_over_R=np.nan,
            s0_over_R=np.nan,
            Tmid=1000.0,
            W=28.0,
        )
        mech = MechanismDataset([bad], "test", 1000.0, 300.0, 3000.0)
        mech.fit_all(verbose=False)
        msg = mech.failed_species.get("BAD2", "")
        print(msg)
        self.assertIn("tune your tolerances", msg)

    def test_check_quality_before_fit_raises(self):
        """check_quality before fit_thermo should raise RuntimeError."""
        T = np.linspace(300, 3000, 50)
        sd = SpeciesDataset(
            name="X",
            T=T,
            cp=np.ones_like(T),
            h=np.ones_like(T),
            s=np.ones_like(T),
            mu=np.ones_like(T),
            kappa=np.ones_like(T),
            cp0_over_R=1.0,
            dhf_over_R=1.0,
            s0_over_R=1.0,
            Tmid=1000.0,
        )
        with self.assertRaises(RuntimeError):
            sd.check_quality()

    def test_to_foam_dict_before_fit_raises(self):
        """to_foam_dict before fit_thermo should raise RuntimeError."""
        T = np.linspace(300, 3000, 50)
        sd = SpeciesDataset(
            name="X",
            T=T,
            cp=np.ones_like(T),
            h=np.ones_like(T),
            s=np.ones_like(T),
            mu=np.ones_like(T),
            kappa=np.ones_like(T),
            cp0_over_R=1.0,
            dhf_over_R=1.0,
            s0_over_R=1.0,
            Tmid=1000.0,
        )
        with self.assertRaises(RuntimeError):
            sd.to_foam_dict()


class TestNASA7ReuseLogic(unittest.TestCase):
    """Test smart reuse of Cantera NASA7 coefficients."""

    def test_reuses_cantera_coeffs_when_valid(self):
        """Reuse Cantera coefficients when continuous, consistent, and Tmid matches."""
        import cantera as ct

        # Load H2O from h2o2 mechanism
        gas = ct.Solution("h2o2.yaml")
        Tmid = 1000.0
        T = np.linspace(300, 3000, 50)

        # Ensure Tmid is in T array
        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        i = gas.species_index("H2O")
        sp_obj = gas.species(i)

        # Extract Cantera NASA7 coefficients
        coeffs = sp_obj.thermo.coeffs
        ct_Tmid = float(coeffs[0])
        ct_c_hi = np.array(coeffs[1:8])
        ct_c_lo = np.array(coeffs[8:15])
        cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

        # Evaluate data
        R = ct.gas_constant
        p0 = ct.one_atm
        cp_arr = np.zeros(len(T))
        h_arr = np.zeros(len(T))
        s_arr = np.zeros(len(T))
        mu_arr = np.zeros(len(T))
        kappa_arr = np.zeros(len(T))

        for j, Tj in enumerate(T):
            gas.TPX = Tj, p0, "H2O:1.0"
            cp_arr[j] = gas.cp_mole
            h_arr[j] = gas.enthalpy_mole
            s_arr[j] = gas.entropy_mole
            mu_arr[j] = gas.viscosity
            kappa_arr[j] = gas.thermal_conductivity

        cp0_over_R = sp_obj.thermo.cp(298.15) / R
        dhf_over_R = sp_obj.thermo.h(298.15) / R
        s0_over_R = sp_obj.thermo.s(298.15) / R

        # Create SpeciesDataset with Cantera coefficients
        sd = SpeciesDataset(
            name="H2O",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=mu_arr,
            kappa=kappa_arr,
            cp0_over_R=cp0_over_R,
            dhf_over_R=dhf_over_R,
            s0_over_R=s0_over_R,
            Tmid=ct_Tmid,  # Use Cantera's Tmid
            cantera_nasa7=cantera_nasa7,
            is_nasa7=True,
        )

        # Fit with verbose to verify reuse message
        result = sd.fit_thermo(verbose=False)

        # Should return same object (reused)
        self.assertIs(result, cantera_nasa7)
        self.assertEqual(sd.nasa7.Tmid, ct_Tmid)

    def test_refits_when_tmid_differs(self):
        """Refit (cp-only) when continuous but Tmid differs."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        target_Tmid = 1200.0  # Different from Cantera's
        T = np.linspace(300, 3000, 50)

        if target_Tmid not in T:
            T = np.sort(np.append(T, target_Tmid))

        i = gas.species_index("H2")
        sp_obj = gas.species(i)

        # Extract Cantera NASA7 coefficients
        coeffs = sp_obj.thermo.coeffs
        ct_Tmid = float(coeffs[0])
        ct_c_hi = np.array(coeffs[1:8])
        ct_c_lo = np.array(coeffs[8:15])
        cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

        # Evaluate data
        R = ct.gas_constant
        p0 = ct.one_atm
        cp_arr = np.zeros(len(T))
        h_arr = np.zeros(len(T))
        s_arr = np.zeros(len(T))
        mu_arr = np.zeros(len(T))
        kappa_arr = np.zeros(len(T))

        for j, Tj in enumerate(T):
            gas.TPX = Tj, p0, "H2:1.0"
            cp_arr[j] = gas.cp_mole
            h_arr[j] = gas.enthalpy_mole
            s_arr[j] = gas.entropy_mole
            mu_arr[j] = gas.viscosity
            kappa_arr[j] = gas.thermal_conductivity

        cp0_over_R = sp_obj.thermo.cp(298.15) / R
        dhf_over_R = sp_obj.thermo.h(298.15) / R
        s0_over_R = sp_obj.thermo.s(298.15) / R

        sd = SpeciesDataset(
            name="H2",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=mu_arr,
            kappa=kappa_arr,
            cp0_over_R=cp0_over_R,
            dhf_over_R=dhf_over_R,
            s0_over_R=s0_over_R,
            Tmid=target_Tmid,  # Different Tmid
            cantera_nasa7=cantera_nasa7,
            is_nasa7=True,
        )

        result = sd.fit_thermo(verbose=False)

        # Should refit with new Tmid
        self.assertIsNot(result, cantera_nasa7)
        self.assertEqual(sd.nasa7.Tmid, target_Tmid)
        self.assertNotEqual(sd.nasa7.Tmid, ct_Tmid)

    def test_force_refit_skips_cantera_coeffs(self):
        """force_refit=True skips reuse check and always refits."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        Tmid = 1000.0
        T = np.linspace(300, 3000, 50)

        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        i = gas.species_index("O2")
        sp_obj = gas.species(i)

        # Extract Cantera NASA7 coefficients
        coeffs = sp_obj.thermo.coeffs
        ct_Tmid = float(coeffs[0])
        ct_c_hi = np.array(coeffs[1:8])
        ct_c_lo = np.array(coeffs[8:15])
        cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

        # Evaluate data
        R = ct.gas_constant
        p0 = ct.one_atm
        cp_arr = np.zeros(len(T))
        h_arr = np.zeros(len(T))
        s_arr = np.zeros(len(T))
        mu_arr = np.zeros(len(T))
        kappa_arr = np.zeros(len(T))

        for j, Tj in enumerate(T):
            gas.TPX = Tj, p0, "O2:1.0"
            cp_arr[j] = gas.cp_mole
            h_arr[j] = gas.enthalpy_mole
            s_arr[j] = gas.entropy_mole
            mu_arr[j] = gas.viscosity
            kappa_arr[j] = gas.thermal_conductivity

        cp0_over_R = sp_obj.thermo.cp(298.15) / R
        dhf_over_R = sp_obj.thermo.h(298.15) / R
        s0_over_R = sp_obj.thermo.s(298.15) / R

        sd = SpeciesDataset(
            name="O2",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=mu_arr,
            kappa=kappa_arr,
            cp0_over_R=cp0_over_R,
            dhf_over_R=dhf_over_R,
            s0_over_R=s0_over_R,
            Tmid=ct_Tmid,
            cantera_nasa7=cantera_nasa7,
            is_nasa7=True,
        )

        # Fit with force_refit=True
        result = sd.fit_thermo(force_refit=True, verbose=False)

        # Should NOT reuse (even though Tmid matches)
        self.assertIsNot(result, cantera_nasa7)

    def test_raises_on_inconsistent_cantera_coeffs(self):
        """Raise RuntimeError when Cantera coefficients are inconsistent."""
        T = np.linspace(300, 3000, 50)
        Tmid = 1000.0

        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        # Create bad NASA7 coefficients (all zeros)
        bad_coeffs_lo = np.zeros(7)
        bad_coeffs_hi = np.zeros(7)
        bad_nasa7 = NASA7Polynomial(bad_coeffs_lo, bad_coeffs_hi, Tmid)

        # Create realistic data that won't match bad coefficients
        cp_arr = 30000.0 * np.ones(len(T))  # J/kmol/K
        h_arr = 30000.0 * T  # J/kmol
        s_arr = 200000.0 * np.ones(len(T))  # J/kmol/K

        sd = SpeciesDataset(
            name="BAD",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=np.ones(len(T)) * 1e-5,
            kappa=np.ones(len(T)) * 0.02,
            cp0_over_R=3.5,
            dhf_over_R=0.0,
            s0_over_R=20.0,
            Tmid=Tmid,
            cantera_nasa7=bad_nasa7,
            is_nasa7=True,
        )

        # Should raise RuntimeError for inconsistent coefficients
        with self.assertRaises(RuntimeError) as cm:
            sd.fit_thermo(verbose=False)

        self.assertIn("INCONSISTENT", str(cm.exception))
        self.assertIn("BAD", str(cm.exception))

    def test_custom_tolerances_accepted(self):
        """Custom FittingTolerances are respected."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        Tmid = 1000.0
        T = np.linspace(300, 3000, 50)

        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        i = gas.species_index("H2")
        sp_obj = gas.species(i)

        coeffs = sp_obj.thermo.coeffs
        ct_Tmid = float(coeffs[0])
        ct_c_hi = np.array(coeffs[1:8])
        ct_c_lo = np.array(coeffs[8:15])
        cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

        # Evaluate data
        R = ct.gas_constant
        p0 = ct.one_atm
        cp_arr = np.zeros(len(T))
        h_arr = np.zeros(len(T))
        s_arr = np.zeros(len(T))
        mu_arr = np.zeros(len(T))
        kappa_arr = np.zeros(len(T))

        for j, Tj in enumerate(T):
            gas.TPX = Tj, p0, "H2:1.0"
            cp_arr[j] = gas.cp_mole
            h_arr[j] = gas.enthalpy_mole
            s_arr[j] = gas.entropy_mole
            mu_arr[j] = gas.viscosity
            kappa_arr[j] = gas.thermal_conductivity

        cp0_over_R = sp_obj.thermo.cp(298.15) / R
        dhf_over_R = sp_obj.thermo.h(298.15) / R
        s0_over_R = sp_obj.thermo.s(298.15) / R

        sd = SpeciesDataset(
            name="H2",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=mu_arr,
            kappa=kappa_arr,
            cp0_over_R=cp0_over_R,
            dhf_over_R=dhf_over_R,
            s0_over_R=s0_over_R,
            Tmid=ct_Tmid,
            cantera_nasa7=cantera_nasa7,
            is_nasa7=True,
        )

        # Use custom strict tolerances
        strict_tol = FittingTolerances.strict()
        result = sd.fit_thermo(tolerances=strict_tol, verbose=False)

        # Should still work with strict tolerances
        self.assertIsNotNone(result)

    def test_non_nasa7_format_triggers_warning(self):
        """Non-NASA7 format prints warning and uses full refit."""
        T = np.linspace(300, 3000, 50)
        Tmid = 1000.0

        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        cp_arr = 30000.0 * np.ones(len(T))
        h_arr = 30000.0 * T
        s_arr = 200000.0 * np.ones(len(T))

        sd = SpeciesDataset(
            name="NASA9_SPECIES",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=np.ones(len(T)) * 1e-5,
            kappa=np.ones(len(T)) * 0.02,
            cp0_over_R=3.5,
            dhf_over_R=0.0,
            s0_over_R=20.0,
            Tmid=Tmid,
            cantera_nasa7=None,
            is_nasa7=False,  # Non-NASA7 format
        )

        # Capture stdout to verify warning
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        try:
            result = sd.fit_thermo(verbose=False)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Should print warning
        self.assertIn("Warning", output)
        self.assertIn("non-NASA7", output)
        self.assertIn("NASA9_SPECIES", output)

        # Should still fit successfully
        self.assertIsNotNone(result)

    def test_verbose_parameter_prints_decisions(self):
        """verbose=True prints fitting decisions."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        Tmid = 1000.0
        T = np.linspace(300, 3000, 50)

        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        i = gas.species_index("H2O")
        sp_obj = gas.species(i)

        coeffs = sp_obj.thermo.coeffs
        ct_Tmid = float(coeffs[0])
        ct_c_hi = np.array(coeffs[1:8])
        ct_c_lo = np.array(coeffs[8:15])
        cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

        # Evaluate data
        R = ct.gas_constant
        p0 = ct.one_atm
        cp_arr = np.zeros(len(T))
        h_arr = np.zeros(len(T))
        s_arr = np.zeros(len(T))
        mu_arr = np.zeros(len(T))
        kappa_arr = np.zeros(len(T))

        for j, Tj in enumerate(T):
            gas.TPX = Tj, p0, "H2O:1.0"
            cp_arr[j] = gas.cp_mole
            h_arr[j] = gas.enthalpy_mole
            s_arr[j] = gas.entropy_mole
            mu_arr[j] = gas.viscosity
            kappa_arr[j] = gas.thermal_conductivity

        cp0_over_R = sp_obj.thermo.cp(298.15) / R
        dhf_over_R = sp_obj.thermo.h(298.15) / R
        s0_over_R = sp_obj.thermo.s(298.15) / R

        sd = SpeciesDataset(
            name="H2O",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=mu_arr,
            kappa=kappa_arr,
            cp0_over_R=cp0_over_R,
            dhf_over_R=dhf_over_R,
            s0_over_R=s0_over_R,
            Tmid=ct_Tmid,
            cantera_nasa7=cantera_nasa7,
            is_nasa7=True,
        )

        # Capture stdout
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        try:
            sd.fit_thermo(verbose=True)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Should print decision message
        self.assertIn("H2O", output)
        self.assertTrue(
            "Reusing" in output or "Fitting" in output or "Refitting" in output
        )

    def test_reuse_count_in_fit_all_result(self):
        """fit_all() returns reuse count in result dict."""
        mech = MechanismDataset.from_cantera("h2o2.yaml", Tmid=1000.0)
        result = mech.fit_all(verbose=False)

        # Check that reused key exists
        self.assertIn("reused", result)
        self.assertIn("succeeded", result)
        self.assertIn("failed", result)

        # Reuse count should be >= 0 and <= total species
        self.assertGreaterEqual(result["reused"], 0)
        self.assertLessEqual(result["reused"], len(mech.species_datasets))

        # For h2o2 mechanism with matching Tmid, should have high reuse rate
        total = len(mech.species_datasets)
        reuse_percent = 100 * result["reused"] / total if total > 0 else 0
        # Expect at least 50% reuse for h2o2 mechanism
        self.assertGreater(reuse_percent, 50.0)

    def test_strategy_overrides_reuse_logic(self):
        """Explicit strategy bypasses reuse logic."""
        import cantera as ct

        gas = ct.Solution("h2o2.yaml")
        Tmid = 1000.0
        T = np.linspace(300, 3000, 50)

        if Tmid not in T:
            T = np.sort(np.append(T, Tmid))

        i = gas.species_index("H2")
        sp_obj = gas.species(i)

        coeffs = sp_obj.thermo.coeffs
        ct_Tmid = float(coeffs[0])
        ct_c_hi = np.array(coeffs[1:8])
        ct_c_lo = np.array(coeffs[8:15])
        cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

        # Evaluate data
        R = ct.gas_constant
        p0 = ct.one_atm
        cp_arr = np.zeros(len(T))
        h_arr = np.zeros(len(T))
        s_arr = np.zeros(len(T))
        mu_arr = np.zeros(len(T))
        kappa_arr = np.zeros(len(T))

        for j, Tj in enumerate(T):
            gas.TPX = Tj, p0, "H2:1.0"
            cp_arr[j] = gas.cp_mole
            h_arr[j] = gas.enthalpy_mole
            s_arr[j] = gas.entropy_mole
            mu_arr[j] = gas.viscosity
            kappa_arr[j] = gas.thermal_conductivity

        cp0_over_R = sp_obj.thermo.cp(298.15) / R
        dhf_over_R = sp_obj.thermo.h(298.15) / R
        s0_over_R = sp_obj.thermo.s(298.15) / R

        sd = SpeciesDataset(
            name="H2",
            T=T,
            cp=cp_arr,
            h=h_arr,
            s=s_arr,
            mu=mu_arr,
            kappa=kappa_arr,
            cp0_over_R=cp0_over_R,
            dhf_over_R=dhf_over_R,
            s0_over_R=s0_over_R,
            Tmid=ct_Tmid,
            cantera_nasa7=cantera_nasa7,
            is_nasa7=True,
        )

        # Use explicit strategy='full' (should skip reuse logic)
        result = sd.fit_thermo(strategy="full", verbose=False)

        # Should NOT reuse (even though Tmid matches and coeffs are valid)
        self.assertIsNot(result, cantera_nasa7)


if __name__ == "__main__":
    unittest.main()
