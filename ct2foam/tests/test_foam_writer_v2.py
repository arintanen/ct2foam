"""Test suite comparing v2 foam_writer output with old implementation.

This test verifies that the v2 foam_writer produces identical output
to the original implementation in ct2foam.thermo_transport.foam_writer.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
import difflib

from ct2foam.v2 import foam_writer as new_writer
from ct2foam.thermo_transport import foam_writer as old_writer


class TestFoamWriterEquivalence(unittest.TestCase):
    """Test that v2 foam_writer produces identical output to old implementation."""

    def setUp(self):
        """Create temporary directories for output files."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_dir = Path(self.temp_dir) / "old"
        self.new_dir = Path(self.temp_dir) / "new"
        self.old_dir.mkdir()
        self.new_dir.mkdir()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def read_file(self, filepath):
        """Read file and return lines."""
        with open(filepath, "r") as f:
            return f.readlines()

    def assert_files_identical(self, old_file, new_file, msg=None):
        """Assert two files have identical content."""
        old_lines = self.read_file(old_file)
        new_lines = self.read_file(new_file)

        if old_lines != new_lines:
            diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=str(old_file),
                tofile=str(new_file),
                lineterm="",
            )
            diff_text = "\n".join(diff)
            fail_msg = f"Files differ:\n{diff_text}"
            if msg:
                fail_msg = f"{msg}\n{fail_msg}"
            self.fail(fail_msg)

    def test_write_species_list_identical(self):
        """Test write_species_list produces identical output."""
        species_names = ["H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2"]

        old_file = self.old_dir / "species.foam"
        new_file = self.new_dir / "species.foam"

        old_writer.write_species_list(old_file, species_names)
        new_writer.write_species_list(new_file, species_names)

        self.assert_files_identical(
            old_file, new_file, "write_species_list output differs"
        )

    def test_write_reactions_identical(self):
        """Test write_reactions produces identical output."""
        old_file = self.old_dir / "reactions.foam"
        new_file = self.new_dir / "reactions.foam"

        old_writer.write_reactions(old_file)
        new_writer.write_reactions(new_file)

        self.assert_files_identical(
            old_file, new_file, "write_reactions output differs"
        )

    def test_write_thermo_transport_simple(self):
        """Test write_thermo_transport with simple data."""
        name = "H2"
        MW = 2.016
        As = 1.67212e-06
        Ts = 170.672

        # Simple polynomial coefficients (order 3 = 4 coefficients)
        poly_mu = np.array([1.0e-5, 2.0e-8, 3.0e-11, 4.0e-14])
        poly_kappa = np.array([5.0e-3, 6.0e-6, 7.0e-9, 8.0e-12])
        logpoly_mu = np.array([1.1e-5, 2.1e-8, 3.1e-11, 4.1e-14])
        logpoly_kappa = np.array([5.1e-3, 6.1e-6, 7.1e-9, 8.1e-12])

        # NASA7 coefficients
        nasa7_Tmid = 1000.0
        nasa7_Tlo = 300.0
        nasa7_Thi = 3000.0
        nasa7_lo = np.array(
            [
                2.34433112,
                0.00798052075,
                -1.9478151e-05,
                2.01572094e-08,
                -7.37611761e-12,
                -917.935173,
                0.683010238,
            ]
        )
        nasa7_hi = np.array(
            [
                3.3372792,
                -4.94024731e-05,
                4.99456778e-07,
                -1.79566394e-10,
                2.00255376e-14,
                -950.158922,
                -3.20502331,
            ]
        )

        old_file = self.old_dir / "thermo.foam"
        new_file = self.new_dir / "thermo.foam"

        old_writer.write_thermo_transport(
            old_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
        )

        new_writer.write_thermo_transport(
            new_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
        )

        self.assert_files_identical(
            old_file, new_file, "write_thermo_transport output differs for simple case"
        )

    def test_write_thermo_transport_with_elements(self):
        """Test write_thermo_transport with elements dictionary."""
        name = "CH4"
        MW = 16.043
        As = 1.2e-06
        Ts = 150.0

        poly_mu = np.array([1.0e-5, 2.0e-8, 3.0e-11, 4.0e-14])
        poly_kappa = np.array([5.0e-3, 6.0e-6, 7.0e-9, 8.0e-12])
        logpoly_mu = np.array([1.1e-5, 2.1e-8, 3.1e-11, 4.1e-14])
        logpoly_kappa = np.array([5.1e-3, 6.1e-6, 7.1e-9, 8.1e-12])

        nasa7_Tmid = 1000.0
        nasa7_Tlo = 300.0
        nasa7_Thi = 3000.0
        nasa7_lo = np.array(
            [
                5.14987613,
                -0.0136709788,
                4.91800599e-05,
                -4.84743026e-08,
                1.66693956e-11,
                -10246.6476,
                -4.64130376,
            ]
        )
        nasa7_hi = np.array(
            [
                0.074851495,
                0.0133909467,
                -5.73285809e-06,
                1.22292535e-09,
                -1.0181523e-13,
                -9468.34459,
                18.437318,
            ]
        )

        elements = {"C": 1, "H": 4}

        old_file = self.old_dir / "thermo_elem.foam"
        new_file = self.new_dir / "thermo_elem.foam"

        old_writer.write_thermo_transport(
            old_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
            elements=elements,
        )

        new_writer.write_thermo_transport(
            new_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
            elements=elements,
        )

        self.assert_files_identical(
            old_file, new_file, "write_thermo_transport output differs with elements"
        )

    def test_write_thermo_transport_multiple_species(self):
        """Test writing multiple species to same file (append mode)."""
        species_data = [
            {
                "name": "H2",
                "MW": 2.016,
                "As": 1.67212e-06,
                "Ts": 170.672,
            },
            {
                "name": "O2",
                "MW": 31.999,
                "As": 2.0e-06,
                "Ts": 150.0,
            },
        ]

        # Common data
        poly_mu = np.array([1.0e-5, 2.0e-8, 3.0e-11, 4.0e-14])
        poly_kappa = np.array([5.0e-3, 6.0e-6, 7.0e-9, 8.0e-12])
        logpoly_mu = np.array([1.1e-5, 2.1e-8, 3.1e-11, 4.1e-14])
        logpoly_kappa = np.array([5.1e-3, 6.1e-6, 7.1e-9, 8.1e-12])
        nasa7_Tmid = 1000.0
        nasa7_Tlo = 300.0
        nasa7_Thi = 3000.0
        nasa7_lo = np.array([2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -0.446682])
        nasa7_hi = np.array([2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -0.446682])

        old_file = self.old_dir / "thermo_multi.foam"
        new_file = self.new_dir / "thermo_multi.foam"

        # Write multiple species
        for sp in species_data:
            old_writer.write_thermo_transport(
                old_file,
                sp["name"],
                sp["MW"],
                sp["As"],
                sp["Ts"],
                poly_mu,
                poly_kappa,
                logpoly_mu,
                logpoly_kappa,
                nasa7_Tmid,
                nasa7_Tlo,
                nasa7_Thi,
                nasa7_lo,
                nasa7_hi,
            )

            new_writer.write_thermo_transport(
                new_file,
                sp["name"],
                sp["MW"],
                sp["As"],
                sp["Ts"],
                poly_mu,
                poly_kappa,
                logpoly_mu,
                logpoly_kappa,
                nasa7_Tmid,
                nasa7_Tlo,
                nasa7_Thi,
                nasa7_lo,
                nasa7_hi,
            )

        self.assert_files_identical(
            old_file,
            new_file,
            "write_thermo_transport output differs for multiple species",
        )

    def test_polynomial_coefficient_reversal(self):
        """Test that polynomial coefficients are properly reversed for OpenFOAM."""
        # This tests the numpy flip operation that reverses coefficient order
        name = "TEST"
        MW = 10.0
        As = 1.0e-06
        Ts = 100.0

        # Use distinct coefficients to verify reversal
        poly_mu = np.array([1.0, 2.0, 3.0, 4.0])  # Should be reversed to [4,3,2,1]
        poly_kappa = np.array([5.0, 6.0, 7.0, 8.0])  # Should be reversed to [8,7,6,5]
        logpoly_mu = np.array([10.0, 20.0, 30.0])  # Should be reversed and padded
        logpoly_kappa = np.array([50.0, 60.0])  # Should be reversed and padded

        nasa7_Tmid = 1000.0
        nasa7_Tlo = 300.0
        nasa7_Thi = 3000.0
        nasa7_lo = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        nasa7_hi = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        old_file = self.old_dir / "test_reversal.foam"
        new_file = self.new_dir / "test_reversal.foam"

        old_writer.write_thermo_transport(
            old_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
        )

        new_writer.write_thermo_transport(
            new_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
        )

        self.assert_files_identical(
            old_file, new_file, "Polynomial coefficient reversal differs"
        )

        # Also verify the content has expected reversal
        content = self.read_file(new_file)
        content_str = "".join(content)

        # Check that reversed coefficients appear in output
        self.assertIn(
            "4.0 3.0 2.0 1.0", content_str, "poly_mu should be reversed [4,3,2,1]"
        )
        self.assertIn(
            "8.0 7.0 6.0 5.0", content_str, "poly_kappa should be reversed [8,7,6,5]"
        )


class TestFoamWriterAPI(unittest.TestCase):
    """Test the foam_writer API directly."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_species_list_format(self):
        """Test species list has correct OpenFOAM format."""
        species_names = ["H2", "O2", "H2O"]
        output_file = self.output_dir / "species.foam"

        new_writer.write_species_list(output_file, species_names)

        with open(output_file, "r") as f:
            content = f.read()

        self.assertIn("species\n(", content)
        self.assertIn("\tH2\n", content)
        self.assertIn("\tO2\n", content)
        self.assertIn("\tH2O\n", content)
        self.assertIn(");\n", content)

    def test_reactions_format(self):
        """Test reactions file has correct format."""
        output_file = self.output_dir / "reactions.foam"

        new_writer.write_reactions(output_file)

        with open(output_file, "r") as f:
            content = f.read()

        self.assertIn("reactions", content)
        self.assertIn("{\n", content)
        self.assertIn("}\n", content)

    def test_thermo_transport_structure(self):
        """Test thermo transport output has all required sections."""
        name = "H2"
        MW = 2.016
        As = 1.67212e-06
        Ts = 170.672
        poly_mu = np.array([1.0e-5, 2.0e-8, 3.0e-11, 4.0e-14])
        poly_kappa = np.array([5.0e-3, 6.0e-6, 7.0e-9, 8.0e-12])
        logpoly_mu = np.array([1.1e-5, 2.1e-8, 3.1e-11, 4.1e-14])
        logpoly_kappa = np.array([5.1e-3, 6.1e-6, 7.1e-9, 8.1e-12])
        nasa7_Tmid = 1000.0
        nasa7_Tlo = 300.0
        nasa7_Thi = 3000.0
        nasa7_lo = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        nasa7_hi = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        output_file = self.output_dir / "thermo.foam"

        new_writer.write_thermo_transport(
            output_file,
            name,
            MW,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            nasa7_Tmid,
            nasa7_Tlo,
            nasa7_Thi,
            nasa7_lo,
            nasa7_hi,
        )

        with open(output_file, "r") as f:
            content = f.read()

        # Check main sections exist
        self.assertIn(f"{name}\n{{", content)
        self.assertIn("specie", content)
        self.assertIn("thermodynamics", content)
        self.assertIn("transport", content)

        # Check specie section
        self.assertIn("molWeight", content)
        self.assertIn(str(MW), content)

        # Check thermodynamics section
        self.assertIn("Tlow", content)
        self.assertIn("Thigh", content)
        self.assertIn("Tcommon", content)
        self.assertIn("lowCpCoeffs", content)
        self.assertIn("highCpCoeffs", content)

        # Check transport section
        self.assertIn("As", content)
        self.assertIn("Ts", content)
        self.assertIn("muLogCoeffs<8>", content)
        self.assertIn("muCoeffs<8>", content)
        self.assertIn("kappaLogCoeffs<8>", content)
        self.assertIn("kappaCoeffs<8>", content)


if __name__ == "__main__":
    unittest.main()
