import sys
import os
import unittest
import tempfile
from pathlib import Path

from ct2foam.scripts import mech2foam
from ct2foam.thermo_transport import ct_properties


class TestMech2foamWorkflow(unittest.TestCase):
    """Test mech2foam end-to-end workflows."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mech2foam_h2o2_complete_workflow_creates_files(self):
        """Test mech2foam produces all required output files."""

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(self.output_dir),
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        self.assertTrue((self.output_dir / "species.foam").exists())
        self.assertTrue((self.output_dir / "thermo.foam").exists())
        self.assertTrue((self.output_dir / "reactions.foam").exists())
        self.assertTrue((self.output_dir / "log.txt").exists())

    def test_mech2foam_gri30_large_mechanism_completes(self):
        """Test mech2foam processes successfully."""

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "gri30.yaml",
                "--output",
                str(self.output_dir),
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        self.assertTrue((self.output_dir / "species.foam").exists())
        self.assertTrue((self.output_dir / "thermo.foam").exists())
        self.assertTrue((self.output_dir / "reactions.foam").exists())

    def test_mech2foam_output_file_validity_openfoam_format(self):
        """Test thermo.foam has valid OpenFOAM dictionary format."""

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(self.output_dir),
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        thermo_file = self.output_dir / "thermo.foam"
        content = thermo_file.read_text()

        # Check for OpenFOAM syntax elements
        self.assertIn("{", content)
        self.assertIn("}", content)
        # Count braces to verify balance
        open_count = content.count("{")
        close_count = content.count("}")
        self.assertEqual(open_count, close_count)

    def test_mech2foam_creates_output_directory_if_missing(self):
        """Test output directory is created automatically if missing."""
        new_output_dir = self.output_dir / "output"

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(new_output_dir),
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        # Check directory was created
        self.assertTrue(new_output_dir.exists())
        self.assertTrue((new_output_dir / "species.foam").exists())


class TestMech2foamArguments(unittest.TestCase):
    """Test mech2foam command-line argument processing."""

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_mech2foam_custom_Tmid_parameter_respected(self):
        """Test --Tmid parameter is used in fitting."""
        tmid_value = 1500.0

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(self.output_dir),
                "--Tmid",
                str(tmid_value),
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        self.assertTrue((self.output_dir / "thermo.foam").exists())

    def test_mech2foam_custom_temperature_bounds_applied(self):
        """Test --Tlow and --Thigh parameters are respected."""

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(self.output_dir),
                "--Tlow",
                "250",
                "--Thigh",
                "3500",
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        self.assertTrue((self.output_dir / "thermo.foam").exists())

    def test_mech2foam_plot_flag_generates_figures_directory(self):
        """Test --plot flag creates Figures directory."""

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(self.output_dir),
                "--plot",
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

        figures_dir = self.output_dir / "Figures"
        self.assertTrue(figures_dir.exists())

        png_files = list(figures_dir.glob("*.png"))
        self.assertGreater(len(png_files), 0)

    def test_mech2foam_default_output_directory_current_working(self):
        """Test default output directory is current working directory."""

        original_argv = sys.argv
        original_cwd = os.getcwd()

        try:
            os.chdir(str(self.output_dir))
            sys.argv = ["mech2foam", "--input", "h2o2.yaml"]
            mech2foam.main()
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)

        self.assertTrue((self.output_dir / "thermo.foam").exists())


class TestMech2foamOutputValidation(unittest.TestCase):
    """Test mech2foam output file validation."""

    def setUp(self):
        """Create temporary directory and generate output."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        original_argv = sys.argv
        try:
            sys.argv = [
                "mech2foam",
                "--input",
                "h2o2.yaml",
                "--output",
                str(self.output_dir),
            ]
            mech2foam.main()
        finally:
            sys.argv = original_argv

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_mech2foam_all_species_in_output_files(self):
        """Test all mechanism species are present in output."""
        mech = ct_properties.ctThermoTransport("h2o2.yaml", verbose=False)
        expected_species = set(mech.names)

        species_file = self.output_dir / "species.foam"
        content = species_file.read_text()

        for species in expected_species:
            self.assertIn(species, content)


if __name__ == "__main__":
    unittest.main()
