from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from ct2foam.species import SpeciesList


def write_species_list(file_name: Path, species_names: list[str]):
    """
    Writes a species.foam file with an OpenFOAM formatted list of species.
    """
    with open(file_name, "a") as output:
        output.write("species\n(\n")
        for sp_i in species_names:
            output.write("\t")
            output.write(sp_i)
            output.write("\n")
        output.write(");\n\n")


def write_reactions(file_name: Path):
    """
    Writes an empty reactions.foam file with an OpenFOAM formatted list of 0 reactions.
    """
    with open(file_name, "a") as output:
        output.write("// - This dictionary intentionally left blank.\n")
        output.write("reactions\n{\n")
        output.write("}\n\n")


def write_thermo_transport(
    file_name: Path,
    name: str,
    MW: float,
    As: float,
    Ts: float,
    poly_mu: np.ndarray,
    poly_kappa: np.ndarray,
    logpoly_mu: np.ndarray,
    logpoly_kappa: np.ndarray,
    nasa7_Tmid: float,
    nasa7_Tlo: float,
    nasa7_Thi: float,
    nasa7_lo: np.ndarray,
    nasa7_hi: np.ndarray,
    elements: Optional[dict] = None,
):
    """Write thermophysicalProperties block for one species or mixture.

    Args:
        file_name: output file path (opened in append mode)
        name: species or mixture name used as the dict key
        MW: molecular weight [kg/kmol]
        As: Sutherland coefficient [Pa·s/K^0.5]
        Ts: Sutherland temperature [K]
        poly_mu: polynomial viscosity coefficients (descending order)
        poly_kappa: polynomial thermal conductivity coefficients (descending order)
        logpoly_mu: log-polynomial viscosity coefficients (descending order)
        logpoly_kappa: log-polynomial thermal conductivity coefficients
                       (descending order)
        nasa7_Tmid: NASA7 common mid-point temperature [K]
        nasa7_Tlo: lower temperature bound [K]
        nasa7_Thi: upper temperature bound [K]
        nasa7_lo: 7 NASA7 coefficients for the low-temperature range
        nasa7_hi: 7 NASA7 coefficients for the high-temperature range
        elements: elemental composition dict (e.g. ``{"C": 1, "H": 4}``)
    """
    # Numpy-based polynomial fit has a reversed order to OpenFoam dictionary definition
    poly_mu_rev = np.copy(poly_mu)
    poly_mu_rev = np.flip(poly_mu_rev)
    logpoly_mu_rev = np.copy(logpoly_mu)
    logpoly_mu_rev = np.flip(logpoly_mu_rev)

    poly_kappa_rev = np.copy(poly_kappa)
    poly_kappa_rev = np.flip(poly_kappa_rev)
    logpoly_kappa_rev = np.copy(logpoly_kappa)
    logpoly_kappa_rev = np.flip(logpoly_kappa_rev)

    # Write the thermo output in openfoam format
    with open(file_name, "a") as output:

        output.write(name + "\n{\n")
        output.write("\t")
        #############################################################################
        output.write("specie\n\t{\n")
        output.write("\t\tnMoles \t 1;\n")
        output.write("\t\tmolWeight \t" + str(MW) + ";")
        output.write("\n\t}\n\n")
        #############################################################################

        #############################################################################
        output.write("\tthermodynamics")
        #############################################################################
        output.write("\n\t{\n")
        output.write("\t\tTlow\t\t" + str(nasa7_Tlo) + ";\n")
        output.write("\t\tThigh\t\t" + str(nasa7_Thi) + ";\n")
        output.write("\t\tTcommon\t\t" + str(nasa7_Tmid) + ";\n")
        output.write("\t\tlowCpCoeffs\t(\t")
        for wi in range(0, 7):  # NASA pol has 7 coeffs
            output.write(str(nasa7_lo[wi]))
            output.write(" ")
        output.write(" );\n")
        output.write("\t\thighCpCoeffs\t(\t")
        for wi in range(0, 7):  # NASA pol has 7 coeffs
            output.write(str(nasa7_hi[wi]))
            output.write(" ")
        output.write(" );\n")
        output.write("\t}\n\n")

        #############################################################################
        output.write("\ttransport \n\t{\n")
        #############################################################################

        output.write("\t\tAs\t" + str(As) + ";\n")
        output.write("\t\tTs\t" + str(Ts) + ";\n")

        output.write("\t\tmuLogCoeffs<8>\t(\t")
        for wi in range(8):
            if wi < len(logpoly_mu_rev):
                output.write(str(logpoly_mu_rev[wi]))
            else:
                output.write("0")
            output.write(" ")
        output.write(" );\n")

        output.write("\t\tmuCoeffs<8>\t(\t")
        for wi in range(8):
            if wi < len(poly_mu_rev):
                output.write(str(poly_mu_rev[wi]))
            else:
                output.write("0")
            output.write(" ")
        output.write(" );\n")
        #############################################################################
        output.write("\t\tkappaLogCoeffs<8>\t(\t")
        for wi in range(8):
            if wi < len(logpoly_kappa_rev):
                output.write(str(logpoly_kappa_rev[wi]))
            else:
                output.write("0")
            output.write(" ")
        output.write(" );\n")
        output.write("\t\tkappaCoeffs<8>\t(\t")
        for wi in range(8):
            if wi < len(poly_kappa_rev):
                output.write(str(poly_kappa_rev[wi]))
            else:
                output.write("0")
            output.write(" ")
        output.write(" );\n")
        output.write("\t}\n\n")
        #############################################################################

        #############################################################################
        if elements is not None:
            output.write("\telements")
            #############################################################################
            output.write("\n\t{\n")
            # Variables for the elemental composition entry:
            for elem_i in elements.keys():
                output.write(
                    "\t\t" + elem_i + "\t" + str(int(elements[elem_i])) + ";\n"
                )
            output.write("\t}\n")
            #############################################################################
        output.write("}\n\n")


def write_foam(species: "SpeciesList", output_dir: Path):
    """Write OpenFOAM output files using foam_writer."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thermo_file = output_dir / "thermo.foam"
    reactions_file = output_dir / "reactions.foam"
    species_file = output_dir / "species.foam"

    # Remove existing files
    thermo_file.unlink(missing_ok=True)
    reactions_file.unlink(missing_ok=True)
    species_file.unlink(missing_ok=True)

    write_reactions(reactions_file)

    names = [sp.name for sp in species if sp.nasa7 is not None]
    write_species_list(species_file, names)

    for sp in species:
        if not sp.nasa7:
            continue

        poly_mu = sp.polynomial.coeffs_mu if sp.polynomial else np.zeros(4)
        poly_kappa = sp.polynomial.coeffs_kappa if sp.polynomial else np.zeros(4)
        logpoly_mu = sp.log_polynomial.coeffs_mu if sp.log_polynomial else np.zeros(4)
        logpoly_kappa = (
            sp.log_polynomial.coeffs_kappa if sp.log_polynomial else np.zeros(4)
        )
        As = sp.sutherland.As if sp.sutherland else 0.0
        Ts = sp.sutherland.Ts if sp.sutherland else 0.0

        write_thermo_transport(
            thermo_file,
            sp.name,
            sp.W,
            As,
            Ts,
            poly_mu,
            poly_kappa,
            logpoly_mu,
            logpoly_kappa,
            sp.nasa7.Tmid,
            sp.nasa7.Tlow,
            sp.nasa7.Tmax,
            sp.nasa7.coeffs_low,
            sp.nasa7.coeffs_high,
            elements=sp.elements,
        )
