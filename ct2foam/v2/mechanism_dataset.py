"""MechanismDataset orchestrator - manages Species objects and Cantera data."""

import numpy as np
from pathlib import Path
import cantera as ct

from .species import Species
from .nasa7 import NASA7Polynomial
from .sutherland import Sutherland
from .polynomial import Polynomial
from .fitting_tolerances import FittingTolerances
from ct2foam.thermo_transport import foam_writer as writer


class MechanismDataset:
    """Orchestrator that creates and manages Species objects for a mechanism."""

    def __init__(self, species_list, mechanism_name, Tmid, Tlow, Thigh):
        """Initialize MechanismDataset.

        Args:
            species_list: List of Species objects
            mechanism_name: Name of the mechanism
            Tmid: Midpoint temperature
            Tlow: Low temperature limit
            Thigh: High temperature limit
        """
        self.species_list = list(species_list)
        self.mechanism_name = str(mechanism_name)
        self.Tmid = float(Tmid)
        self.Tlow = float(Tlow)
        self.Thigh = float(Thigh)
        self.fit_log = []
        self.failed_species = {}

    @classmethod
    def from_cantera(
        cls,
        mech_file,
        Tmid,
        Tlow=None,
        Thigh=None,
        T_eval=None,
        tolerances=None,
        force_refit=False,
        verbose=True,
    ):
        """Create a MechanismDataset by loading and fitting a Cantera mechanism.

        This method:
        1. Loads the Cantera mechanism
        2. Evaluates thermodynamic and transport data at T_eval
        3. Decides whether to reuse Cantera NASA7 coefficients or refit
        4. Fits transport properties
        5. Creates Species objects with fitted coefficients

        Args:
            mech_file: Path to Cantera mechanism file
            Tmid: Midpoint temperature (K)
            Tlow: Low temperature limit (default 200K)
            Thigh: High temperature limit (default 5000K)
            T_eval: Temperature evaluation array (default: linspace(Tlow, Thigh, 100))
            tolerances: FittingTolerances instance (default: FittingTolerances.default())
            force_refit: If True, always refit (skip Cantera coefficient reuse)
            verbose: If True, print fitting messages

        Returns:
            MechanismDataset instance with fitted Species objects
        """
        if tolerances is None:
            tolerances = FittingTolerances.default()

        gas = ct.Solution(str(mech_file))
        R = ct.gas_constant
        p0 = ct.one_atm

        if Tlow is None:
            Tlow = 200.0
        if Thigh is None:
            Thigh = 5000.0
        if T_eval is None:
            T_eval = np.linspace(Tlow, Thigh, 100)

        T_eval = np.sort(T_eval)
        # Insert Tmid into the T array if not already present
        idx = T_eval.searchsorted(Tmid)
        if idx >= len(T_eval) or T_eval[idx] != Tmid:
            T_eval = np.concatenate((T_eval[:idx], [Tmid], T_eval[idx:]))

        nT = len(T_eval)
        species_list = []
        fit_log = []
        failed_species = {}

        gas.transport_model = "multicomponent"

        succeeded = 0
        failed = 0
        reused = 0

        for sp_name in gas.species_names:
            i = gas.species_index(sp_name)
            sp_obj = gas.species(i)
            reactants = sp_name + ":1.0"

            try:
                # ==== STEP 1: Extract Cantera NASA7 coefficients (if available) ====
                cantera_nasa7, is_nasa7 = _extract_cantera_nasa7(sp_obj)

                # ==== STEP 2: Evaluate thermodynamic and transport data ====
                T, cp, h, s, mu, kappa, cv = _evaluate_cantera_data(
                    gas, sp_name, T_eval, p0
                )

                # ==== STEP 3: Get standard-state properties ====
                cp0_over_R = sp_obj.thermo.cp(298.15) / R
                dhf_over_R = sp_obj.thermo.h(298.15) / R
                s0_over_R = sp_obj.thermo.s(298.15) / R

                # ==== STEP 4: Get elemental composition ====
                elements = {}
                for elem in gas.element_names:
                    na = gas.n_atoms(sp_name, elem)
                    if na > 0:
                        elements[elem] = na

                # ==== STEP 5: Decide reuse vs. refit for thermodynamics ====
                can_reuse, reason = _can_reuse_cantera_coeffs(
                    cantera_nasa7,
                    is_nasa7,
                    T,
                    cp,
                    h,
                    s,
                    Tmid,
                    R,
                    tolerances,
                    force_refit,
                )

                if can_reuse:
                    # REUSE Cantera coefficients
                    nasa7 = cantera_nasa7
                    reused += 1
                    if verbose:
                        print(f"{sp_name}: Reused Cantera NASA7 coefficients")
                else:
                    # REFIT using NASA7Polynomial class methods
                    cp_over_R = cp / R
                    h_over_RT = h / (R * T)
                    s_over_R = s / R

                    if reason == "non_nasa7":
                        # Non-NASA7 format: use full refit
                        if verbose:
                            print(f"{sp_name}: Non-NASA7 format, using full refit")
                        nasa7 = NASA7Polynomial.fit_full(
                            T,
                            cp_over_R,
                            h_over_RT,
                            s_over_R,
                            cp0_over_R,
                            dhf_over_R,
                            s0_over_R,
                            Tmid,
                        )
                    elif reason == "tmid_differs":
                        # Continuous but different Tmid: cp-only refit
                        if verbose:
                            print(
                                f"{sp_name}: Tmid differs "
                                f"({cantera_nasa7.Tmid:.1f} → {Tmid:.1f}), "
                                "using cp-only refit"
                            )
                        nasa7 = NASA7Polynomial.fit_cp_only(
                            T, cp_over_R, cp0_over_R, dhf_over_R, s0_over_R, Tmid
                        )
                    elif reason == "not_continuous":
                        # Not continuous: full refit
                        if verbose:
                            print(f"{sp_name}: Not continuous, using full refit")
                        nasa7 = NASA7Polynomial.fit_full(
                            T,
                            cp_over_R,
                            h_over_RT,
                            s_over_R,
                            cp0_over_R,
                            dhf_over_R,
                            s0_over_R,
                            Tmid,
                        )
                    elif reason == "force_refit":
                        # Force refit requested
                        if verbose:
                            print(f"{sp_name}: Force refit, using auto strategy")
                        nasa7 = NASA7Polynomial.fit_auto(
                            T,
                            cp_over_R,
                            h_over_RT,
                            s_over_R,
                            cp0_over_R,
                            dhf_over_R,
                            s0_over_R,
                            Tmid,
                        )
                    else:
                        # Default: auto strategy
                        nasa7 = NASA7Polynomial.fit_auto(
                            T,
                            cp_over_R,
                            h_over_RT,
                            s_over_R,
                            cp0_over_R,
                            dhf_over_R,
                            s0_over_R,
                            Tmid,
                            verbose=verbose,
                        )

                # ==== STEP 6: Fit transport properties ====
                sutherland = Sutherland.fit(T, mu)
                polynomial = Polynomial.fit_polynomial(T, mu, kappa)
                log_polynomial = Polynomial.fit_log_polynomial(T, mu, kappa)

                # ==== STEP 7: Create Species object ====
                species = Species(
                    name=sp_name,
                    W=gas.molecular_weights[i],
                    cp0_over_R=cp0_over_R,
                    dhf_over_R=dhf_over_R,
                    s0_over_R=s0_over_R,
                    elements=elements,
                )

                # Assign fitted coefficients
                species.nasa7 = nasa7
                species.sutherland = sutherland
                species.polynomial = polynomial
                species.log_polynomial = log_polynomial

                # Check quality
                species.check_quality(T, cp, h, s)

                species_list.append(species)
                succeeded += 1
                fit_log.append(f"{sp_name}: fit OK")

            except Exception as e:
                failed += 1
                msg = (
                    f"{sp_name}: fitting failed — {e}. "
                    "Consider adjusting parameters or tune your tolerances."
                )
                failed_species[sp_name] = msg
                fit_log.append(msg)
                if verbose:
                    print(msg)

        # Create MechanismDataset instance
        mech = cls(species_list, str(mech_file), Tmid, T_eval[0], T_eval[-1])
        mech.fit_log = fit_log
        mech.failed_species = failed_species

        # Print summary
        if verbose:
            print(f"\nFitting summary:")
            print(f"  Succeeded: {succeeded}/{len(gas.species_names)}")
            print(f"  Failed: {failed}/{len(gas.species_names)}")
            print(f"  Reused Cantera coefficients: {reused}/{succeeded}")

        return mech

    def print_fit_summary(self):
        """Print a summary of the fitting process."""
        total = len(self.species_list) + len(self.failed_species)
        n_failed = len(self.failed_species)
        n_ok = len(self.species_list)
        print(f"Fit summary for {self.mechanism_name}:")
        print(f"  Total species: {total}")
        print(f"  Succeeded: {n_ok}")
        print(f"  Failed: {n_failed}")
        if n_failed > 0:
            print("  Failed species:")
            for name, msg in self.failed_species.items():
                print(f"    - {msg}")

    def write_output(self, output_dir):
        """Write OpenFOAM output files using foam_writer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        thermo_file = output_dir / "thermo.foam"
        reactions_file = output_dir / "reactions.foam"
        species_file = output_dir / "species.foam"

        # Remove existing files
        for f in (thermo_file, reactions_file, species_file):
            if f.exists():
                f.unlink()

        writer.write_reactions(reactions_file)

        names = [sp.name for sp in self.species_list if sp.nasa7 is not None]
        writer.write_species_list(species_file, names)

        for sp in self.species_list:
            if sp.nasa7 is None:
                continue

            poly_mu = sp.polynomial.coeffs_mu if sp.polynomial else np.zeros(4)
            poly_kappa = sp.polynomial.coeffs_kappa if sp.polynomial else np.zeros(4)
            logpoly_mu = (
                sp.log_polynomial.coeffs_mu if sp.log_polynomial else np.zeros(4)
            )
            logpoly_kappa = (
                sp.log_polynomial.coeffs_kappa if sp.log_polynomial else np.zeros(4)
            )
            As = sp.sutherland.As if sp.sutherland else 0.0
            Ts = sp.sutherland.Ts if sp.sutherland else 0.0

            writer.write_thermo_transport(
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
                self.Tlow,
                self.Thigh,
                sp.nasa7.coeffs_low,
                sp.nasa7.coeffs_high,
                elements=sp.elements,
            )


# ============================================================================ #
#  Helper Functions
# ============================================================================ #


def _extract_cantera_nasa7(sp_obj):
    """Extract NASA7Polynomial from Cantera species object.

    Args:
        sp_obj: Cantera Species object

    Returns:
        (NASA7Polynomial | None, bool): (nasa7_object, is_nasa7_format)
    """
    thermo_type = type(sp_obj.thermo).__name__
    is_nasa7 = thermo_type == "NasaPoly2"

    if not is_nasa7:
        return None, False

    # Extract coefficients: [Tmid, c_hi[0..6], c_lo[0..6]]
    coeffs = sp_obj.thermo.coeffs
    ct_Tmid = float(coeffs[0])
    ct_c_hi = np.array(coeffs[1:8])
    ct_c_lo = np.array(coeffs[8:15])

    return NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid), True


def _evaluate_cantera_data(gas, sp_name, T_eval, p0):
    """Evaluate thermodynamic and transport properties at T_eval.

    Args:
        gas: Cantera Solution object
        sp_name: Species name
        T_eval: Temperature evaluation array
        p0: Pressure (Pa)

    Returns:
        (T, cp, h, s, mu, kappa, cv): Arrays of evaluated properties
    """
    nT = len(T_eval)
    reactants = sp_name + ":1.0"

    mu_arr = np.zeros(nT)
    kappa_arr = np.zeros(nT)
    cp_arr = np.zeros(nT)
    h_arr = np.zeros(nT)
    s_arr = np.zeros(nT)
    cv_arr = np.zeros(nT)

    for j in range(nT):
        gas.TPX = T_eval[j], p0, reactants
        mu_arr[j] = gas.viscosity
        kappa_arr[j] = gas.thermal_conductivity
        cp_arr[j] = gas.cp_mole
        h_arr[j] = gas.enthalpy_mole
        s_arr[j] = gas.entropy_mole
        cv_arr[j] = gas.cv_mole

    return T_eval, cp_arr, h_arr, s_arr, mu_arr, kappa_arr, cv_arr


def _can_reuse_cantera_coeffs(
    cantera_nasa7,
    is_nasa7,
    T,
    cp,
    h,
    s,
    Tmid,
    R,
    tolerances,
    force_refit,
):
    """Decide if Cantera NASA7 coefficients can be reused.

    Args:
        cantera_nasa7: NASA7Polynomial object from Cantera (or None)
        is_nasa7: Whether original format is NASA7
        T: Temperature array
        cp: Specific heat array
        h: Enthalpy array
        s: Entropy array
        Tmid: Target midpoint temperature
        R: Gas constant
        tolerances: FittingTolerances instance
        force_refit: If True, always refit

    Returns:
        (bool, str): (can_reuse, reason)
            - can_reuse: True if can reuse Cantera coefficients
            - reason: Reason for decision (for logging)
                - "reuse": Coefficients can be reused
                - "non_nasa7": Non-NASA7 format
                - "tmid_differs": Continuous but different Tmid
                - "not_continuous": Not continuous at Tmid
                - "force_refit": Force refit requested
                - "no_cantera_coeffs": Cantera coefficients not available

    Raises:
        RuntimeError: If Cantera coefficients fail consistency check
    """
    # Check force_refit flag
    if force_refit:
        return False, "force_refit"

    # Check if NASA7 format
    if not is_nasa7:
        return False, "non_nasa7"

    # Check if Cantera coefficients available
    if cantera_nasa7 is None:
        return False, "no_cantera_coeffs"

    # Prepare dimensionless data
    cp_over_R = cp / R
    h_over_RT = h / (R * T)
    s_over_R = s / R

    # Check 1: Consistency (MUST PASS or raise error)
    consistency = cantera_nasa7.check_consistency(
        T,
        cp_over_R,
        h_over_RT,
        s_over_R,
        abs_tol=tolerances.consistency_abs_tol,
    )

    if not consistency["is_consistent"]:
        raise RuntimeError(
            f"Cantera NASA7 coefficients are INCONSISTENT with evaluated "
            f"thermodynamic data. This indicates a problem with the mechanism file. "
            f"Errors: cp={consistency['cp_error']:.2e}, "
            f"h={consistency['h_error']:.2e}, s={consistency['s_error']:.2e}"
        )

    # Check 2: Continuity
    continuity = cantera_nasa7.check_continuity(
        cp_tol=tolerances.continuity_cp_tol,
        cpdT_tol=tolerances.continuity_cpdT_tol,
        h_tol=tolerances.continuity_h_tol,
        s_tol=tolerances.continuity_s_tol,
    )

    is_continuous = continuity["is_continuous"]

    # Check 3: Tmid match
    tmid_matches = abs(cantera_nasa7.Tmid - Tmid) / Tmid < tolerances.tmid_rel_tol

    # Decision tree
    if is_continuous and tmid_matches:
        return True, "reuse"
    elif is_continuous:
        return False, "tmid_differs"
    else:
        return False, "not_continuous"
