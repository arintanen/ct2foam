"""MechanismDataset orchestrator — manages a list of SpeciesDataset objects."""

import numpy as np
from pathlib import Path

from .species_dataset import SpeciesDataset
from .coefficients import NASA7Polynomial
from ct2foam.thermo_transport import foam_writer as writer


class MechanismDataset:
    """Orchestrator that creates and manages SpeciesDataset objects for a mechanism."""

    def __init__(self, species_datasets, mechanism_name, Tmid, Tlow, Thigh):
        self.species_datasets = list(species_datasets)
        self.mechanism_name = str(mechanism_name)
        self.Tmid = float(Tmid)
        self.Tlow = float(Tlow)
        self.Thigh = float(Thigh)
        self.fit_log = []
        self.failed_species = {}

    @classmethod
    def from_cantera(cls, mech_file, Tmid, Tlow=None, Thigh=None, T_eval=None):
        """Create a MechanismDataset by loading a Cantera mechanism file."""
        import cantera as ct

        gas = ct.Solution(str(mech_file))
        R = ct.gas_constant  # TODO:this must be consistent with old?
        p0 = ct.one_atm

        if Tlow is None:
            Tlow = 200.0  # TODO: check limits
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
        species_datasets = []

        gas.transport_model = "multicomponent"

        for sp_name in gas.species_names:
            i = gas.species_index(sp_name)
            sp_obj = gas.species(i)
            reactants = sp_name + ":1.0"

            # Check thermo format and extract Cantera NASA7 coefficients
            thermo_type = type(sp_obj.thermo).__name__
            is_nasa7 = thermo_type == "NasaPoly2"
            cantera_nasa7 = None

            if is_nasa7:
                # Extract coefficients: [Tmid, c_hi[0..6], c_lo[0..6]]
                coeffs = sp_obj.thermo.coeffs
                ct_Tmid = float(coeffs[0])
                ct_c_hi = np.array(coeffs[1:8])
                ct_c_lo = np.array(coeffs[8:15])
                cantera_nasa7 = NASA7Polynomial(ct_c_lo, ct_c_hi, ct_Tmid)

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

            # Standard-state properties
            cp0_over_R = gas.species(i).thermo.cp(298.15) / R
            dhf_over_R = gas.species(i).thermo.h(298.15) / R
            s0_over_R = gas.species(i).thermo.s(298.15) / R

            # Elements
            elements = {}
            for elem in gas.element_names:
                na = gas.n_atoms(sp_name, elem)
                if na > 0:
                    elements[elem] = na

            sd = SpeciesDataset(
                name=sp_name,
                T=T_eval,
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
                elements=elements,
                cantera_nasa7=cantera_nasa7,
                is_nasa7=is_nasa7,
            )
            species_datasets.append(sd)

        return cls(species_datasets, str(mech_file), Tmid, T_eval[0], T_eval[-1])

    def fit_all(self, verbose=True, tolerances=None, force_refit=False):
        """Fit thermo and transport for all species; log errors and continue.

        Args:
            verbose: If True, print fitting messages
            tolerances: FittingTolerances instance
            force_refit: If True, skip Cantera coefficient reuse

        Returns:
            dict with keys: 'succeeded', 'failed', 'reused', 'details'
        """
        succeeded = 0
        failed = 0
        reused = 0
        details = {}

        for sd in self.species_datasets:
            sp_detail = {
                "thermo": None,
                "transport": None,
                "quality": None,
                "error": None,
                "reused": False,
            }
            try:
                # Track if Cantera coefficients were available
                had_cantera_coeffs = (
                    sd.cantera_nasa7 is not None and sd.is_nasa7 and not force_refit
                )

                # Fit thermo
                sd.fit_thermo(
                    tolerances=tolerances, force_refit=force_refit, verbose=verbose
                )
                sp_detail["thermo"] = "ok"

                # Check if reused (Tmid will match Cantera's if reused)
                if had_cantera_coeffs and sd.cantera_nasa7 is not None:
                    ct_Tmid = sd.cantera_nasa7.Tmid
                    if abs(sd.nasa7.Tmid - ct_Tmid) < 1e-12:
                        reused += 1
                        sp_detail["reused"] = True

                sd.fit_transport()
                sp_detail["transport"] = "ok"

                quality = sd.check_quality()
                sp_detail["quality"] = quality

                succeeded += 1
                self.fit_log.append(f"{sd.name}: fit OK")

            except Exception as e:
                failed += 1
                msg = (
                    f"{sd.name}: fitting failed — {e}. "
                    "Consider adjusting parameters or tune your tolerances."
                )
                sp_detail["error"] = msg
                self.failed_species[sd.name] = msg
                self.fit_log.append(msg)
                if verbose:
                    print(msg)

            details[sd.name] = sp_detail

        return {
            "succeeded": succeeded,
            "failed": failed,
            "reused": reused,
            "details": details,
        }

    def print_fit_summary(self):
        """Print a summary of the fitting process."""
        total = len(self.species_datasets)
        n_failed = len(self.failed_species)
        n_ok = total - n_failed
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

        names = [sd.name for sd in self.species_datasets if sd.nasa7 is not None]
        writer.write_species_list(species_file, names)

        for sd in self.species_datasets:
            if sd.nasa7 is None:
                continue

            poly_mu = sd.polynomial.coeffs_mu if sd.polynomial else np.zeros(4)
            poly_kappa = sd.polynomial.coeffs_kappa if sd.polynomial else np.zeros(4)
            logpoly_mu = (
                sd.log_polynomial.coeffs_mu if sd.log_polynomial else np.zeros(4)
            )
            logpoly_kappa = (
                sd.log_polynomial.coeffs_kappa if sd.log_polynomial else np.zeros(4)
            )
            As = sd.sutherland.As if sd.sutherland else 0.0
            Ts = sd.sutherland.Ts if sd.sutherland else 0.0

            writer.write_thermo_transport(
                thermo_file,
                sd.name,
                sd.W,
                As,
                Ts,
                poly_mu,
                poly_kappa,
                logpoly_mu,
                logpoly_kappa,
                self.Tmid,
                self.Tlow,
                self.Thigh,
                sd.nasa7.coeffs_low,
                sd.nasa7.coeffs_high,
                elements=sd.elements,
            )
