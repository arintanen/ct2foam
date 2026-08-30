import unittest
import shutil
import subprocess
import tempfile
import re
import os
from pathlib import Path
import math

from ct2foam import nasa7
from ct2foam import transport


def parse_thermo_block(file_path: Path, species_name: str) -> dict:
    """Parse a single species block from an OpenFOAM-style thermo dict.
    Returns a dict with keys: molWeight, Tlow, Thigh, Tcommon, lowCpCoeffs, highCpCoeffs, As, Ts
    """
    text = file_path.read_text()
    # find species block
    m = re.search(r"^" + re.escape(species_name) + r"\s*\{", text, flags=re.M)
    if not m:
        raise RuntimeError(f"Species {species_name} not found in {file_path}")
    start = m.end()
    # find matching closing brace
    depth = 1
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    block = text[start:end]

    def extract_scalar(key):
        mm = re.search(rf"{key}\s+([0-9eE+\-\.]+)\s*;", block)
        return float(mm.group(1)) if mm else None

    def extract_coeffs(key):
        mm = re.search(rf"{key}\s*\(\s*([^\)]+)\)", block)
        if not mm:
            return None
        nums = re.findall(r"[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?", mm.group(1))
        return [float(x) for x in nums]

    result = {}
    result["molWeight"] = extract_scalar("molWeight")
    result["Tlow"] = extract_scalar("Tlow")
    result["Thigh"] = extract_scalar("Thigh")
    # Tcommon sometimes named Tcommon
    result["Tcommon"] = extract_scalar("Tcommon")
    result["lowCpCoeffs"] = extract_coeffs("lowCpCoeffs")
    result["highCpCoeffs"] = extract_coeffs("highCpCoeffs")
    # transport
    mm = re.search(r"transport\s*\{([^}]+)\}", block, flags=re.S)
    if mm:
        trans = mm.group(1)
        mmA = re.search(r"As\s*([0-9eE+\-\.]+)\s*;", trans)
        mmT = re.search(r"Ts\s*([0-9eE+\-\.]+)\s*;", trans)
        result["As"] = float(mmA.group(1)) if mmA else None
        result["Ts"] = float(mmT.group(1)) if mmT else None
        # muCoeffs and muLogCoeffs optional
        result["muCoeffs"] = extract_coeffs_from_text = None
        mm_mu = re.search(r"muCoeffs<\d+>\s*\(\s*([^\)]+)\)", trans)
        if mm_mu:
            nums = re.findall(r"[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?", mm_mu.group(1))
            result["muCoeffs"] = [float(x) for x in nums]
        mm_k = re.search(r"kappaCoeffs<\d+>\s*\(\s*([^\)]+)\)", trans)
        if mm_k:
            nums = re.findall(r"[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?", mm_k.group(1))
            result["kappaCoeffs"] = [float(x) for x in nums]
    return result


class TestOpenFOAMIntegration(unittest.TestCase):
    def setUp(self):
        # locate Test-thermoMixture
        self.binary = shutil.which("Test-thermoMixture")
        # locate reference dicts
        self.ref_dir = Path(__file__).resolve().parents[2] / "openfoam_reference"
        if not self.ref_dir.exists():
            # fallback to repository top-level openfoam_reference
            self.ref_dir = Path(__file__).resolve().parents[3] / "openfoam_reference"

    def test_H2_integration_with_openfoam(self):
        if not self.binary:
            self.skipTest("Test-thermoMixture binary not found in PATH; skipping integration test")
        thermoDict = self.ref_dir / "thermoDict"
        thermoDict_H2 = self.ref_dir / "thermoDict_H2"
        thermoDict_synth = self.ref_dir / "thermoDict_synth"
        if not thermoDict.exists() or not thermoDict_H2.exists():
            self.skipTest("Reference thermoDict files not found; skipping")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # copy reference dicts into temp dir
            shutil.copy(thermoDict, td / "thermoDict")
            shutil.copy(thermoDict_H2, td / "thermoDict_H2")
            if thermoDict_synth.exists():
                shutil.copy(thermoDict_synth, td / "thermoDict_synth")

            # run the OpenFOAM utility
            proc = subprocess.run([self.binary], cwd=str(td), capture_output=True, text=True)
            out = proc.stdout + "\n" + proc.stderr

            # extract blocks robustly (tolerant to Info prefixes and formatting)
            def extract_block_by_header(output: str, header: str) -> str:
                lines = output.splitlines()
                start = None
                for i, l in enumerate(lines):
                    if header in l:
                        start = i + 1
                        break
                if start is None:
                    for i, l in enumerate(lines):
                        if header.lower() in l.lower():
                            start = i + 1
                            break
                if start is None:
                    return ""
                block_lines = []
                for l in lines[start:]:
                    # stop at empty line or a line that looks like a new section header
                    if l.strip() == "":
                        break
                    # remove common OpenFOAM Info prefixes and surrounding quotes
                    s = re.sub(r"^\s*\S+<<\s*", "", l)
                    s = s.strip().strip('"')
                    block_lines.append(s)
                return "\n".join(block_lines)

            snippet_H2 = extract_block_by_header(out, "H2 from python")
            snippet_SYN = extract_block_by_header(out, "SYN from python")

            def parse_snippet(snippet):
                found = {}
                for line in snippet.splitlines():
                    line = line.strip()
                    mm = re.match(r"([A-Za-z0-9_]+)\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", line)
                    if mm:
                        key = mm.group(1)
                        val = float(mm.group(2))
                        found[key] = val
                return found

            found_H2 = parse_snippet(snippet_H2)
            found_SYN = parse_snippet(snippet_SYN)

            # basic checks for H2
            self.assertIn("R", found_H2)
            self.assertIn("cp", found_H2)
            self.assertIn("mu", found_H2)
            self.assertIn("kappa", found_H2)
            self.assertIn("h", found_H2)
            self.assertIn("s", found_H2)

            # parse the thermoDict_H2 to build expected values with ct2foam routines
            parsed = parse_thermo_block(td / "thermoDict_H2", "H2")
            T = 400.0
            p = 1e5
            # build NASA7Polynomial
            coeffs_lo = parsed.get("lowCpCoeffs")
            coeffs_hi = parsed.get("highCpCoeffs")
            Tlow = parsed.get("Tlow")
            Thigh = parsed.get("Thigh")
            Tmid = parsed.get("Tcommon")
            if coeffs_lo is None or coeffs_hi is None:
                self.skipTest("NASA7 coeffs not present in thermoDict_H2; skipping")

            nasa = nasa7.NASA7Polynomial(coeffs_lo, coeffs_hi, Tmid, Tlow, Thigh)
            W = parsed.get("molWeight")  # kg/kmol (same as g/mol)
            R = nasa7.ct.gas_constant    # J/(kmol·K)
            Rspec = R / W                # J/(kg·K) — species-specific gas constant

            # OpenFOAM returns mass-specific properties (J/kg/K, J/kg, etc.)
            cp = nasa.cp_over_R([T])[0] * Rspec
            h  = nasa.h_over_RT([T])[0] * Rspec * T
            s  = nasa.s_over_R([T])[0]  * Rspec

            # viscosity via Sutherland if As/Ts present
            As = parsed.get("As")
            Ts = parsed.get("Ts")
            if As is None or Ts is None:
                self.skipTest("Sutherland transport params missing; skipping")

            mu = As * math.sqrt(T) / (1.0 + Ts / T)
            # Eucken: kappa = mu * cv * (1.32 + 1.77 * Rspec / cv)
            cv = cp - Rspec
            kappa = mu * cv * (1.32 + 1.77 * Rspec / cv)

            # compare with values from OpenFOAM
            # tolerances: relative
            rel_tol_cp = 1e-6
            rel_tol_h = 1e-6
            rel_tol_s = 1e-6
            rel_tol_mu = 1e-6
            rel_tol_kappa = 1e-5

            def rel_err(a, b):
                return abs(a - b) / (abs(b) + 1e-300)

            self.assertIn("cp", found_H2)
            self.assertTrue(rel_err(found_H2["cp"], cp) < rel_tol_cp,
                            f"cp mismatch: got {found_H2['cp']} expected {cp}")
            self.assertTrue(rel_err(found_H2["h"], h) < rel_tol_h,
                            f"h mismatch: got {found_H2['h']} expected {h}")
            self.assertTrue(rel_err(found_H2["s"], s) < rel_tol_s,
                            f"s mismatch: got {found_H2['s']} expected {s}")
            self.assertTrue(rel_err(found_H2["mu"], mu) < rel_tol_mu,
                            f"mu mismatch: got {found_H2['mu']} expected {mu}")
            self.assertTrue(rel_err(found_H2["kappa"], kappa) < rel_tol_kappa,
                            f"kappa mismatch: got {found_H2['kappa']} expected {kappa}")

            # --- Now synthetic SYN checks ---
            if not found_SYN:
                self.skipTest("SYN block not found in OpenFOAM output; skipping synthetic checks")

            parsed_syn = parse_thermo_block(td / "thermoDict_synth", "SYN")
            coeffs_lo = parsed_syn.get("lowCpCoeffs")
            coeffs_hi = parsed_syn.get("highCpCoeffs")
            Tlow = parsed_syn.get("Tlow")
            Thigh = parsed_syn.get("Thigh")
            Tmid = parsed_syn.get("Tcommon")
            nasa_syn = nasa7.NASA7Polynomial(coeffs_lo, coeffs_hi, Tmid, Tlow, Thigh)
            W_syn = parsed_syn.get("molWeight")
            Rspec_syn = R / W_syn
            cp_syn   = nasa_syn.cp_over_R([T])[0] * Rspec_syn
            h_syn    = nasa_syn.h_over_RT([T])[0]  * Rspec_syn * T
            s_syn    = nasa_syn.s_over_R([T])[0]   * Rspec_syn
            As_syn   = parsed_syn.get("As")
            Ts_syn   = parsed_syn.get("Ts")
            mu_syn   = As_syn * math.sqrt(T) / (1.0 + Ts_syn / T)
            cv_syn   = cp_syn - Rspec_syn
            kappa_syn = mu_syn * cv_syn * (1.32 + 1.77 * Rspec_syn / cv_syn)

            # compare
            self.assertTrue(rel_err(found_SYN.get("cp", 0),    cp_syn)    < rel_tol_cp)
            self.assertTrue(rel_err(found_SYN.get("h", 0),     h_syn)     < rel_tol_h)
            self.assertTrue(rel_err(found_SYN.get("s", 0),     s_syn)     < rel_tol_s)
            self.assertTrue(rel_err(found_SYN.get("mu", 0),    mu_syn)    < rel_tol_mu)
            self.assertTrue(rel_err(found_SYN.get("kappa", 0), kappa_syn) < rel_tol_kappa)


if __name__ == "__main__":
    unittest.main()
