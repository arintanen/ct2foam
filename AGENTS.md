# ct2foam Agent Guidelines

This document provides guidelines for AI agents working in the ct2foam codebase.

## Quick orientation

ct2foam converts Cantera-based thermophysical data (NASA7 polynomials, Sutherland and polynomial transport) into OpenFOAM-compatible dictionary files. The central design constraint is that **all species must share a common NASA7 mid-point temperature** (`Tmid`), which OpenFOAM requires. The package refits polynomials when necessary using a constrained least-squares solver (`lsqlin.py`).

**Python ≥ 3.11** is required (`typing.Self` is used throughout).

## Package structure

```
ct2foam/
├── __init__.py           # re-exports NASA7Polynomial, Species, SpeciesList, Mixture
├── __main__.py           # python -m ct2foam entry point
├── ct2foam.py            # CLI argument parsing → SpeciesList / Mixture workflow
├── nasa7.py              # ThermoData dataclass + NASA7Polynomial (fit, evaluate, quality)
├── transport.py          # TransportData + Sutherland / Polynomial / LogPolynomial
├── species.py            # Species + SpeciesList (iterable container)
├── mixture.py            # Mixture (single-composition gas)
├── foam_writer.py        # write_thermo_transport(), write_species_list(), write_reactions(), write_foam()
├── lsqlin.py             # MIT-licensed constrained least-squares solver (third-party)
└── tests/
    ├── test_core.py                  # 57 unit + integration tests for the Python API
    └── test_openfoam_integration.py  # end-to-end test against the OF C++ binary (skipped if no binary)

openfoam_reference/
├── Test-thermoMixture.C  # C++ OF-dev utility that evaluates thermoDicts
├── thermoDict            # reference dict for H2O mixture
├── thermoDict_H2         # reference dict for H2 species
├── thermoDict_synth      # synthetic NASA7 + Sutherland dict for simple validation
└── Make/                 # wmake build files
```

No `scripts/`, `thermo_transport/`, or `pyjac2foam/` directories exist in v2.

## Build, lint, and test

```bash
# Install (dev mode)
pip install -e .

# Run all tests  ← always do this after any change
python -m unittest discover

# Run a single test file
python -m unittest ct2foam.tests.test_core

# Run one test method
python -m unittest ct2foam.tests.test_core.TestNASA7Polynomial.test_fit_full
```

### OpenFOAM integration test

Automatically skipped when `Test-thermoMixture` is not in `PATH`.

```bash
source ~/OpenFOAM/OpenFOAM-dev/etc/bashrc
python -m unittest ct2foam.tests.test_openfoam_integration
```

Rebuild the C++ binary only after editing `Test-thermoMixture.C`:

```bash
source ~/OpenFOAM/OpenFOAM-dev/etc/bashrc
cd openfoam_reference && wmake
```

> All thermo properties compared in the integration test are **mass-specific**
> (J/kg/K, J/kg, W/m/K) — divide molar values by `molWeight` [kg/kmol].

## CLI usage

```bash
# Convert all species in a mechanism
ct2foam --input h2o2.yaml --output out/ --Tmid 1000.0 --plot

# Convert a mixture
ct2foam --input h2o2.yaml --mixture "O2:1,N2:3.67" --mixture-name air --output out/

ct2foam -h   # full help
```

## Code conventions

### Language and style
- **Python ≥ 3.11**; type hints required on all public methods (`typing.Self`, `Optional`, `Union`, `np.ndarray`, `npt.NDArray`)
- `-> None` return annotations are **omitted** by convention
- Google-style docstrings: class-level `Args:` for constructor params; `Returns:` only when the structure is non-obvious; no `Raises:` section
- 4-space indentation, no tabs; ~80–100 character line length
- `snake_case` functions/variables, `PascalCase` classes, `_underscore` private members

### Key patterns
- **Data not stored in objects** — `Species` and `Mixture` hold fitted coefficients only; raw `(T, cp, mu, …)` arrays are passed to fitting methods, not stored
- **Classmethod constructors** — `from_ct()` / `from_ct_mech()` / `from_ct_mixture()` are the standard entry points; call them by concrete class name (not `cls.method()`) to avoid LSP false positives
- **Constrained fitting** — NASA7 fits use `lsqlin.py` (CVXOPT QP); always check `NASA7Polynomial.fit_quality()` after fitting
- **NumPy polynomial order** — `np.polyfit` returns coefficients in **descending** order; `foam_writer.py` reverses them before writing (OpenFOAM wants ascending)
- `pathlib.Path` for all file handling

### Docstring format reference

```python
class Foo:
    """One-line summary.

    Longer description if needed.

    Args:
        x: description with units [unit]
        y: description
    """

    def bar(self, T: np.ndarray) -> dict:
        """One-line summary.

        Args:
            T: sample temperatures [K]

        Returns:
            dict with keys 'a' and 'b' - description.
        """
```

## Common tasks

### Add a new species fit
1. Add fitting logic in `nasa7.py` or `transport.py`
2. Expose via `from_ct()` classmethod if applicable
3. Wire into `Species.from_ct()` / `SpeciesList.from_ct_mech()` in `species.py`
4. Update `foam_writer.write_thermo_transport()` if the output format changes
5. Add tests in `ct2foam/tests/test_core.py`

### Debug a poor fit
- Pass `plot=True` to `SpeciesList.from_ct_mech()` — plots are saved under `<output>/Figures/`
- Call `nasa7.fit_quality(data)` to inspect `c0_continuity` and `consistency` dicts
- Use `h2o2.yaml` (in test data) for quick iteration

### Modify OpenFOAM output format
1. Edit `foam_writer.py`
2. Update `Test-thermoMixture.C` if the dict structure changes, then rebuild with `wmake`
3. Re-run the integration test

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| cantera | ≥ 2.5.1 | thermodynamics / transport reference data |
| numpy | ≥ 1.20 | numerics (`npt.NDArray` requires 1.20+) |
| scipy | ≥ 1.5.2 | `curve_fit` for Sutherland fitting |
| cvxopt | ≥ 1.2.0 | constrained QP for NASA7 fitting |
| matplotlib | ≥ 3.3 | fit quality plots |

Cantera is **not** in `install_requires` — install it separately via conda:

```bash
conda install -c cantera cantera
```

## Notes

- GPLv3 licensed; `lsqlin.py` is MIT (compatible) — see `LICENSE` for attribution
- `typing.Self` requires Python 3.11+; do not lower `python_requires` in `setup.py`
- The `openfoam_reference/` C++ code targets **OF-dev** API (`Cp()`/`Cv()` not `cp()`/`cv()`)

