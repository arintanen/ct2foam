#!/usr/bin/env python3

from pathlib import Path
import sys

import cantera as ct


OUTPUT = Path(__file__).resolve().parent / "cantera.out"


def main():
    gas = ct.Solution('gri30.yaml')
    gas.TPX = 1000.0, 1.36789e6, {"CH4": 0.5, "O2": 1.0, "N2": 3.76}
    r = ct.IdealGasConstPressureReactor(gas, clone=False)
    sim = ct.ReactorNet([r])
    sim.verbose = True

    # limit advance when temperature difference is exceeded
    delta_T_max = 20.
    r.set_advance_limit('temperature', delta_T_max)

    dt_max = 1.e-6
    t_end = 0.1
    states = ct.SolutionArray(gas, extra=['t'])
    print('{:10s} {:10s} {:10s} {:14s}'.format(
        't [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))


    with OUTPUT.open("w", encoding="ascii") as stream:
        stream.write("# Time [s]    Temperature [K]\n")
        time = 0.0
        while time < t_end:
            time += dt_max
            sim.advance(time)
            stream.write(f"{time:.12g}    {r.T:.12g}\n")

    print(f"Wrote Cantera samples to {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Cantera simulation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
