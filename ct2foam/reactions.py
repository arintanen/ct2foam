#!/usr/bin/env python3

import cantera as ct
from contextlib import contextmanager


class ReactionsWriter:

    def __init__(self):
        self.lines = []
        self.level = 0

    def write(self, line=""):
        self.lines.append("    " * self.level + line)

    @contextmanager
    def block(self, name):
        self.write(name)
        self.write("{")
        self.level += 1

        try:
            yield
        finally:
            self.level -= 1
            self.write("}")

    def write_arrhenius(self, rate):

        A = rate.pre_exponential_factor
        beta = rate.temperature_exponent
        Ta = rate.activation_energy / ct.gas_constant

        self.write(f"A       {A:.16e};")
        self.write(f"beta    {beta:.16g};")
        self.write(f"Ta      {Ta:.16g};")

    def write_efficiencies(self, third_body):

        if not third_body:
            return

        if not third_body.efficiencies:
            return

        self.write("coeffs")
        self.write(str(len(third_body.efficiencies)))
        self.write("(")
        self.level += 1
        for sp, eff in sorted(third_body.efficiencies.items()):
            self.write(f"({sp:<12} {eff})")
        self.level -= 1
        self.write(");")
    def format_species(self, species_dict):
        terms = []

        for sp, nu in species_dict.items():
            if nu == 1:
                terms.append(sp)
            else:
                terms.append(f"{nu:g}{sp}")

        return " + ".join(terms)

    def write_type(self, type, rxn):
        if rxn.reversible:
            type = "reversible" + type
        else:
            type = "irreversible" + type
        self.write(f"type {type};")

        lhs = self.format_species(rxn.reactants)
        rhs = self.format_species(rxn.products)

        self.write(f'reaction "{lhs} = {rhs}";')

    def write_reaction(self, rxn):

        rate = rxn.rate

        match rate:

            case ct.ArrheniusRate():

                reaction_type = "Arrhenius"

                if rxn.third_body:
                    reaction_type = "ThirdBody" + reaction_type

                self.write_type(reaction_type, rxn)

                self.write_arrhenius(rate)

                self.write_efficiencies(
                    rxn.third_body,
                )

            case ct.LindemannRate():

                self.write_type(
                    "ArrheniusLindemannFallOff", 
                    rxn
                )
                with self.block("k0"):
                    self.write_arrhenius(rate.low_rate)

                with self.block("kInf"):
                    self.write_arrhenius(rate.high_rate)

                with self.block("thirdBodyEfficiencies"):
                    self.write_efficiencies(
                        rxn.third_body,
                    )

            case ct.TroeRate():

                self.write_type(
                    "ArrheniusTroeFallOff", 
                    rxn
                )

                with self.block("k0"):
                    self.write_arrhenius(rate.low_rate)

                with self.block("kInf"):
                    self.write_arrhenius(rate.high_rate)

                with self.block("F"):

                    alpha, Tsss, Ts = rate.falloff_coeffs[:3]

                    Tss = (
                        rate.falloff_coeffs[3]
                        if len(rate.falloff_coeffs) > 3
                        else 4.503599627e15 # great value in openfoam
                    )

                    with self.block("F"):
                        self.write(f"alpha {alpha};")
                        self.write(f"Tsss  {Tsss};")
                        self.write(f"Ts    {Ts};")
                        self.write(f"Tss   {Tss};")
                with self.block("thirdBodyEfficiencies"):
                    self.write_efficiencies(
                        rxn.third_body
                    )

            case ct.PlogRate():
                raise RuntimeError("PLOG reactions not supported yet in OpenFOAM")
                self.write_type(
                    "ArrheniusPLOG", 
                    rxn
                )

                self.write("rateCoeffs")
                self.write("(")
                self.level +=1
                for P, rates in rate.rates:

                    self.write("(")
                    self.level +=1
                    self.write(f"        {P}")

                    for r in rates:

                        A = r.pre_exponential_factor
                        beta = r.temperature_exponent
                        Ta = (
                            r.activation_energy
                            / ct.gas_constant
                        )

                        self.write("(")
                        self.level +=1
                        self.write(f"{A:.16e}")
                        self.write(f"{beta:.16g}")
                        self.write(f"{Ta:.16g}")
                        self.level -=1
                        self.write(")")
                        self.level -=1

                    self.write(")")
                    self.level -=1

                self.write(");")

            case _:
                raise RuntimeError("Unsupported reaction type:", type(rate).__name__)

    def convert(self, mechanism):

        gas = ct.Solution(mechanism)

        with self.block("reactions"):
            for i, rxn in enumerate(gas.reactions(), start=1):

                with self.block(f"reaction{i}"):

                    self.write_reaction(rxn)


    def save(self, filename):
        with open(filename, "w") as f:
            f.write("\n".join(self.lines))