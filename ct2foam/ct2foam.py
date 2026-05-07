import argparse
from datetime import date
from pathlib import Path


from ct2foam.species import SpeciesList
from ct2foam.mixture import Mixture

# TODO:
# 4) Test that output files are 1-1
# TODO: add tolerance limits to avoid error. Add print help.
# TODO: remove test data
# TODO: how many n points?
# TODO: go through TODOS
# TODO: Add README examples
# LICENSE + docstrings - CHECK lsqlin MIT licence - is it contaminating this to MIT as well?
# Go through the original TODO.md
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert/refit cantera-based transport and thermodynamic"
            "data into OpenFOAM format. To fit data for mixtures, user"
            "can define --mixture argument."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Cantera mechanism (.yaml/.xml) file path.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory path. (default is current directory)",
        default=Path.cwd(),
        required=False,
    )
    parser.add_argument(
        "-T",
        "--Tmid",
        type=float,
        help="Common temperature for NASA-7 thermodynamical fits.",
        default=1000.0,
        required=False,
    )
    parser.add_argument(
        "-Tl",
        "--Tlow",
        type=float,
        help="Temperature low-limit for NASA-7 thermodynamical fits.",
        default=280.0,
        required=False,
    )
    parser.add_argument(
        "-Th",
        "--Thigh",
        type=float,
        help="Temperature high-limit for NASA-7 thermodynamical fits.",
        default=3000.0,
        required=False,
    )
    parser.add_argument(
        "-n",
        "--mixture_name",
        type=str,
        help='(Optional) Mixture name, e.g. "air".',
        required=False,
    )
    parser.add_argument(
        "-m",
        "--mixture",
        type=str,
        help='(Optional) Molecular mixture ratio in cantera style: "O2:1, N2:3.76" ',
        required=False,
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="(Optional) Generate plots when available.",
        required=False,
)
    args = parser.parse_args()

    mechanism = args.input

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    fig_dir = Path(output_dir, "Figures")
    fig_dir.mkdir(exist_ok=True)

    # For mixtures, we need both arguments to be defined
    if args.mixture and not args.mixture_name:
        parser.error("--mixture_name must be specified")

    today = date.today()
    print("Date: " + str(today.strftime("%B %d, %Y")))
    print("Using Mechanism: " + mechanism)

    # Generate output for a mixture
    if args.mixture_name:
        print("Creating mixture - " + str(args.mixture_name) + ": " + args.mixture)
        mixture = Mixture.from_ct(
            mechanism_file=mechanism,
            mixture_name=args.mixture_name,
            mixture=args.mixture,
            Tmin=280.0,
            Tmax=3000.0,
            Tmid=1000.0,
            n=256,
            plot=True,
            fig_dir=fig_dir,
            tol=1e-2,
            tol_c0=1e-6
        )

        mixture.write_foam(output_dir)
        print("\nDone")
        return

    # Generate output for individual species
    species_list = SpeciesList.from_ct_mech(
        mechanism,
        Tmin=280.0,
        Tmax=3000.0,
        Tmid=1000.0,
        n=256,
        plot=True,
        fig_dir=fig_dir,
        tol=1e-2,
        tol_c0=1e-6
    )
    species_list.write_foam(output_dir)
    print("\nDone")


if __name__ == "__main__":
    main()
