import argparse
import os
from thermo_transport import ct_properties
from thermo_transport import ct2foam_utils as utils
from thermo_transport import  warning_msg


def init_dirs(output_dir=None):
    if(output_dir==None):
        output_dir = os.getcwd()
        print("Using current directory as an output directory.")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Convert/refit cantera .cti -based transport and thermodynamic data into OpenFOAM format.')
    parser.add_argument('-i','--input', type=str, help='Mechanism (.cti/.yaml/.xml) file path.', default=None, required=True)
    parser.add_argument('-o','--output', type=str, help='Output directory path. (default is current directory)', default=None, required=False)
    parser.add_argument('-T','--Tmid', type=float, help='Common temperature for NASA-7 thermodynamical fits.', default=1000.0, required=False)
    parser.add_argument('-p','--plot', action='store_true', help='Generate plots when available.', required=False)
    args = parser.parse_args()

    output_dir = init_dirs(args.output)
    warning_msg.init_log(output_dir, args.input)

    data = ct_properties.ctThermoTransport(args.input, Tmid=args.Tmid)
    data.evaluate_properties()
   
    print("\nFitting transport properties:")
    transport_fits = utils.fit_ct_transport(data, poly_order=3)
    success = utils.transport_fit_quality(data, transport_fits, output_dir, plot=args.plot)
    if(not success):
        print("\nSome transport fits have failed.\n")
    else:
        print("Success.")


    print("\nFitting thermodynamic properties:")
    thermo_fits = utils.refit_ct_thermo(data, data.Tmid, output_dir)
    success = utils.nasa7_fit_quality(data, thermo_fits, output_dir, plot=args.plot)
    if(not success):
        print("Some NASA7 polynomial fits have failed.\n")
    else:
        print("Success.")


    print("\nWriting files to OpenFOAM format")
    species_file = os.path.join(output_dir, "species.foam")
    thermo_file = os.path.join(output_dir, "thermo.foam")
    reactions_file = os.path.join(output_dir, "reactions.foam")
    utils.ct2foam_thermo_writer(species_file, thermo_file, reactions_file, data, transport_fits, thermo_fits)

    print("\nDone")