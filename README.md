# ct2foam 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Convert Cantera based thermophysical data to OpenFoam format

ct2foam enables user to generate OpenFOAM dictionary entries for NASA-polynomials, Sutherland and other transport models required by compressible multi-species solvers. Conversion and re-fitting procedures ensure NASA-polynomials will share the same common temperature between low and high temperature range coefficients, which is required in OpenFOAM. 

This python package utilises [Cantera](https://cantera.org/) to generate such entries for all species in a given chemical mechanism or for a gaseous mixture defined by the user. With the general functions provided in this package, it is also possible to create fits for thermophysical data based on experiments.

## Installation

```
git clone git@github.com:kahilah/ct2foam.git
cd ct2foam
pip install .
```

## Run tests:
```
python -m unittest discover
```

Note that <code>openfoam_reference/Test-thermoMixture.C</code> can be used for advanced testing and debugging how data, units and fit evaluations are handled on OpenFOAM side. 

## Running ct2foam:
### Convert all species in chemical mechanism
```
ct2foam --input h2o2.yaml --output test_output --Tmid 1000.0 --plot
```
outputs OpenFoam compatible thermodynamical and transport dictionary entries under output directory such that NASA-polynomials have a middle temperature of 1000 K. Thermodynamical fits are plotted under <code>test_output/Figures</code> directory.

### Convert a mixture 
```
ct2foam --input h2o2.yaml --mixture "O2:1, N2:3.67" --mixture-name GG --output test
```

### See <code>ct2foam -h</code> for help.

## Acknowledgements
- The constrained least-squares fitting procedure in <code>ct2foam/thermo_transport/lsqlin.py</code> is based on the work by Valeriy Vishnevskiy and Michael Hirsch, originally published in http://maggotroot.blogspot.ch/2013/11/constrained-linear-least-squares-in.html
