# anisoap
A Python Package for Computing the Smooth Overlap of Anisotropic Positions

## Installation

The installation of the library for python use can be done simply with

    pip install .

The code is currently still being developed. To make sure that your version behaves properly, please run the internal tests from the main directory with

    pytest tests/.

Please contact the developers if some tests fail.

### Dependencies

Before installing anisoap, please make sure you have at least the
following packages installed:
* python (3.6 or higher)
* numpy (1.13 or higher)
* scipy (1.4.0 or higher)
* ASE (3.18 or higher)
* math
* Equistore
* Rascaline 

## For developers:
Please run pytest and check that all tests pass before pushing new changes to the main branch with

    pytest tests/.

