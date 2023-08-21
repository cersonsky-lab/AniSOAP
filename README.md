AniSOAP
=======

<a href="https://github.com/cersonsky-lab/anisoap/actions?query=workflow%3ATest">
  <img src="https://github.com/cersonsky-lab/anisoap/workflows/Test/badge.svg"/>
</a><a href="https://codecov.io/gh/cersonsky-lab/anisoap/">
  <img src="https://codecov.io/gh/cersonsky-lab/anisoap/branch/main/graph/badge.svg?token=UZJPJG34SM" />
</a>

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

Contributors
------------

Thanks goes to all people that make AniSOAP possible:

<a href="https://github.com/cersonsky-lab/anisoap/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cersonsky-lab/anisoap" />
</a>
