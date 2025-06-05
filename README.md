AniSOAP
=======

<a href="https://github.com/cersonsky-lab/anisoap/actions?query=workflow%3ATest">
  <img src="https://github.com/cersonsky-lab/anisoap/workflows/Test/badge.svg"/>
</a><a href="https://codecov.io/gh/cersonsky-lab/anisoap/">
  <img src="https://codecov.io/gh/cersonsky-lab/anisoap/branch/main/graph/badge.svg?token=UZJPJG34SM" />
</a>

## Warning!

While technically complete, AniSOAP is in beta mode, and subject to new changes regularly. 
Please use with caution as we iron out some of the finer details.

## Documentation

Please read our latest documentation, containing examples and API usage here: [https://anisoap.readthedocs.io/en/latest/](https://anisoap.readthedocs.io/en/latest/)
## Dependencies

Before installing anisoap, please make sure you have at least the
following packages installed:
* python (3.9 or 3.10)
* numpy (1.13 or higher)
* scipy (1.4.0 or higher)
* ASE (3.18 or higher)
* math
* metatensor
* featomic

AniSOAP also requires the Rust language.  If you do not already have Rust installed, we recommend using the rustup tool, available [here](https://rustup.rs).

## Installation

The installation of the library for python use can be done simply with

    pip install .

which installs all of AniSOAP's dependencies and AniSOAP itself. This installs the latest version of each dependency. If these results in conflict, you can use

    pip install -r requirements.txt

to install all the dependencies with frozen versions, followed by `pip install .` to install AniSOAP itself. You can test the library itself using

    pytest tests/.

Please contact the developers if some tests fail.

## For developers:

Contributions are welcome!  For more information, please see the [guidelines](./CONTRIBUTING.md).

Contributors
------------

Thanks goes to all people that make AniSOAP possible:

<a href="https://github.com/cersonsky-lab/anisoap/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cersonsky-lab/anisoap" />
</a>
