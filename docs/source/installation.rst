============ 
Installation
============

Dependencies
------------

A portion of AniSOAP is written in Rust, so installing the Rust compiler (rustc) and package manager (Cargo) is a prerequisite. Please follow the instructions for your platform here: https://rustup.rs/. Note that on unix systems, one must first install the C-compiler toolchain before installing rust. Rust and cargo are properly installed when ``rustc --version`` and ``cargo --version`` don't produce an error.

The rest of the python dependencies are in ``pyproject.toml`` in the root folder and are all available on PyPI. **Installing the AniSOAP package will automatically install the dependencies.** We recommend using an environment manager like conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a clean working environment for AniSOAP.

Installing AniSOAP
------------------

Navigate to the directory where you would like the AniSOAP package to be located, then copy and paste the 
following into your shell::

  git clone https://github.com/cersonsky-lab/AniSOAP

Then navigate to the AniSOAP directory with::

  cd AniSOAP

Then, install the AniSOAP library and *the latest version* of all dependencies::

  pip install .

Note that this step compiles the rust dependencies, so this step will fail if rust is not installed.

If this step fails because of incompatabilities related to dependency-versioning, we provide a ``requirements.txt`` file that contains pinned versions for most of these dependencies, located in the root directory of the AniSOAP project. Use::

  pip install -r requirements.txt 

to install dependencies, then run ``pip install .`` again to install the AniSOAP library itself.

Testing
-------

AniSOAP is still under active development, so you may want to run some tests to ensure that your installation is working properly.  From the main directory you can run the internal tests with::

  pytest tests/.


Local Documentation Build
-------------------------

To build the documentation locally, you can install additional Sphinx dependencies using::

  pip install -r docs/requirements.txt 

Then, you can use the Makefile in ``docs/`` to build the docs. For example, navigating to the ``docs/`` folder, then running ``make html`` will create the html documentation and place it in the ``docs/build``. Doing so will also run the examples in ``notebooks/example01_invariances_of_powerspectrum.py`` and ``notebooks/example02_learn_benzene.py`` and embed these examples in the documentation. To build documentation without running the examples, you can use ``make html-noplot``. This will run much more quickly.
