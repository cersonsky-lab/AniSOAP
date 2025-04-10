============ 
Installation
============

Dependencies
------------

Before installing AniSOAP, please make sure you have the following installed (We recommend using an environment manager like `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html/>`_):

* `Python: 3.9 or 3.10 <https://www.python.org/downloads/>`_
* `numPy: 1.13 or higher <https://numpy.org/install/>`_
* `sciPy: 1.4.0 or higher <https://scipy.org/install/>`_
* `Atomic Simulation Environment (ASE): 3.18 or higher <https://wiki.fysik.dtu.dk/ase/install.html>`_
* `Metatensor <https://docs.metatensor.org/latest/index.html>`_
* `Featomic (formerly known as Rascaline) <https://metatensor.github.io/featomic/latest/index.html>`_
* `Rust -- We recommend using rustup <https://rustup.rs/>`_


Installing AniSOAP
------------------

Navigate to the directory where you would like the AniSOAP package to be located, then copy and paste the 
following into your shell::

  git clone https://github.com/cersonsky-lab/AniSOAP

Then navigate to the AniSOAP directory with::

  cd AniSOAP

First, install all of AniSOAP's dependencies with::

  pip install -r requirements.txt

Now use pip to install the AniSOAP library::

  pip install .


Testing
-------

AniSOAP is still under active development, so you may want to run some tests to ensure that your installation is working properly.  From the main directory you can run the internal tests with::

  pytest tests/.


Local Documentation Build
-------------------------

To build the documentation locally, you can install Sphinx and related dependencies using::

  pip install -r docs/requirements.txt 

Then, you can use the Makefile in docs/ to build the docs. 
