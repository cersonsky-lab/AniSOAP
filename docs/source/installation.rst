============ 
Installation
============

Dependencies
------------

Before installing AniSOAP, please make sure you have the following installed\:

* `Python: 3.6 or higher <https://www.python.org/downloads/>`_
* `numPy: 1.13 or higher <https://numpy.org/install/>`_
* `sciPy: 1.4.0 or higher <https://scipy.org/install/>`_
* `Atomic Simulation Environment (ASE): 3.18 or higher <https://wiki.fysik.dtu.dk/ase/install.html>`_
* `Metatensor <https://lab-cosmo.github.io/metatensor/latest/get-started/installation.html>`_
* `Rascaline <https://luthaf.fr/rascaline/latest/get-started/installation.html>`_
* `Rust -- We reccommend using 'rustup <https://rustup.rs/>`_


Installing AniSOAP
------------------

Navigate to the directory where you would like the AniSOAP package to be located, then copy and paste the 
following into your shell::

  git clone https://github.com/cersonsky-lab/AniSOAP

Then navigate to the AniSOAP directory with::

  cd AniSOAP

Now use pip to install the library::

  pip install .


Testing
-------

AniSOAP is still under active development, so you may want to run some tests to ensure that your installation is working properly.  From the main directory you can run the internal tests with::

  pytest tests/.



