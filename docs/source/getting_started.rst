===============
Getting Started
===============

Introduction
------------
AniSOAP is a package for creating machine-learnable representations of systems 
of particles in a way that preserves geometrical information.  AniSOAP works 
similarly to SOAP (Smooth Overlap of Atomic Potentials) in many ways, with the 
chief difference being AniSOAP's ability to represent ellipsoidal particles.

First Steps
-----------
AniSOAP depends extensively on the packages `rascaline 
<https://luthaf.fr/rascaline/latest/get-started/rascaline.html>`_ 
and `metatensor <https://lab-cosmo.github.io/metatensor/latest/>`_.  
It may be helpful to familiarize yourself with these packages before diving into
AniSOAP.

Key Concepts
------------
In the AniSOAP representation, particles are treated as multivariate Gaussians (MVGs).
For each particle, we can consider its n-body interactions with nearby particles.
Each local density field is then decomposed in terms of spherical harmonics and 
some choice of radial basis functions in order to be compactly represented in
vector form.

A single AniSOAP vector represents the environment of a single central ellipsoid.
A system of such particles can then be represented, for example, as a matrix of
AniSOAP vectors.

