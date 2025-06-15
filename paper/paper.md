---
title: 'AniSOAP: Machine Learning Representations for Coarse-grained and Non-spherical Systems'
tags:
  - Python
  - machine learning
  - molecular simulation

authors:
  - name: Arthur Yan Lin
    orcid: 0000-0002-7665-3767
    affiliation: 1
  - name: Lucas Ortengren
    orcid: 0009-0002-8899-7513
    affiliation: 1
  - name: Seonwoo Hwang
    affiliation: 1
  - name: Yong-Cheol Cho
    orcid: 0009-0001-6038-6764
    affiliation: 1
  - name: Jigyasa Nigam
    orcid: 0000-0001-6857-4332
    affiliation: 2
  - name: Rose K. Cersonsky
    orcid: 0000-0003-4515-3441
    affiliation: 1
    corresponding: true
affiliations:
 - name: Department of Chemical and Biological Engineering, University of Wisconsin-Madison, USA
   index: 1
 - name: Research Laboratory of Electronics, Massachusetts Institute of Technology, USA
   index: 2
 - name: Department of Computer Science and Engineering, University of Wisconsin-Madison, USA
   index: 3

date: 31 August 2024
bibliography: paper.bib

---

# Summary

`AniSOAP` is a package that translates coarse-grained molecular configurations into tensorial representations that are ideal for supervised machine-learning models of thermodynamic quantities and unsupervised data-driven analyses. This generalization of existing spherical ML representations therefore aims to bridge the gap between two scientific communities: The machine-learned atomistic simulation community, whose primary concern is obtaining fast and (quantum) accurate descriptions of the complex interactions occuring between (spherical) atoms, and the coarse-grained and colloid modeling community, whose primary concern is understanding emergent behavior of macroscopic particles with (plausibly) complex geometries. `AniSOAP` provide a common framework to answer scientific questions at the intersection of these two fields.

# Statement of need

Machine learning (ML) has greatly advanced atomistic molecular dynamics (MD), enabling both quick and quantum-accurate MD simulations and offering powerful tools for analyzing simulation results. Key to these advancements are the increasingly sophisticated strategies and software used to featurize atomistic environments to capture subtle differences between molecular configurations; these have enabled supervised, semisupervised, and unsupervised studies across a wide variety of chemical spaces[@behler_atom-centered_2011; @bartok_representing_2013; @de_comparing_2016; @drautz_atomic_2019; @cersonsky_data-driven_2023]. However, these techniques are largely limited to atomistic resolution, and fall short in reliably describing coarse-grained entities (``particles'' or groups of atoms) that have anisotropic and non-spherical geometries. In such cases, it is essential to resolve the orientation-dependence of their interactions with neighboring particles.

While software exists for creating spherical atomistic descriptors (e.g. DScribe[@dscribe], librascal[@librascal; @librascal_paper], featomic[@featomic]), currently, there are currently no available packages to implement their anisotropic counterparts. In this software, we present the implementation of AniSOAP, an anisotropic generalization of the popular Smooth Overlap of Atomic Positions (SOAP) featurization[@bartok_representing_2013]. SOAP, like other atomistic representations, offers a concise and numerically efficient parameterization of atomistic environments, incorporating correlations of the central atom with up to two of its neighbors. Along with several methods that refine its construction [@nice, @dusson2022], it provides a framework to systematically build higher-order geometric and symmetrized ``fingerprints'' that can be used to model complex interaction potentials and extract machine-learning-enabled insights from data. AniSOAP extends SOAP by additionally incorporating the non-sphericity of particles in terms of particle geometry and orientation. Hence, AniSOAP can be used as a geometrically accurate, high body-order coarse-grained (CG) featurization of molecular and macromolecular systems. As a generalization of SOAP, AniSOAP retains full compatibility with SOAP, and the two representations can be used together to represent molecules at multiple resolutions.

This technology is motivated by the need to reduce computational complexity in molecular studies, either from a performance-based standpoint (despite high-performance capabilities, many systems cannot be simulated with all-atom resolution in reasonable times) or from a conceptual standpoint, as we may not always want to analyze the behavior of _atoms_, but superatomic entities, such as functional groups. 
While CG has made enormous strides in accelerating simulation and analysis, it's important to note that most CG techniques still reduce macromolecules to a set of spherical beads. This is often adequate for  dilute simulations, where the anisotropy of a group of atoms may not play a big role in determining system behavior. However, in concentrated or highly condensed systems (e.g., liquid crystals, glasses, molecular crystals), molecular anisotropy often plays a huge role in determining system behavior, and coarse-graining entities can lose a significant amount of information on the interactions within the system. Still, physically-grounded and tractable simulation of shaped particles is a long-standing challenge. The flexibility of the AniSOAP representation coupled with learning algorithms may help address these challenges, as we can build anisotropic potentials off of high-quality, first-principles data.

By incorporating anisotropy, AniSOAP extends the advances of atomistic machine learning to coarse-grained and mesoscale systems. With the flexibility of machine learning algorithms and anchored by the concepts underlying the feature-rich SOAP representation, AniSOAP provides a promising conduit for providing data-driven insights to many chemical systems at different length and time-scales.

# What is an AniSOAP vector?
An AniSOAP vector is a series of numbers that represents a system of ellipsoidal particles, which in the most general case, consists of $\geq 1$ particles with different positions, shapes, orientations, and species identifications. Essentially, this is very similar to other point-cloud, many-body representations used in atomistic machine learning, like SOAP[@bartok_representing_2013] or ACE[@drautz_atomic_2019], except for the fact that AniSOAP vectors incorporate additional geometric information. Specific use-cases of AniSOAP can be seen in our paper here[@lin_expanding_2024].

# Implementation details
The AniSOAP package currently takes in as input a list of frames in the `Atomic Simulation Environment` package[@hjorth_larsen_atomic_2017]. Each frame contains the particles' positions, dimensions, and orientations. If using periodic boundary conditions, the frame also needs the dimensions of the unit cell. Additional information about each frame can also be stored (e.g. the system energy) and used as a target for supervised ML.

With this information, one can construct an `EllipsoidalDensityProjection` object, whose main functionality is to calculate the expansion coefficients of the density field in each frame.
Procedurally, calculating the expansion coefficients amounts to repeatedly and recursively computing high-order moments of an underlying multivariate gaussian, as outlined in [@lin_expanding_2024]. For efficient computation, we have ported these highly-repeated calculations to Rust, a high-performance compiled language.

One can take Clebsch-Gordan products of these expansion coefficients to create higher body-order descriptors, and we optimize this step by caching intermediate results with a Least Recently Used (LRU) cache.

As many users will be primarily interested in power-spectrum representations, we provide all the functionality required for these processes, and also provide the convenience method `power_spectrum` to calculate the 3-body descriptors of each frame. 

The library is thoroughly tested, with unit-tests to test basic functionality, integration-tests to ensure that AniSOAP vectors are calculated correctly, and caching and speed tests to ensure that our aforementioned optimizations yield faster code. These tests are integrated into a Github CI, and we ensure that future features should necessistate additional tests and should pass existing ones.

# Conclusion and future developments
AniSOAP is a powerful featurization that can be used for supervised and unsupervised analyses of molecular systems. AniSOAP is under active development and we envision it being used in a wide variety of contexts. Our main future development goals involve using AniSOAP as the underlying representation for machine-learned anisotropic potentials, and to understand how the relationship behind AniSOAP and its all-atom counterpart SOAP fits into the broad theory of bottom-up coarse-graining. We hope that accomplishing these goals can enable fast, accurate, and interpretable macromolecular or colloidal simulations.

# Acknowledgements
This project was funded by the Wisconsin Alumni Research Fund (R.K.C.), NSF through the University of Wisconsin Materials Research Science and Engineering Center (Grant No. DMR-2309000, A.L.), the European Research Council (ERC) under the research and innovation program (Grant Agreement No. 101001890-FIAMMA, J.N.), and the MIT Postdoc Fellowship for Excellence (J.N.).

We extend our un-ending gratitude to Guillaume Fraux and the developers of featomic for fielding our many questions during the implementation and validation of AniSOAP, and Kevin Kazuki Huguenin-Dumittan for building the first iteration of AniSOAP.

# References
