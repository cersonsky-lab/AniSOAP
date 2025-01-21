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
 - name: Laboratory of Computational Science and Modeling, École Polytechnique Fédérale de Lausanne, Switzerland
   index: 2
 - name: Department of Computer Science and Engineering, University of Wisconsin-Madison, USA
   index: 3

date: 31 August 2024
bibliography: paper.bib

---

# Summary

`AniSOAP` is a package that creates Machine Learning (ML) representations of non-spherical particle configurations; these representations can then be used in ML-driven simulations and analyses. This generalization of existing spherical ML representations therefore aims to bridge the gap between two scientific communities: The machine-learned atomistic simulation community, whose primary concern is obtaining fast and (quantum) accurate descriptions of the complex interactions occuring between (spherical) atoms, and the coarse-grained and colloid modeling community, whose primary concern is understanding emergent behavior of macroscopic particles with (plausibly) complex geometries. `AniSOAP` provide a common framework to answer scientific questions at the intersection of these two fields.

# Statement of need

Machine learning (ML) has greatly advanced atomistic molecular dynamics (MD), enabling 1. Quick and quantum-accurate MD simulations, and 2. Robust techniques to analyze simulation results. Key to these advancements are the increasingly sophisticated strategies used to *featurize* atomistic environments to sensitively differentiate different configurations; this has enabled supervised, semisupervised, and unsupervised studies across a wide variety of chemical spaces[@behler_atom-centered_2011; @bartok_representing_2013; @de_comparing_2016; @cersonsky_data-driven_2023] (more citations?). However, these techniques are often limited to atomistic resolution, as coarse-grained entities (``particles'' or groups of atoms) are not spherical and thus cannot be fully resolved with many atom-inspired ML representations. In these contexts, where molecules or particles have complex, anisotropic geometry, it is important to resolve the orientation-dependence of their interactions with neighboring particles. 

In this software, we present the implementation of AniSOAP, an anisotropic generalization of the popular Smooth Overlap of Atomic Positions (SOAP) featurization[@bartok_representing_2013]. The power of SOAP, like other atomistic featurizations, lies in the incorporation of relevant physical information and symmetries, creating numerically efficient, symmetrized, and high-body order ``fingerprints'' that are commonly used in creating complex interaction potentials and machine-learning-enabled analyses. AniSOAP extends SOAP by additionally enabling information about the particle geometry and orientation within its featurization. Hence, AniSOAP can be used as a geometrically accurate, high body-order coarse-grained (CG) featurization of molecular and macromolecular systems. As a generalization of SOAP, AniSOAP retains full compatibility with SOAP, and the two representations can be used together to represent molecules at multiple resolutions.

This technology is motivated by the need to reduce computational complexity in molecular studies, either from a performance-based standpoint (despite high-performance capabilities, many systems cannot be simulated with all-atom resolution in reasonable times) or from a conceptual standpoint, as we may not always want to analyze the behavior of _atoms_, but superatomic entities, such as functional groups. 
While CG has made enormous strides in accelerating simulation and analysis, it's important to note that most CG techniques still reduce macromolecules to a set of spherical beads. This is often adequate for  dilute simulations, where the anisotropy of a group of atoms may not play a big role in determining system behavior. However, in concentrated or highly condensed systems (e.g., liquid crystals, glasses, molecular crystals), molecular anisotropy often plays a huge role in determining system behavior, and coarse-graining entities can lose a significant amount of information on the interactions within the system. Still, physically-grounded and tractable simulation of shaped particles is a long-standing challenge. The flexibility of the AniSOAP representation coupled with learning algorithms may help address these challenges, as we can build anisotropic potentials off of high-quality, first-principles data.

By incorporating anisotropy, AniSOAP extends the advances of atomistic machine learning to coarse-grained and mesoscale systems. With the flexibility of machine learning algorithms and anchored by the concepts underlying the feature-rich SOAP representation, AniSOAP provides a promising conduit for providing data-driven insights to many chemical systems at different length and time-scales.

# What is an AniSOAP vector?
An AniSOAP vector is a series of numbers that represents a system of ellipsoidal particles, which in the most general case, consists of $\geq 1$ particles with different positions, shapes, orientations, and species identifications. Essentially, this is very similar to other point-cloud, many-body representations used in atomistic machine learning, like SOAP[@bartok_representing_2013] or ACE[@drautz_atomic_2019], except for the fact that AniSOAP vectors incorporate additional geometric information. Specific use-cases of AniSOAP can be seen in our paper here[@lin_expanding_2024].

# What does the AniSOAP package do to take a configuration and construct the representation?
The AniSOAP package currently takes in as input a list of frames in the `Atomic Simulation Environment` package[@hjorth_larsen_atomic_2017]. Each frame contains the particles' positions, dimensions, and orientations. If using periodic boundary conditions, the frame also needs the dimensions of the unit cell. Additional information about each frame can also be stored (e.g. the system energy) and used as a target for supervised ML.

With this information, one can construct an `EllipsoidalDensityProjection` object, whose main functionality is to calculate the expansion coefficients of the density field in each frame. One can take Clebsch-Gordan products of these expansion coefficients to create higher body-order descriptors. We provide all the functionality required for these processes, and also provide the convenience method `power_spectrum` to calculate the 3-body descriptors of each frame.

# Conclusion and future developments
AniSOAP is a powerful featurization that can be used for supervised and unsupervised analyses of molecular systems. AniSOAP is under active development and we envision it being used in a wide variety of contexts. Our main future development goals involve using AniSOAP as the underlying representation for machine-learned anisotropic potentials, and to understand how the relationship behind AniSOAP and its all-atom counterpart SOAP fits into the broad theory of bottom-up coarse-graining. We hope that accomplishing these goals can enable fast, accurate, and interpretable macromolecular or colloidal simulations.

# Acknowledgements
This project was funded by the Wisconsin Alumni Research Fund (R.K.C.), NSF through the University of Wisconsin Materials Research Science and Engineering Center (Grant No. DMR-2309000, A.L.), and the European Research Council (ERC) under the research and innovation program (Grant Agreement No. 101001890-FIAMMA, J.N.).

We extend our un-ending gratitude to Guillaume Fraux and the developers of rascaline for fielding our many questions during the implementation and validation of AniSOAP, and Kevin Kazuki Huguenin-Dumittan for building the first iteration of AniSOAP.

# References
