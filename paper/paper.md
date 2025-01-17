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
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Seonwoo Hwang
    orcid: 0000-0000-0000-0000
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
An AniSOAP vector is a series of numbers that represents a system of ellipsoidal particles, which in the most general case, consists of $\geq 1$ particles with different positions, shapes, orientations, and species identifications. Essentially, this is very similar to other point-cloud representations used in atomistic machine learning, like SOAP[@bartok_representing_2013] or ACE[@drautz_atomic_2019], except for the fact that AniSOAP vectors incorporate additional geometric information. 

The AniSOAP vector captures $n$-body correlations between different particles, which means that it encodes information about a central particle and its interactions with $(n-1)$ of its neighbors. In the majority of cases, we take $n=3$, which we call "the power spectrum", which encodes information about each particle, and the distance and subtended angle from the central particle to each possible pair of two neighboring particles within a cutoff radius. Ocassionally, we take $n=2$, called "the radial spectrum" which only captures all pairwise interactions up to a cutoff radius. 

The radial spectra is built by expanding local atom density fields in terms of radial basis functions and spherical harmonics -- this construction ensures that the values are translationally, rotationally, and permutationally invariant. To obtain higher body order, we take higher order correlations of these radial spectra. These entries are stored in a multidimensional array using the `TensorMap` format in the package `metatensor`[@guillaume_fraux_metatensor_2024]; this multidimensional array is usually flattened into a 1-dimensional representation to be used in ML algorithms. These vectors are information-rich, and can be used in various machine learning models. Traditionally, the power spectrum is used as an input to shallower machine learning models, like gaussian processes or kernel/linear regresssion, to learn system energies and forces. The AniSOAP vector can also be used as an order parameter to describe phase transitions. Specific use-cases of AniSOAP can be seen in our paper here[@lin_expanding_2024].

# What does the AniSOAP package do to take a configuration and construct the representation?
The AniSOAP package currently takes in as input a list of frames in the `Atomic Simulation Environment` package[@hjorth_larsen_atomic_2017]. Each frame contains the particles' positions, dimensions, and orientations. If using periodic boundary conditions, the frame also needs the dimensions of the unit cell. Additional information about each frame can also be stored (e.g. the system energy) and used as a target for supervised ML. Note that while ASE terminology calls each particle an "atom", in the case of AniSOAP, the ellipsoidal bodies are often used to describe groups of atoms.

With this information, one can construct an `EllipsoidalDensityProjection` object, whose main functionality is to calculate the expansion coefficients of the density field in each frame. One can take Clebsch-Gordan products of these expansion coefficients to create higher body-order descriptors. We provide all the functionality required for these processes, and an example of constructing an arbitrarily high body-order descriptor is shown in an example. We also provide the convenience method `power_spectrum` and `radial_spectrum` to calculate the 3-body and 2-body descriptors of each frame.

# Conclusion and future developments
AniSOAP is a powerful featurization that can be used for supervised and unsupervised analyses of molecular systems. AniSOAP is under active development and we envision it being used in a wide variety of contexts. Below, we outline two AniSOAP development goals.

Our first goal is to directly use the wide variety of ML driven techniques used in the atomistic simulation community to derive physical insights on the molecular level. In particular, one major use case of AniSOAP is to derive Machine-Learned Inter-_Particle_ Potentials (MLIPPs), which requires learning both energies and forces. While forces can be learned directly, it is preferable to obtain froces by taking gradients of learned energies as it enforces conservation of energy. We are currently working on efficient implementations of AniSOAP gradients to enable MD simulations.

We are also interested in unifying the theory of MLIPPs with the vast theory and developments of bottom-up coarse-graining. While bottom-up coarse-graining has recently benefited from significantly neural network potentials[@wang_machine_2019; @majewski_machine_2023; @wilson_anisotropic_2023], such techniques typically have large data requirements due to the neural network architecture, which require more training data than shallower models, and the inherently low signal-to-noise of the training set due to high mapping degeneracy in coarse-grained models[@durumeric_machine_2023]. AniSOAP may remedy this data-barrier on two different fronts: AniSOAP based potentials tend to be shallow (linear/kernel regression), and the geometric accuracy of AniSOAP particles will significantly decrease the mapping degeneracy. We hope that these benefits can be tied with existing theoretical frameworks on anisotropic coarse-graining[@nguyen_systematic_2022] to create robust, interpretable, and practical tools to perform accurate molecular coarse-grained simulations.

# References

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

This project was funded by the Wisconsin Alumni Research Fund (R.K.C.), NSF through the University of Wisconsin Materials Research Science and Engineering Center (Grant No. DMR-2309000, A.L.), and the European Research Council (ERC) under the research and innovation program (Grant Agreement No. 101001890-FIAMMA, J.N., K.K.H.D.).

We extend our un-ending gratitude to Guillaume Fraux and the developers of rascaline for fielding our many questions during the implementation and validation of AniSOAP. Kevin?
