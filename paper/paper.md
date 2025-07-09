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

`AniSOAP` is a package that translates coarse-grained molecular configurations into tensorial representations that are ideal for supervised machine-learning models of thermodynamic quantities and unsupervised data-driven analyses. It generalizes several popular spherical representations for atomistic ML and aims to bridge the gap between two scientific communities: the machine-learning enabled atomistic simulation community, which leverages ML to accelerate and refine quantum modeling of complex interactions between spherical atoms, and the coarse-grained and colloid modeling community, which focuses on understanding emergent behavior of macroscopic particles with (plausibly) complex geometries. AniSOAP provides a common framework to tackle scientific questions at the intersection of these two fields.

# Statement of need

Machine learning (ML) has greatly advanced atomistic molecular dynamics (MD), enabling both quick and quantum-accurate simulations and offering powerful tools for analyzing simulation results. Key to these advancements are the increasingly sophisticated strategies and software used to featurize atomistic environments that capture subtle differences between molecular configurations, either explicitly [@behler_atom-centered_2011; @bartok_representing_2013; @drautz_atomic_2019] or implicitly [@mace; @nequip]. These techniques have enabled supervised, semisupervised, and unsupervised studies across a wide variety of chemical spaces [@de_comparing_2016; @Cheng2019; @cersonsky_data-driven_2023]. However, these techniques are largely limited to atomistic resolution, and fall short in reliably describing coarse-grained entities (``particles'' or groups of atoms) that have anisotropic geometries, where it is essential to resolve the orientation-dependence of their interactions with neighboring particles.

While many implementations construct spherical atomistic descriptors (e.g. DScribe [@dscribe], librascal [@librascal; @librascal_paper], featomic [@featomic]), currently, there are no available packages for their anisotropic counterparts. In this software, we present the implementation of AniSOAP, an anisotropic generalization of the popular Smooth Overlap of Atomic Positions (SOAP) featurization [@bartok_representing_2013]. SOAP, like other atomistic representations, offers a concise and numerically efficient parameterization of atomistic environments, incorporating correlations of the central atom with up to two of its neighbors. Along with several methods that refine its construction [@nice; @dusson2022], it provides a framework to systematically build higher-order geometric and symmetrized ``fingerprints'' that can be used to model complex interaction potentials and extract machine-learning-enabled insights from data. AniSOAP extends this framework by allowing individual particles to be non-spherical. Hence, AniSOAP can be used as a geometrically accurate, high body-order coarse-grained (CG) featurization of molecular and macromolecular systems. As AniSOAP retains full compatibility with SOAP, two representations can be used together to represent molecules at both atomistic and CG resolutions.

This is especially relevant as many systems cannot be simulated with all-atom resolution in reasonable times, despite high-performance capabilities of machine-learning interatomic potentials. Additionally, from a conceptual standpoint, we may not always want to analyze the behavior of _atoms_, but superatomic entities, such as functional groups. 
Most CG techniques still reduce macromolecules to a set of spherical beads. While often adequate for dilute simulations, this reduction to spherical beads significantly oversimplifies anisotropy, which is curcial in condensed systems (e.g., liquid crystals, glasses, molecular crystals). The flexibility of the AniSOAP representation coupled with learning algorithms can help address these challenges, as we can infer anisotropic coarse-graining from high-quality, first-principles data. This featurization could be easily plugged into other software, enabling mesoscopic simulations and enable data-driven insights to many chemical systems at different time and length scales.

The `AniSOAP` package enables the creation of AniSOAP feature vectors, which represent systems of ellipsoidal particles. Analogous to how SOAP or ACE create _atom_-centered representations, `AniSOAP` creates _particle_-centered representations, where a particle could be a single atom or a coarsened group of several atoms. @lin_expanding_2024 provided demonstrations of specific use-cases of AniSOAP.

# Implementation details
The AniSOAP package currently takes in as input a list of frames in the `Atomic Simulation Environment` package [@hjorth_larsen_atomic_2017]. Each frame contains the particles' positions, dimensions, and orientations. If using periodic boundary conditions, the frame also needs to contain the dimensions and orientations of the unit cell. Additional information about each frame can also be stored (e.g. the system energy) and used as a target for supervised ML.

With this information, one can construct an `EllipsoidalDensityProjection` object, whose main functionality is to calculate the expansion coefficients of an anisotropic density field in each frame via the `transform` method.
Procedurally, calculating the expansion coefficients amounts to repeatedly and recursively computing high-order moments of an underlying multivariate Gaussian [@lin_expanding_2024]. For efficient computation, we have ported these highly-repeated calculations to Rust, a high-performance compiled language. The intermediate results utilize the `metatensor TensorMap` format [@guillaume_fraux_metatensor_2024], which efficiently stores the AniSOAP featurizations and their associated metadata.

One can take Clebsch-Gordan products of these expansion coefficients to create higher body-order descriptors, and we optimize this step by caching intermediate results with a Least Recently Used (LRU) cache. This functionality is enabled by the `anisoap.metatensor_utils` module in AniSOAP, in particular, the `cg_combine` function.

As many users will be primarily interested in power-spectrum (i.e. 3-body) representations, we provide all the functionality required for these processes, and also provide the convenience method `power_spectrum` to calculate the 3-body descriptors of each frame. By default, this method returns the featurization as a $n_{\mathrm{samples}}\times n_{\mathrm{features}}$ `NumPy` array, which can be used as input into a machine learning algorithm. Alternatively, by setting the keyword argument `mean_over_samples=False`, this method returns a `metatensor TensorMap` object, which contains the power-spectrum representation for each atom in each frame as well as associated metadata. This is a much larger, unaggregated data object that requires more processing before it can be used in an ML algorithm. Examples of the various ways of creating AniSOAP representations can be found in the examples section of our documentation [@anisoap_documentation].

The library is thoroughly tested and documented, with unit-tests to test basic functionality, integration-tests to ensure that AniSOAP vectors are calculated correctly, and caching and speed tests to ensure that our aforementioned optimizations yield faster code. These tests are integrated into a Github Continuous Integration (CI), and we ensure that future features should necessistate additional tests and should pass existing ones.

# Conclusion and future developments
AniSOAP is a powerful featurization that can be used for supervised and unsupervised analyses of molecular systems. AniSOAP is under active development and we envision it being used in a wide variety of contexts. Our main future development goals involve using AniSOAP as the underlying representation for machine-learned anisotropic potentials, and to understand how the relationship between AniSOAP and its all-atom counterpart SOAP fits into the broad theory of bottom-up coarse-graining. We hope that accomplishing these goals can enable fast, accurate, and interpretable macromolecular or colloidal simulations.

# Acknowledgements
This project was funded by the Wisconsin Alumni Research Fund (R.K.C.), NSF through the University of Wisconsin Materials Research Science and Engineering Center (Grant No. DMR-2309000, A.L.), the European Research Council (ERC) under the research and innovation program (Grant Agreement No. 101001890-FIAMMA, J.N.), and the MIT Postdoc Fellowship for Excellence in Engineering (PFPFEE, J.N.).

We extend our un-ending gratitude to Guillaume Fraux and the developers of featomic for fielding our many questions during the implementation and validation of AniSOAP, and Kevin Kazuki Huguenin-Dumittan for building the first iteration of AniSOAP.

# References
