
RomCom User Guide
===========================================================================================

.. rubric:: Reduced Order Modelling via GSA/GPR, by Robert A. Milton and Solomon F. Brown.


.. danger:: This documentation is currently under construction


.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`info` Installation
        :link: https://github.com/miltonra/RomCom/tree/dev
        :link-type: url

        Instructions provided in the README.

    .. grid-item-card::  :octicon:`info` Glossary
        :link: glossary
        :link-type: doc

        RomCom terminology and abbreviations.

    .. grid-item-card::  :octicon:`info` Conventions
        :link: conventions
        :link-type: doc

        GSA assesses the relevance of a system's inputs to its outputs.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`package` Base
        :link: pages/base
        :link-type: doc

        GPR interpolates training data with uncertainty quantification.

    .. grid-item-card::  :octicon:`package` Data
        :link: pages/data
        :link-type: doc

        GSA assesses the relevance of a system's inputs to its outputs.

    .. grid-item-card::  :octicon:`package` Task
        :link: pages/task
        :link-type: doc

        ROM uses GSA to find the minimal set (Active Subspace) of inputs.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`package` Gaussian Process Regression
        :link: pages/gpr
        :link-type: doc

        GPR interpolates training data with uncertainty quantification.

    .. grid-item-card::  :octicon:`package` Global Sensitivity Analysis
        :link: pages/gsa
        :link-type: doc

        GSA assesses the relevance of a system's inputs to its outputs.

    .. grid-item-card::  :octicon:`package` Reduced Order Modelling
        :link: pages/rom
        :link-type: doc

        ROM uses GSA to find the minimal set (Active Subspace) of inputs.





Welcome
---------------

The RomCom Python library performs Reduction of Order by Marginalization (:term:`ROM`) Computations via Global Sensitivity Analysis (:term:`GSA`)
using Gaussian Process Regression (:term:`GPR`). The mathematics behind this software is covered in some detail in the following preprint
`Milton and Brown 2025 <https://arxiv.org/abs/2501.04602>`_.


.. glossary::

    GPR
        Gaussian Process Regression.
        A quite general technique for representing a functional dataset as a (Gaussian) stochastic process described thoroughly in
        [`Rasmussen and Williams 2005 <https://direct.mit.edu/books/book/2320/Gaussian-Processes-for-Machine-Learning>`_].

    GSA
        Global Sensitivity Analysis.
        This Assesses and ranks the relevance of a system's inputs to its outputs by a variety of methods covered broadly in
        [`Saltelli et al. 2007 <https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184>`_] and
        [`Razavi et al. 2021 <https://doi.org/10.1016/j.envsoft.2020.104954>`_].
        RomCom deals exclusively with the variance based method of Ilya M. Sobol, extended to novel `Sobol' matrices  <https://arxiv.org/abs/2501.04602>`_.

    ROM
        Reduction of Order by Marginalization. A novel approach to locating an Active Subspace (AS) using conditional variances or Sobol' indices.
        In the Active Subspace technique [`Constantine 2014 <https://epubs.siam.org/doi/book/10.1137/1.9781611973860>`_]
        the input basis is rotated to align with the eigenvectors of the squared Jacobian vector.
        In ROM, the input basis is rotated to maximise the Sobol' index of the first :math:`m` inputs.

We recommend reading this page up to the general :doc:`glossary`, which is intended for reference.
RomCom considers source code and User Guide as one: :doc:`conventions` and :doc:`glossary` entries apply equally to both, within format limitations.


Installation
---------------

Detailed installation instructions are contained in RomCom's
`README.md <https://github.com/miltonra/RomCom/blob/dev/README.md>`_.


Contents
----------

We recommend reading :doc:`pages/intro`, :doc:`pages/usage` and :doc:`pages/data`, in that order, then experiment.

.. toctree::
    :maxdepth: 1
    :hidden:

    Glossary <glossary>
    Base <pages/intro>
    Data <pages/data>
    Task <pages/usage>
    GPR <pages/gpr>
    GSA <pages/gsa>
    ROM <pages/rom>
    API <pages/api>
    genindex

    :ref: `genindex`

.. include:: conventions.rst

.. include:: refs.rst

.. include:: glossary.rst

