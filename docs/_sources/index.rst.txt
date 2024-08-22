
RomCom User Guide
==========================================================================================================================================

.. rubric:: Reduced Order Modelling via Global Sensitivity Analysis using Gaussian Process Regression, by Robert A. Milton of The University of Sheffield


.. error:: This documentation is currently under construction

Welcome to the RomCom Python package, which implements Reduction of Order by Marginalization (:term:`ROM`) via Global Sensitivity Analysis (:term:`GSA`)
using Gaussian Process Regression (:term:`GPR`). The mathematics behind this software is covered in some detail in a
`paper currently under peer review for publication <https://github.com/miltonra/RomDoc/blob/dev/Sobol%20Matrices/Sobol%20Matrices.pdf>`_.

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
        RomCom deals exclusively with the variance based method of Ilya M. Sobol.
        This has been somewhat extended, as described in gory technical detail in
        [`Milton et al. 2023 <https://github.com/C-O-M-M-A/rom-papers/blob/main/Sobol%20Matrices/Sobol%20Matrices.pdf>`_].

    ROM
        Reduction of Order by Marginalization. A novel approach to locating an Active Subspace (AS) using conditional variances or Sobol' indices.
        In the Active Subspace technique [`Constantine 2014 <https://epubs.siam.org/doi/book/10.1137/1.9781611973860>`_]
        the input basis is rotated to align with the eigenvectors of the squared Jacobian vector.
        In ROM, the input basis is rotated to maximise the Sobol' index of the first :math:`m` inputs.

Installation
---------------

Detailed installation instructions are contained in RomCom's
`README.md <https://github.com/miltonra/RomCom/blob/dev/README.md>`_.


Contents
----------

.. toctree::
    :maxdepth: 2

    pages/intro
    pages/api/api
    glossary


Search
--------------------

:ref:`search`

:ref:`genindex`
