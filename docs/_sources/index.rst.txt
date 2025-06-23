
RomCom User Guide
===========================================================================================

.. rubric:: Reduced Order Modelling via GSA/GPR, by Robert A. Milton and Solomon F. Brown.


.. danger:: This documentation is currently under construction

These cards are clickable, preferably in order.

.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`download` Installation
        :link: https://github.com/miltonra/RomCom/tree/dev
        :link-type: url

        Installation instructions at the GitHub site README

    .. grid-item-card::  :octicon:`project-roadmap` Plan
        :link: pages/plan
        :link-type: doc

        The form and content of RomCom

    .. grid-item-card::  :octicon:`log` Glossary
        :link: pages/glossary
        :link-type: doc

        RomCom terms and abbreviations

.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`database` base
        :link: pages/base
        :link-type: doc

        Foundational type, storage and access facilities

    .. grid-item-card::  :octicon:`file` data
        :link: pages/data
        :link-type: doc

        Presenting your data to RomCom

    .. grid-item-card::  :octicon:`play` tasks
        :link: pages/tasks
        :link-type: doc

        Scripted computations for mainstream tasks

.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`package` Gaussian Process Regression
        :link: pages/gpr
        :link-type: doc

        GPR interpolates training data with uncertainty quantification

    .. grid-item-card::  :octicon:`package` Global Sensitivity Analysis
        :link: pages/gsa
        :link-type: doc

        GSA assesses the relevance of GPR inputs to outputs

    .. grid-item-card::  :octicon:`package` Reduced Order Modelling
        :link: pages/rom
        :link-type: doc

        ROM uses GSA to find the minimal set (Active Subspace) of GPR inputs


.. toctree::
    :maxdepth: 1
    :hidden:

    Plan <pages/plan>
    Glossary <pages/glossary>
    base <pages/base>
    data <pages/data>
    tasks <pages/tasks>
    GLR <pages/glr>
    GPR <pages/gpr>
    GSA <pages/gsa>
    ROM <pages/rom>
    API <pages/api>
    genindex

    :ref: `genindex`

