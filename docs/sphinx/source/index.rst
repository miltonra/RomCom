
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
        :link: pages/plan/index
        :link-type: doc

        The form and structure of RomCom.

    .. grid-item-card::  :octicon:`log` Glossary
        :link: pages/glossary
        :link-type: doc

        RomCom terms and abbreviations

.. grid:: 3
    :gutter: 1

    .. grid-item-card::  :octicon:`database` base
        :link: pages/base/index
        :link-type: doc

        Foundational data typing, storage and access facilities

    .. grid-item-card::  :octicon:`file` data
        :link: pages/data/index
        :link-type: doc

        Presenting your data to RomCom

    .. grid-item-card::  :octicon:`play` task
        :link: pages/task
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

        GSA assesses the relevance of a system's inputs to its outputs

    .. grid-item-card::  :octicon:`package` Reduced Order Modelling
        :link: pages/rom
        :link-type: doc

        ROM uses GSA to find the minimal set (Active Subspace) of inputs


.. toctree::
    :maxdepth: 1
    :hidden:

    Plan <pages/plan/index>
    Glossary <pages/glossary>
    base <pages/base/index>
    data <pages/data/index>
    task <pages/task/index>
    GLR <pages/glr/index>
    GPR <pages/gpr/index>
    GSA <pages/gsa/index>
    ROM <pages/rom/index>
    API <pages/api>
    genindex

    :ref: `genindex`


.. _PEP8:  https://peps.python.org/pep-0008/
.. _TheLizardBook:  https://www.fluentpython.com/
.. _EffectivePython:  https://effectivepython.com/
.. _PythonDocs:  https://docs.python.org/3/

