
Plan
=============================

Form
-------

Structure
^^^^^^^^^^

    All right everyone, line up alphabetically according to your height. [Casey Stengel]

The RomCom library is a functional hierarchy of namespaces, organised by package, then module.
It is also ordered alphabetically.

library
++++++++
Refers to the :doc:`RomCom library <api>` imported as ``rc``.

packages
+++++++++
The RomCom library is organised into alphabetically ordered packages.
Each package adds a layer of functionality depending only on alphabetically prior packages.
For example, the :doc:`api/rc/base/index` package is foundational, whereas the :doc:`api/rc/tasks/index` package is the most user-friendly and efficient interface
for performing common tasks. In between, the :doc:`api/rc/data/index` package is used to represent your data in RomCom.
Packages serve only to organise the library.

modules
+++++++++
Each package is organised into alphabetically ordered modules providing functionality.
Each module adds a layer of functionality depending only on alphabetically prior modules.
For example, the :doc:`api/rc/tasks/index/scripts` module is the gateway to common tasks, employing result summarising functionality provided by :doc:`api/rc/tasks/index/results``.
All content resides in modules.

Style
^^^^^^^
RomCom is object-oriented and explicitly statically typed.
As with all questions of Python style, this is mainly but not entirely true.

    A foolish consistency is the hobgoblin of little minds.
    [`PEP 8 <https://peps.python.org/pep-0008/>`__, second paragraph]

The rules and conventions of RomCom will flex when needs must.
**The sole purpose of type, structure and protocol in RomCom is to document RomCom's intention**, not to legislate usage.

The coding idiom of RomCom woefully impersonates three essential sources

* `The Lizard Book <https://www.fluentpython.com/>`__  by Luciano Ramalho.
* `Effective Python <https://effectivepython.com/>`__  by Brett Slatkin.
* `Python Docs <https://docs.python.org/3/>`__  by the Python Software Foundation.

Guide
^^^^^^^^^^^
Learning RomCom means getting to know the :doc:`api/rc/base/index`, :doc:`api/rc/data/index`, and :doc:`api/rc/tasks/index` packages.
These should suffice to perform many common tasks.

Mastering RomCom means getting to know the :doc:`api/rc/gpr/index`, :doc:`api/rc/gsa/index`, and :doc:`api/rc/rom/index` packages which do the heavy maths.
These can be leveraged to tailor RomCom to your every whim.

Supplementary to mastery of RomCom is the :doc:`api/rc/glr/index` package.
This provides linear regression facilities with an interface borrowed from :doc:`api/rc/gpr/index`. Occasionally useful.


.. include:: content.rst
