
Content
-------------

Names
^^^^^^^^^^^^

RomCom views any Python entity as falling into one of three categories, each with its own naming convention.

namespace
+++++++++++++
Packages and modules are referred to in lowercase (abbreviations or single words).

Class
++++++++++++++
Classes and Types are referred to in UpperCamelCase. The documentation italicizes *AbstractClasses* --
i.e those not instantiable as objects.

.. _object:
object
+++++++++++++
Anything which is neither namespace nor Class is referred to in lowerCamelCase.
This includes classInstances, classAttributes and instanceAttributes.


Objects
^^^^^^^^^

Any `object`_ may be further classified as

method
++++++++++++++
A Python function.

property
+++++++++++++
A method which gets or sets a private attribute of a classInstance. Often read-only,
if not, setting the property has side effects. There are no classProperties in RomCom,
because Python does not support them.

attribute
+++++++++++++
Any object which not a Python Function. Data, so far as we are concerned.


Constants
^^^^^^^^^^^

Constancy in RomCom is determined by scope, not name.
Every namespace or classAttribute in RomCom is a constant which must not be modified.
Bespoke constants should be implemented by SubClassing, leaving the RomCom parent unmodified.
An UpperCamelCase constant (e.g. ModuleAttribute or ClassAttribute) refers to a Class.
A lowerCamelCase constant (e.g. moduleAttribute or classAttribute) does not.


Protocols
^^^^^^^^^^^
    In the context of object-oriented programming, a protocol is an informal interface, defined only in documentation and not in code. [Fluent Python pp.402]

A Protocol documents an interface a Class supports via undocumented :term:`dunder methods`.
Protocols are pure documentation in code, and are essentially meaningless outside a Class definition.
Protocols are implemented in RomCom as (docstring only) ClassAttributes which are Classes themselves (Python BaseExceptions, in fact), and are named in UpperCamelCase.

.. _ecosystem:

Ecosystem
^^^^^^^^^^^

The facilities and idiom of RomCom naturally depend on the Python ecosystem.
Users may struggle if they are unacquainted with the key Python libraries supporting RomCom.

RomCom cannot live without

* `pandas <https://pandas.pydata.org/>`__ for high-level data representation. Referred to as ``pd`` or ``Pd``.
* `numpy <https://numpy.org/>`__ for intermediation. Referred to as ``np`` or ``Np``.
* `torch <https://pytorch.org/>`__ for numerical methods. Referred to as ``tc`` or ``Tc``.

RomCom would be impaired without

* `SALib <https://salib.readthedocs.io/en/latest/index.html>`__ for benchmarking functions.
* `scipy <https://www.scipy.org/>`__ for occasional statistical methods.

A huge debt of gratitude is owed to those responsible for these libraries.
