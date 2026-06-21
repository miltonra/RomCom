
Content
-------------
    We hold empty names.
        [The Name of the Rose]


Names
^^^^^^^^^^^^
RomCom views any Python entity as falling into one of three categories, each with its own naming convention.

.. _plannamespaces:
namespaces
+++++++++++++
These are packages and modules, which have lowercase names (abbreviations or single words).

.. _planClasses:
Classes
++++++++++++++
Classes and Types have UpperCamelCase names. RomCom documentation italicizes *AbstractClasses* --
i.e those not instantiable as objects.

.. _planobjects:
objects
+++++++++++++
Anything which is neither namespace nor Class has a lowerCamelCase name.
This includes classInstances, classAttributes and instanceAttributes.


Objects
^^^^^^^^^
RomCom :ref:`planobjects` may be exhaustively classified as

methods
++++++++++++++
Python functions, either moduleMethods, classMethods or instanceMethods.

properties
+++++++++++++
Python properties which get or set an instanceAttribute. If a property is not read-only,
you should assume that setting it has side effects. All RomCom properties are instanceProperties,
because Python does not (as yet properly) support classMethods.

attributes
+++++++++++++
Any :ref:`planobjects` which are neither method nor property. In other words, data.


Constants
^^^^^^^^^^^
Constancy in RomCom is determined by scope, not name.
Every moduleAttribute or classAttribute in RomCom is a constant which must not be modified.
Constants should only be tailored by SubClassing, leaving the RomCom parent unmodified.
Any lowercase constants (e.g. pl, or base) refer to :ref:`plannamespaces`.
Any UpperCamelCase constants (e.g. ModuleAttribute or ClassAttribute) refer to :ref:`planClasses`.
Any lowerCamelCase constants (e.g. moduleAttribute or classAttribute) refer to :ref:`planobjects`.


.. _Protocols:
*Protocols*
^^^^^^^^^^^
    In the context of object-oriented programming, a protocol is an informal interface, defined only in documentation and not in code.
        [The Lizard Book pp.402]

A *Protocol* is an *AbstractClass* which documents an interface implemented using undocumented :term:`dunder methods`.
*Protocols* are pure documentation in code, and are essentially meaningless outside a Class definition.
*Protocols* are implemented in RomCom as (docstring only) ClassAttributes which are Classes themselves (Python *BaseExceptions*, in fact),
They are named in *UpperCamelCaseP*, which is UpperCamelCase with a big P at the end which stands for *Protocol* and avoids tedious naming conflicts.


.. _ecosystem:
Ecosystem
^^^^^^^^^^^

The facilities and idiom of RomCom naturally depend on the Python ecosystem.
Users may struggle if they are unacquainted with the key Python libraries supporting RomCom.

RomCom cannot live without

* `polars <https://https://pola.rs/>`__ for high-level data representation. Referred to as ``pl``, its DataFrame ``df``.
* `numpy <https://numpy.org/>`__ for intermediation. Referred to as ``np`` or ``Np``.
* `torch <https://pytorch.org/>`__ for numerical methods. Referred to as ``tc`` or ``Tc``.

RomCom would be impaired without

* `SALib <https://salib.readthedocs.io/en/latest/index.html>`__ for benchmarking functions.
* `scipy <https://www.scipy.org/>`__ for occasional statistical methods.

A huge debt of gratitude is owed to those responsible for these libraries.
