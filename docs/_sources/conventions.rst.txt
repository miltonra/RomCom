
RomCom views Python entities as falling into one of three categories, each with its own naming convention.

.. glossary::

    namespace
        Packages and modules are referred to in lowercase (abbreviations or single words).

    Class
        Classes and Types are referred to in UpperCamelCase. The User Guide italicizes *AbstractClasses* --
        i.e those not instantiable as objects.

    object
        Anything which is neither :term:`namespace` nor :term:`Class` is referred to in lowerCamelCase.
        This includes classInstances, classAttributes and instanceAttributes.


Constants
^^^^^^^^^^^

Constancy in RomCom is determined by scope, not name.
Every namespace or Class attribute in RomCom is a constant which must not be modified.
Bespoke constants should be implemented by subclassing, leaving the RomCom parent unmodified.
An UpperCamelCase constant refers to a Class, a lowerCamelCase constant does not.


Annotations
^^^^^^^^^^^^^^^^^^

RomCom is copiously annotated, in the spirit of a statically typed library.
Type annotation is the subject of ``rc.base.definitions``, which simply extends
`Python Standard Library typing <https://docs.python.org/3/library/typing.html>`_.
For aliasing and naming conflict resolution, ``rc.base.definitions`` imports the core libraries ``pandas``,
``numpy``, ``torch``, and wraps them in aliasing classes ``Pd``, ``Np``, ``Tc``. This provides the namespace/Class duals
:term:`pd | Pd`, :term:`np | Np`, and :term:`tc | Tc`.


Flexibility and PEP 8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the previous paragraph demonstrates, RomCom conventions may be flexed when needs must.
This aligns with the spirit of `PEP 8 <https://peps.python.org/pep-0008/>`_,
to which RomCom abides, except for source code line wraps at 114 characters and
a neurotic aversion to ugly under_scores and BLOCK_CAPS.







