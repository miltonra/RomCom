
Conventions
-----------------

RomCom abides by the spirit of `PEP 8 <https://peps.python.org/pep-0008/>`_,
which allows conventions to be flexed, but only when needs must.
`Annotations`_ demonstrate this.

RomCom abides by the letter of `PEP 8 <https://peps.python.org/pep-0008/>`_ except for:

* source code line wraps at 114 characters, to fit web browsers.
* a neurotic aversion to ugly under_scores and BLOCK_CAPS.

Names
^^^^^^^^^^^

RomCom views any Python entity as falling into one of three categories, each with its own naming convention.

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


Character Set
^^^^^^^^^^^^^^^^^^^

RomCom assumes `UTF-8 encoding <https://www.w3schools.com/charsets/ref_html_utf8.asp>`_ but actually uses only one non-ASCII character, the Box Drawing Light Vertical │ (U+2502).
It denotes probablistic conditioning, which delimits categories in a semantically coherent and instructive way.
Semantics affect function in RomCom file generation, so they are expressed using a character unlikely to occur in user data.
User data may safely contain any printable UTF character except ``│``, including the common Vertical Line `|` (U+007C, ASCII x7C) which is noticeably shorter.

If needs must, the category delimiter can be changed globally in user code by resetting the constant
``rc.data.models.Normalization.defaultMetaData['category delimiter']`` before any RomCom code is executed.
