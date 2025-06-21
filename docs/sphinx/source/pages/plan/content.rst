
Content
===============================

RomCom views any Python entity as falling into one of three categories, each with its own naming convention.

.. glossary::

    namespace
        Packages and modules are referred to in lowercase (abbreviations or single words).

    Class
        Classes and Types are referred to in UpperCamelCase. The User Guide italicizes *AbstractClasses* --
        i.e those not instantiable as objects.

    object
        Anything which is neither namespace nor Class is referred to in lowerCamelCase.
        This includes classInstances, classAttributes and instanceAttributes.


Objects
-----------

Any :term:`object` may be classified as

.. glossary::

    method
        A Python function.

    property
        A method which gets or sets a private attribute of a classInstance. Often read-only,
        if not, setting the property has side effects. There are no classProperties in RomCom,
        because Python does not support them.

    attribute
        Any object which not a Python Function. Data, so far as we are concerned.


Constants
------------

Constancy in RomCom is determined by scope, not name.
Every namespace or class attribute in RomCom is a constant which must not be modified.
Bespoke constants should be implemented by subclassing, leaving the RomCom parent unmodified.
An UpperCamelCase constant refers to a Class, a lowerCamelCase constant does not.


Protocols
------------
