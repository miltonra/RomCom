
API
==================

.. include:: api/api.rst

Code Conventions
-----------------

When ordered alphabetically, the API hierarchy is best understood top-down, and used bottom-up.
So the ``base`` module is fundamental to everything, but the ``user`` module is the best interface
to achieve most tasks.
The code dependency structure is naturally top-down alphabetically.

These observations apply within submodules too, so the ``data`` module depends only on ``base``,
and consists of three submodules:
``data.functions`` which is independent of ``data.models`` which is independent of ``data.samples``.

.. glossary::
    :sorted:

    packages/modules
        Package and module names are lowercase words or abbreviations.

    ClassNames/TypeNames
        Class and type names are UpperCamelCase.

    functionNames/variableNames
        Function and variable names are lowerCamelCase.

    Constants/constants
        Constancy is determined by scope, not name. Any Class or module attribute is a constant,
        which must not be modified.
        An UpperCamelCase constant refers to a Class/Type, a lowerCamelCase constant does not.
