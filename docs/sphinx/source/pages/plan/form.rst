
Form
=============

The RomCom library is an alphabetically ordered functional hierarchy of namespaces, ordered by package, then module.

.. glossary::

    library
        Refers to the RomCom library imported as ``rc``.

    package
        The RomCom library is organised into alphabetically ordered packages ``base``,
        ``data``, ``glr``, ``gpr``, ``gsa``, ``rom``, ``task``.
        Each package adds a layer of functionality depending only on alphabetically prior packages.
        For example, the ``rc.base`` package is foundational, whereas the ``rc.task`` package is the most user-friendly and efficient interface
        for performing common tasks.

    module
        Each package is organised into alphabetically ordered modules providing functionality.
        Each module adds a layer of functionality depending only on alphabetically prior modules.
        For example, the ``task.scripts`` module is the gateway to common tasks, employing results summarising functionality provided by ``task.results``.

Packages serve only to organise the library. All content resides in modules.

Style
---------

RomCom is object-oriented and strongly statically typed.
As with all questions of Python style, this is not entirely true.

    A foolish consistency is the hobgoblin of little minds.  [:ref:`PEP8`, second paragraph]

The rules and conventions of RomCom will flex when needs must.
The coding idiom of RomCom borrows from three essential references

* :ref:`TheLizardBook`  by Luciano Ramalho.
* :ref:`EffectivePython`  by Brett Slatkin.
* :ref:`PythonDocs`  by the Python Software Foundation.

