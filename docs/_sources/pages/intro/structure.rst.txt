
Structure
-----------------

The RomCom library is an alphabetically ordered functional hierarchy of namespaces, organised by package and module.

.. glossary::

    library
        Refers to the RomCom library imported as ``rc``.

    package
        The RomCom library is organised into alphabetically ordered packages ``rc.base``,
        ``rc.data``, ``rc.glr``, ``rc.gpr``, ``rc.gsa``, ``rc.rom``, ``rc.task``.
        Each package adds a layer of functionality depending only on alphabetically prior packages.
        For example, the ``base`` package is foundational, whereas the ``task`` package is the most user-friendly and efficient interface
        for performing common tasks.

    module
        Each package is organised into alphabetically ordered modules providing functionality.
        Each module adds a layer of functionality depending only on alphabetically prior modules.
        For example, the ``rc.task.scripts`` module is the gateway to common tasks, employing results summarising functionality provided by ``rc.task.results``.


