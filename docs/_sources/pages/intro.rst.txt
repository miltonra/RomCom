
Intro
==========================================================================================================================================

This page is intended to introduce the building blocks of the RomCom Library, providing notation and vocabulary used throughout the User Guide.
The building blocks constitute Python package :doc:`api/rc.base`, providing functionality, constants and type annotations fundamental to RomCom.

The building blocks are available unqualified in every RomCom namespace: they all wildcard import ``rc.base``.

Throughout RomCom, source code dependency is bottom-up alphabetically.
Alphabetical ordering applies to submodules too.
Two submodules comprise ``base``: ``base.models`` depends on ``base.definitions``, which provides nothing but constants and type annotations.


.. include:: intro/conventions.rst

Base Classes for CRUD
-----------------------

The `Lifecycle of Software Objects <https://en.wikipedia.org/wiki/The_Lifecycle_of_Software_Objects>`_ in RomCom
follows the conventional `CRUD <https://www.fluentpython.com/lingo/>`_ narrative, told on the user's filesystem.
The detailed implementation of CRUD is relegated to the following base classes, from which most RomCom software objects derive.

.. glossary::

    *Store*
        An abstract base class whose objects are endowed with (filesystem) storage at ``self.path``.

    MetaData
        Alias for ``dict[str, Any]``.

    Meta
        A concrete :term:`Store`, consisting of :term:`MetaData` stored in a ``.json`` file.
        MetaData item ``'key'`` in any Meta ``object`` is accessed as ``object['key']``.

    Matrix
        Alias for ``pd.DataFrame | Np.Matrix | Tc.Matrix``.

    Table
        A concrete :term:`Store`, consisting of a ``pd.DataFrame`` stored in a ``.csv`` file.
        The :term:`Matrix` held in any Table ``object`` is accessed in the desired format as the property ``object.pd``, ``object.np``, or ``object.tc``.
        Although Table is concrete, the class constant Table.options governs ``.csv`` file options, which are frequently tailored by subclassing.

    *Tables*
        An abstract :term:`Store`, consisting of a ``NamedTuple`` of :term:`Table` s stored in a folder.
        Any concrete subclass such as ``MyTables`` must define ``MyTables.NT(NamedTuple)`` listing ``MyTables.defaults`` by ``MyTables.names``.

    *DataBase*
        An abstract :term:`Store`, consisting of :term:`Tables` plus :term:`Meta`, all stored in the :term:`Tables` folder.
        Any concrete subclass such as ``MyDataBase`` must define ``MyDataBase.Tables(Tables)``, as a concrete :term:`Tables` class.
        The majority of RomCom's work is performed by concrete *Database* s.


CRUD Lifecycle Protocol
-----------------------------

Most RomCom software objects undergo the following major lifecycle events.

Creating and/or Copying
------------------------

Every derived ``Class(Store)`` possesses ``Class.create(path)`` and ``Class.copy(object, path)`` class methods
which create a ``Class`` in ``path``, leaving other all other files intact.
So any *Tables* or *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parent folders.

Reading *Store* Objects
------------------------------------------

Every ``object`` of derived ``Class(Store)`` read from ``path`` by the constructor ``object = Class(path)``
defined in ``Class.__init__(path)``.

Updating *Store* Objects
------------------------------------------

Every ``object`` of derived ``Class(Store)`` is updated in place and stored in ``object.path`` by the function ``object(**kwargs)`` defined in
``Class.__call__(**kwargs)``.
This function usually embodies some optimization (or calibration), so that optimizing a concrete `DataBase`_ object amounts to
evaluating (calling) it as a function.

Deleting *Store* Objects
------------------------------------------------

Every class derived from *Store* has a ``delete(path)`` class method
which deletes a *Store* in ``path``, leaving all other files intact.
So any *Tables* or *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parent folders.
