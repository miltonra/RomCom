
Intro
==========================================================================================================================================

This page is intended to introduce the building blocks of the RomCom Library, providing notation and vocabulary used throughout the User Guide.
The building blocks constitute Python package :doc:`api/rc.base`, which provides further detail, constants and static typing fundamental to RomCom.

Conventions
^^^^^^^^^^^^

Throughout this User Guide, classes will be Capitalized, and *Italicized* when abstract (not instantiable).
Pandas, NumPy and PyTorch will be abbreviated as ``PD | pd``, ``NP | np``, and ``TC | tc`` respectively.
The RomCom library itself is contained in the ``rc`` package, and will be abbreviated as such.

Base Classes for CRUD
^^^^^^^^^^^^^^^^^^^^^^^^

The `Lifecycle of Software Objects <https://en.wikipedia.org/wiki/The_Lifecycle_of_Software_Objects>`_ in RomCom
follows the conventional `CRUD <https://www.fluentpython.com/lingo/>`_ narrative, told on the user's filesystem.
The detailed implementation of CRUD is relegated to the following base classes, from which most RomCom software objects derive.

*Store*
---------

An abstract base class whose objects are endowed with (filesystem) storage at ``self.path``.

MetaData
---------

Alias for ``dict[str, Any]``.

Meta
---------

A concrete `Store`_, consisting of `MetaData`_ stored in a ``.json`` file.
The metadata ``dict`` held in any Meta ``object`` is accessed as the property ``object.data``.

Matrix
---------

Alias for ``PD.DataFrame | NP.Matrix | TC.Matrix``.

Table
------------

A concrete `Store`_, consisting of a ``PD.DataFrame`` stored in a ``.csv`` file.
The `Matrix`_ held in any Table ``object`` is accessed in the desired format as the property ``object.pd``, ``object.np`` , or ``object.tc``.

*Tables*
------------

An abstract `Store`_, consisting of a ``NamedTuple`` of `Table`_'s stored in a folder.
Any concrete subclass such as ``MyTables`` must define ``MyTables.NT(NamedTuple)`` listing ``MyTables.defaults`` by ``MyTables.names``.

*DataBase*
---------

An abstract `Store`_, consisting of `Tables`_ plus `Meta`_, all stored in the `Tables`_ folder. Any concrete subclass such
as ``MyDataBase`` must define ``MyDataBase.Tables(Tables)``, as a concrete `Tables`_ class.
The majority of RomCom's work is performed by concrete *Database*'s.

CRUD Lifecycle Protocol
^^^^^^^^^^^^^^^^^^^^^^^^

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
