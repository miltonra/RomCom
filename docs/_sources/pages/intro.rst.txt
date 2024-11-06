
Intro
==========================================================================================================================================

This page is intended to introduce the building blocks of the RomCom Library, providing notation and vocabulary used throughout the User Guide.
The building blocks constitute Python package :doc:`api/rc.base`, which provides further detail, constants and static typing fundamental to RomCom.

Conventions
----------------

Throughout this User Guide, classes will be Capitalized, and *Italicized* when abstract (not instantiable).
Pandas, NumPy and PyTorch will be abbreviated as ``PD | pd``, ``NP | np``, and ``TC | tc`` respectively.
The RomCom library itself is contained in the ``rc`` package, and will be abbreviated as such.

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

Creating, Copying and Deleting *Store* Objects
------------------------------------------------

Every class derived from *Store* has ``create``, ``copy`` and ``delete`` class methods
which create or delete a *Store* in a folder, leaving other folder contents intact.
So any *DataBase* or *Model* may safely reside alongside other files (or folders) in the folder ``self.path``.

Reading *Store* Objects
------------------------------------------

Every ``object`` of class ``Object`` derived from `Store`_ is read from ``path`` by the constructor ``object = Object(path)``
defined in ``Object.__init__(path)``.

Updating *Store* Objects
------------------------------------------

Every ``object`` of class ``Object`` derived from `Store`_ is updated in place and stored in ``object.path`` by the function ``object(**kwargs)`` defined in
``Object.__call__(**kwargs)``.
This function usually embodies some optimization (or calibration), so that optimizing a concrete `DataBase`_ object amounts to
evaluating (calling) it as a function.
