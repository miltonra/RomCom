
Intro
==========================================================================================================================================

This page is intended to introduce the building blocks of the RomCom Library, providing notation and vocabulary used throughout the User Guide.
The building blocks constitute Python package :doc:`api/rc.base`, which provides further detail, constants and static typing fundamental to RomCom.

Conventions
----------------

Throughout this User Guide, classes will be Capitalized, and *Italicized* when abstract (not instantiable).
Pandas, NumPy and TensorFlow will be abbreviated as ``PD | pd``, ``NP | np``, and ``TF | tf`` respectively.
The RomCom library itself is contained in the ``rc`` package, and will be abbreviated as such.

*Store*
---------

An abstract base class whose objects are endowed with (filesystem) storage at ``self.path``.

Meta
---------

A concrete *Store*, consisting of a Python ``dict[str, Any]`` stored in a `.json` file.
The metadata ``dict`` held in any Meta ``object`` is accessed as the property ``object.data``.

DataTable
------------

A concrete *Store*, consisting of a ``pd.DataFrame`` stored in a `.csv` file.
The data held in any DataTable ``object`` is accessed in the desired format as the property ``object.pd``, ``object.np`` , or ``object.tf``.

*DataBase*
------------

An abstract *Store*, consisting of a ``NamedTuple`` of DataTables stored in a folder.
Any concrete subclass such as ``MyDataBase`` must define ``MyDataBase.Tables(NamedTuple)`` listing the ``DataBase.table_names`` and
``DataBase.table_defaults`` for ``MyDataBase``.

*Model*
---------

An abstract *Store*, consisting of a *DataBase* plus Meta, all stored in the *DataBase* folder. Any concrete subclass such
as ``MyModel`` must define ``MyModel.DataBase(DataBase)``, as a concrete *DataBase*.
The majority of RomCom's work is performed by concrete *Models*.

Creating, Copying and Deleting *Store* Objects
------------------------------------------------

Every class derived from *Store* has ``create``, ``copy`` and ``delete`` class methods
which create or delete a *Store* in a folder, leaving other folder contents intact.
So any *DataBase* or *Model* may safely reside alongside other files (or folders) in the folder ``self.path``.

Reading *Store* Objects
------------------------------------------

Every ``object`` of class ``Object`` derived from *Store* is read from ``path`` by the constructor ``object = Object(path)``
defined in ``Object.__init__(path)``.

Updating *Store* Objects
------------------------------------------

Every ``object`` of class ``Object`` derived from *Store* is updated in place and stored in ``object.path`` by the function ``object(**kwargs)`` defined in
``Object.__call__(**kwargs)``.
This function usually embodies some optimization (or calibration), so that optimizing a ``Model`` object amounts to evaluating (calling) it as a function.
