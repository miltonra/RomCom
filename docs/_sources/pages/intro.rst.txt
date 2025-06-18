
Intro
==========================================================================================================================================

This page introduces the structure and foundation of the RomCom library, providing notation and vocabulary used throughout the User Guide.
Familiarity with RomCom :doc:`../conventions` is assumed.

.. include:: intro/structure.rst


Foundation
-----------------

The functional foundation of RomCom is the :doc:`api/rc.base` package, the focus of the remainder of this intro.
The package consists of two modules

* ``base.definitions`` provides nothing but basic constants and type annotations.
* ``base.models`` provides base classes for RomCom software objects.

However, internal wildcard imports mean that the ``base`` namespace exposes all functionality without (module) qualification.
Furthermore, the ``base`` foundation is exposed using ``from rc.base import *`` in every other RomCom namespace.


Classes
-----------------------

``base.models`` comprises the following classes, from which much of RomCom is derives.

.. glossary::

    *Store*
        An abstract base class whose objects are endowed with filesystem storage in ``self.path``.

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
        For example, ``DesignMatrix`` is a concrete subclass of Table tailored to house training data.

    *DataBase*
        An abstract :term:`Store`, containing ``NamedTables`` (a ``NamedTuple`` of :term:`Table` s) with :term:`Meta`.
        Any concrete subclass such as ``MyDataBase`` must define ``MyDataBase.NamedTables(NamedTuple)``
        which indexes ``MyDataBase.defaults()`` by ``MyDataBase.names()``.
        Every model in RomCom is some Type of concrete *Database*.


CRUD (Create Read Update Delete) Protocol
----------------------------------------------

The `Lifecycle of Software Objects <https://en.wikipedia.org/wiki/The_Lifecycle_of_Software_Objects>`_ in RomCom
follows the conventional `CRUD <https://www.fluentpython.com/lingo/>`_ narrative, told on the user's filesystem.
Implementing CRUD is the primary responsibility of the base Classes.

.. glossary::

    Creating or Copying *Store* objects
        Every derived ``Class(Store)`` possesses ``Class.create(path)`` and ``Class.copy(object, path)`` class methods
        which create an ``object`` of type ``Class`` in ``path``, leaving other all other files intact.
        So any *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parents.

    Reading *Store* objects
        Every ``object`` of derived ``Class(Store)`` is read from ``path`` by the constructor ``object = Class(path)``
        defined in ``Class.__init__(path)``.

    Updating *Store* objects
        Every ``object`` of derived ``Class(Store)`` is updated in place and written in ``object.path`` by the
        function ``object(**kwargs)`` defined in ``Class.__call__(**kwargs)``.
        This is often overridden to perform some calibration or optimization before writing.

    Deleting *Store* objects
        Every class derived from *Store* has a ``delete(path)`` class method
        which deletes a *Store* in ``path``, leaving all other files intact.
        So any *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parent folders.


Container Protocol
----------------------

Items are retrieved from *Store* objects using the container protocol, which is implemented by the ``__getitem__`` method.
.. glossary::

    *Store*
        An abstract base class whose objects are endowed with filesystem storage in ``self.path``.

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
        For example, ``DesignMatrix`` is a concrete subclass of Table tailored to house training data.

    *DataBase*
        An abstract :term:`Store`, containing ``NamedTables`` (a ``NamedTuple`` of :term:`Table` s) with :term:`Meta`.
        Any concrete subclass such as ``MyDataBase`` must define ``MyDataBase.NamedTables(NamedTuple)``
        which indexes ``MyDataBase.defaults()`` by ``MyDataBase.names()``.
        Every model in RomCom is some Type of concrete *Database*.


