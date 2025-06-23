Library Foundations
================================================

Recall from :doc:`plan` that RomCom is an alphabetically ordered hierarchy.
This page describes the beginning of RomCom's alphabet.

The ``base`` package
----------------------

The functional foundation of RomCom is the :doc:`api/rc/base/index` package.
Names in the :doc:`api/rc/base/index` namespace are exposed throughout :doc:`api/rc/index` without (package or module) qualification.
Yes, wildcard imports have been used to expose the foundations.

The package consists of two modules.

``base.definitions``
^^^^^^^^^^^^^^^^^^^^^
Provides nothing but basic Protocols, constants and Type annotations.

``base.models``
^^^^^^^^^^^^^^^^^^^^
Provides BaseClasses for RomCom software objects.



Classes
---------------------

:doc:`api/rc/base/models/index` comprises the following classes, from which much of RomCom derives.

*Store*
^^^^^^^^^^^^

An abstract base class whose objects are endowed with filesystem storage in ``self.path``.
*Store* regards itself as SuperClass of all files and folders.

MetaData
^^^^^^^^^^^

Alias for ``dict[str, Any]``. All Meta content must be of this Type.

Meta
^^^^^^^^^

A concrete :term:`Store`, consisting of :term:`MetaData` stored in a ``.json`` file.
MetaData item ``'key'`` in any Meta ``object`` is accessed as ``object['key']``.

Matrix
^^^^^^^^^^^^

Alias for ``pd.DataFrame | Np.Matrix | Tc.Matrix``. All Table content must be of this Type.

Table
^^^^^^^^^^^^

A concrete *Store*, consisting of a ``pd.DataFrame`` stored in a ``.csv`` file.
The Matrix` held in any Table ``object`` is accessed in the desired format as the property ``object.pd``, ``object.np``, or ``object.tc``.
Although Table is concrete, the class constant ``Table.options`` governs ``.csv`` file options, which are frequently tailored by subclassing.
For example, ``DesignMatrix`` is a concrete subclass of Table designed to hold training data.

*DataBase*
^^^^^^^^^^^^

An abstract *Store*, containing ``NamedTables`` (a ``NamedTuple`` of Tables) with Meta.
The ``NamedTables`` of any *DataBase* ``object`` is the property ``object.namedTables``,
its Meta the property ``object.meta``.

Any concrete SubClass such as ``MyDataBase`` must define ``MyDataBase.NamedTables(NamedTuple)``
which indexes ``MyDataBase.defaults()`` by ``MyDataBase.names()``.

For example, ``MyDataBase`` may define ``MyDataBase.NamedTables(NamedTuple)`` as::

    NamedTables(NamedTuple)
        zero: Table | Matrix | MetaData = pd.DataFrame([0])
        one: Table | Matrix | MetaData = pd.DataFrame([1])

An ``object`` of type ``MyDataBase`` instantiated with ``path`` would appear on the filesystem as

Every model in RomCom is some Type of concrete *Database*.


CRUD (Create Read Update Delete) Protocols
----------------------------------------------

The `Lifecycle of Software Objects <https://en.wikipedia.org/wiki/The_Lifecycle_of_Software_Objects>`__ in RomCom
follows the conventional `CRUD <https://www.fluentpython.com/lingo/>`__ biography, told on the user's filesystem.
The ``base`` Classes take all responsibility for implementing CRUD in RomCom.

.. glossary::

    Create Protocol
        Every derived ``Class(Store)`` possesses a ``Class.create(path)`` classMethod
        returning an ``object`` of type ``Class`` created in ``path`` selectively (without affecting other files in ``path``).
        So any *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parents.
        Because all stored ``objects`` derive from *Store*, ``Store.create(path)`` will delete everything in its ``path``.

    Read Protocol
        Every ``object`` of derived ``Class(Store)`` is read from ``path`` by the constructor ``object = Class(path)``
        defined in ``Class.__init__(path)``.

    Update Protocol
        Every ``object`` of derived ``Class(Store)`` is updated in place and written in ``object.path`` by the
        function ``object(**kwargs)`` defined in ``__call__(self, **kwargs)``.
        SubClasses frequently override ``__call__(self,**kwargs)`` to perform some calibration or optimization before writing.

    Delete Protocol
        Every class derived from *Store* has a ``delete(path)`` classMethod
        which deletes a *Store* in ``path`` selectively  (leaving all other files intact).
        So any *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parent folders.
        Because all stored ``objects`` derive from *Store*, ``Store.delete(path)`` deletes everything in its ``path``.


Container Protocols
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


CRUD (Create Read Update Delete) Protocols
----------------------------------------------

The `Lifecycle of Software Objects <https://en.wikipedia.org/wiki/The_Lifecycle_of_Software_Objects>`__ in RomCom
follows the conventional `CRUD <https://www.fluentpython.com/lingo/>`__ biography, told on the user's filesystem.
Implementing CRUD is the primary responsibility of the ``base`` Classes.

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


