base
================================================
Recall from :doc:`plan` that RomCom is an alphabetically ordered hierarchy.
The functional foundation of RomCom is the  beginning of its alphabet -- the :doc:`api/rc/base/index` package.

These foundations are so pervasive that wildcard imports expose them internally.
In other words, the :doc:`api/rc/index` namespace includes all names in :doc:`api/rc/base/index` modules without
any (package or module) qualification.

The package consists of two modules, ordered alphabetically according to functional dependency.
These will be described in reverse order, because :doc:`api/rc/base/definitions/index` is in practice both
motivated and exemplified by :doc:`api/rc/base/models/index`.
So the latter makes sense of the former, contrary to their functional dependence.


.. _baseModels:
models
---------------------
All RomCom software objects derive from BaseClasses contained in the :doc:`api/rc/base/models/index` module.
They consist of *DataBases* made of Tables and Meta.
These components are **always** perfectly synchronized with the filesystem.

*Store*
^^^^^^^^^^^^
An abstract BaseClass whose objects are endowed with filesystem storage in ``self.path``.

Every Class derived from *Store* has a (constant) classAttribute ``Store.ext`` specifying its file extension.
This is implicitly appended to any path sent or accepted by the *Store*.

*Store* regards itself as the *SuperClass* of all files and folders.

.. _baseMeta:
Meta
^^^^^^^^^
A concrete *Store*, consisting of :ref:`baseMetaData` stored in a ``.json`` file.

.. _baseTable:
Table
^^^^^^^^^^^^
A concrete *Store*, consisting of a :ref:`baseDataFrame` stored in a ``.csv`` file.

Tables are copied from ``.csv`` or created from :ref:`baseTableData`, and may be updated from :ref:`baseTableData`.
These operations are governed by CRUD :ref:`baseProtocols` described below.

The :ref:`baseTableData` held in any Table ``object`` is accessed in the desired :ref:`ecosystem <ecosystem>`
format as the property ``object.df``, ``object.np``, or ``object.tc``.

Every Table begins with a single row of column ``Table.heads``.
Heads may join levels of categorization using the conjunction ``Table.con = │``.
This strange character is the utf8 Box Drawing Light Vertical (U+2502), which is unlikely to occur in user data.
Multi-row heads must be conjoined using ``Table.conjoinHeads(src, dst, headcount)`` upon entry.

The Table body below ``Table.heads`` contains data of just three Types: ``Category``, ``float`` and ``null``.
Any data of Type ``int | str | bool`` in a Table body is cast to :ref:`baseCategory` when first encountered.
All decimal fractions and ``NaN`` (Not a Number) are of Type ``float``.
Any missing entries are of Type ``null``.

RomCom is replete with SubClasses of ``Table`` fulfilling specific needs.
For example, ``DesignMatrix`` is a concrete subclass of Table designed to hold training data.

The classAttributes ``Table.readOptions`` and ``Table.writeOptions`` govern ``.csv`` file options.
These constants should only be tailored by SubClassing.

*DataBase*
^^^^^^^^^^^^
An abstract *Store*, containing NamedTables
(a `NamedTuple <https://typing.python.org/en/latest/spec/namedtuples.html>`_ of Tables) with Meta.

The NamedTables of any *DataBase* ``object`` is the property ``object.namedTables``,
its Meta the property ``object.meta``.

Any concrete SubClass such as ``MyDataBase`` must define ``MyDataBase.NamedTables(NamedTuple)``
which indexes ``MyDataBase.defaults()`` by ``MyDataBase.names()``.

For example, ``MyDataBase`` may define ``MyDataBase.NamedTables(NamedTuple)`` as::

    class MyDataBase(DataBase)
        NamedTables(NamedTuple)
            zero: Table | TableData | type[Table] = DataFrame([0])
            one: Table | TableData | type[Table] = DataFrame([1])

        Tables: NamedTables[type[Table], ...] = NamedTables(zero=Table, one=Table)

The last line is required to tell MyDataBase what TableTypes to expect.
In this way, ``Tables`` encapsulates ``Table.readOptions``, ``Table.writeOptions``, and possibly other functionality.
Reflecting its programmatic content, an ``object`` of type ``MyDataBase`` instantiated with ``path`` would appear on the filesystem as

.. image:: resources/DataBase.1.png
    :scale: 60%

|
Most models in RomCom are some Type of concrete *Database*.


definitions
----------------------
RomCom's Types and *Protocols* are contained in the :doc:`api/rc/base/definitions/index` module.

.. _baseTypes:
Types
^^^^^^^^^^^^^^^^^
RomCom Types abide by the principle

    Be conservative in what you send, be liberal in what you accept.
        [`Postel's Law <https://en.wikipedia.org/wiki/Robustness_principle>`__]

Rom Com methods send ReturnTypes and accept ArgumentTypes.

Path
+++++++++++++++++
Files and folders are of ReturnType Path (alias ``Pathlib.Path``) and ArgumentType PathLike (alias ``str | Path``).

.. _baseIndex:
Index
+++++++++++++++++
Elements are Indexed by ReturnType :ref:`IndexP.Index <baseIndex>` (alias ``Tuple[int]``) and
ArgumentType IndexLike (alias ``str | int | Iterable[str | int] | slice``.

.. _baseDataFrame:
DataFrame
+++++++++++++++++
A DataFrame is a 2D tabular data structure with labeled axes (rows and columns) and heterogeneous data types
(alias `pl.DataFrame <https://docs.pola.rs/api/python/stable/reference/dataframe/index.html>`__).

.. _baseCategory:
Category
+++++++++++++++++
Any data of Type ``int | str | bool`` in a :ref:`baseDataFrame` is cast to :ref:`baseCategory`
(alias `pl.Categorical <https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/#data-type-categorical>`__).

Np and Tc Tensors
++++++++++++++++++++
*Np* and *Tc* are *AbstractClasses* extending Types to NumPy (``np``) and PyTorch (``tc``),
such as ``Np.Tensor=np.ndarray`` , ``Tc.Tensor`` , ``Np.Matrix`` , ``Tc.Matrix`` , ``Np.Vector`` , ``Tc.Vector``.

.. _baseMetaData:
MetaData
+++++++++++++++++
:ref:`baseMeta` is sent and accepted as MetaData (alias ``Mapping[str, Any]``).

.. _baseTableData:
TableData
+++++++++++++++++
:ref:`Tables <baseTable>` are sent and accepted as TableData (alias ``DataFrame | Np.Matrix | Tc.Matrix``).

.. _baseProtocols:
*Protocols*
^^^^^^^^^^^^^^^^^
The `Lifecycle of Software Objects <https://en.wikipedia.org/wiki/The_Lifecycle_of_Software_Objects>`__ in RomCom
follows the conventional :term:`CRUD` biography, told on the user's filesystem.
Software objects in RomCom are also named, and their content indexed by element and compared with other objects.
The :doc:`api/rc/base/index` :ref:`baseModels` are responsible for implementing these *Protocols* in RomCom, as follows.


*NameP*
+++++++++++++++++
The *NameP Protocol* provides ``str(object) = str(object.path.name)`` and ``repr(object) = str(object.path)``
for any ``object`` derived from *Store*.

*IndexP*
+++++++++++++++++
Items are retrieved from, and replaced in, *Store* objects using the *IndexP Protocol*.
This enables expressions such as::

    values = object[index]  # Retrieves item(s) from object
    object[index] = values  # Replaces item(s) in object
    len(object)             # Counts the items accessible by index
    for index in object:
        if index in object: print('is always True')

where ``index`` must be of ArgumentType :ref:`IndexLike <baseIndex>`.
When ``index`` is Iterable, ``values`` must be a ``tuple[Table | TableData, ...]`` of corresponding ``len``.
The ArgumentType of ``index`` is IndexLike

*IndexP* is conceptually identical to a Python `sequence <https://docs.python.org/3/glossary.html#term-sequence>`__.

The specifics for each of the :doc:`api/rc/base/index` :ref:`baseModels` are as follows.

Meta
%%%%%%%
*IndexP* refers to ``dict`` items, but writes updates to disk immediately.

Table
%%%%%%%
*IndexP* refers to ``table.heads``, so ``len(table)`` is its width (complementing ``len(table.pl)`` its height).

*DataBase*
%%%%%%%%%%%%%
*IndexP* refers to ``database.namedTables``.

*EqualsP*
+++++++++++++++++
The *EqualsP Protocol* enables::

    objectA == objectB

which exhaustively compares all content except for ``path``.
If two objects of the same Type share the same location, they are surely identical for the filesystem is always synchronized with memory.
So the only interesting comparison is between objects whose ``path`` differs.

This is straightforward, but we shall highlight some pedantry

*DataBase*
%%%%%%%%%%%%%
Two *DataBase* objects are equal if and only if their ``.meta`` is equal, their ``.namedTables`` are equal,
and their ``.names()`` are equal. This is **not** the same as::

    dataBaseA.meta == dataBaseB.meta and dataBaseA.namedTables == dataBaseB.namedTables

because `NamedTuple <https://typing.python.org/en/latest/spec/namedtuples.html>`_ equality does not compare field names.

Two equal *DataBase* objects need not be of the same Type, nor share ``dataBaseA.NamedTables is dataBaseB.NamedTables``.
They are equal if and only if their **contents** are equal, regardless of their own Types.

*CreateP*
+++++++++++++++++
Every derived ``Class(Store)`` possesses a ``Class.create(path)`` classMethod
returning an ``object`` of type ``Class`` created in ``path`` selectively (without affecting other files in ``path``).
So any *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parents.
Because *Store* considers itself *SuperClass* to all files and folders,
``Store.create(path)`` deletes everything in its ``path`` before (re-)creating it.

*ReadP*
+++++++++++++++++
Every ``object`` of derived ``Class(Store)`` is read from ``path`` by its constructor ``object = Class(path)``
defined in ``Class.__init__(path)``.

*UpdateP*
+++++++++++++++++
Every ``object`` of derived ``Class(Store)`` is updated in place and written in ``object.path`` by the
function ``object(**kwargs)`` defined in ``__call__(self, **kwargs)``.
SubClasses frequently override ``__call__(self,**kwargs)`` to perform some calibration or optimization before writing.

*DeleteP*
+++++++++++++++++
Every class derived from *Store* has a ``delete(path)`` classMethod
which deletes a *Store* in ``path`` selectively  (leaving all other files intact).
So any *DataBase* may safely reside alongside other files (or folders) in ``path`` and its parent folders.
Because *Store* considers itself *SuperClass* to all files and folders,
``Store.create(path)`` deletes everything in its ``path`` before (re-)creating it.

