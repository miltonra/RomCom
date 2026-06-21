
Glossary
------------------

.. rubric:: The User Guide follows Python convention by using | to denote logical OR.

RomCom is, for better or worse, laced with `Python lingo <https://www.fluentpython.com/lingo/>`_.

.. glossary::
    :sorted:

    *Protocol*
    CRUD Protocols
    NameP
    IndexP
    EqualsP
        Documentation in code, detailing the interface of a RomCom software object.
        See :doc:`api/rc/base/index` :ref:`Protocols <baseProtocols>`.

    dunder methods
        Special methods (also known as magic methods) whose names are surrounded by double underscores (dunders).
        The best known example is Python's Class constructor ``__init__(self, **kwargs)``.

    CRUD
        Create, Read, Update, Delete. The conventional lifecycle for software objects stored on a filesystem.

    folder
        directory.

    RMSE
        Root Mean Square Error.

    SD
        Standard Deviation.

    ecosystem
    ``pl``
    ``df``
    ``np``
    ``tc``
        See :ref:`ecosystem <ecosystem>`.

    *Store*
    Meta
    Table
    *DataBase*
        BaseClasses for all RomCom software objects. See :doc:`api/rc/base/index` :ref:`models <baseModels>`.

    ReturnType
    ArgumentType
    Path
    PathLike
    IndexP.Index
    IndexLike
    DataFrame
    Category
    ``Np``
    ``Tc``
    MetaData
    TableData
        Types supported by RomCom's :doc:`api/rc/base/index` :ref:`baseModels`.
        See :doc:`api/rc/base/index` :ref:`baseTypes`.

    categorical input
    discrete input
        An input variable whose value is one of a finite set of discrete values of Type ``int | str | bool``.
        RomCom Types any such data as :term:`Category`.

    model
        An imprecise term for a Class which furnishes core functionality.

    kwargs
        Keyword arguments, an object of Type :term:`MetaData`.

    NaN
        Not a Number. A special floating-point value which is not equal to itself.

    null
        Represents the absence of a value. ``None`` in Python.

    decimal fraction
        Any number with digits, possibly zero, after the decimal point. ``2.0`` is a decimal fraction, but ``2.`` is not.

    percentage
        Any ``str`` whose rightmost character is ``%``.

    Schema
        A `NamedTuple <https://typing.python.org/en/latest/spec/namedtuples.html>`_ enumerating the Tables housed by a :term:`*DataBase*` Type.

    schema
        A specification of the structure of a :term:`Table` (as a ``dict[str, Type]``),
        or a :term:`DataBase` (as a Schema).

    experiment
        The data you wish RomCom to analyse. RomCom sees any experiment as a :term:`*Design*`

    *Design*
        A Type of :term:`Table` with a validated :term:`schema` designed to represent experiments.
        See :ref:`dataDesigns`.

    axis
        A column in a :term:`experiment` or :term:`*Design*`.

    axisType
        A label which identifies the role of an axis in an :term:`experiment`. See :ref:`dataAxes`.