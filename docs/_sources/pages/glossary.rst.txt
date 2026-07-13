
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
        BaseClasses for all RomCom software objects. See :ref:`base models <baseModels>`.

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
        Types supported by RomCom's :ref:`base models<baseModels>`.
        See :ref:`base Types<baseTypes>`.

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
        Represents an absent value of any Type. ``None`` in Python.

    decimal fraction
    float literal
        A string ending with ``%`` or a number containing a decimal point ``.`` or an exponent ``e`` or  ``E``.

    Schema
        A `NamedTuple <https://typing.python.org/en/latest/spec/namedtuples.html>`_ enumerating the Tables housed by a :term:`DataBase` Type.

    schema
        A specification of the structure of a :term:`Table` (as a ``dict[str, Type]``),
        or a :term:`DataBase` (as a Schema).

    experiment
        The ``.csv`` you wish RomCom to analyse. RomCom sees any experiment as a :term:`Design`.
        See :ref:`dataExperiments`.

    Design
        A Type of :term:`Table` with a validated :term:`schema` designed to represent experiments.
        See :ref:`dataDesigns`.

    head
        A row of ``str`` column headings.

    axis
        A column in an :term:`experiment` or :term:`Design`. Each axis has  a ``str`` :term:`head`.

    axisType
        A ``str`` which identifies the role of an axis in an :term:`experiment`. See :ref:`dataAxes`.

    con
    conjoin
    unjoin
        The con character conjoins levels in a single :term:`head`, or conjoins Category axes a single Category.
        It is defined as ``Table.con = │``,
        the utf8 Box Drawing Light Vertical (U+2502) unlikely to occur in user data.
        Usually pronounced \"given\" (at least in the context of \"conditional\" probability), con means

            word-forming element meaning \"together, with\"
                [`Online Etymology Dictionary <https://www.etymonline.com/word/con->`__]

        from which we derive \"conjunction\", and \"contingent\" (on a condition).

    pivoted
    unpivoted
        When a Table is pivoted on an :term:`axis`, the unique values in that axis become :term:`head` for a set of new axes which replace the original one.
        The reverse operation, unpivoting replaces a set of axes with one :term:`Category` axis composed of the set :term:`head` and one (:term:`Category` or :term:`Float`) axis composed of the group body.
        Pivoted tables are short and fat, unpivoted ones tall and thin.
