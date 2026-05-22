
Glossary
------------------

.. rubric:: The User Guide follows Python convention by using | to denote logical OR.

RomCom is, for better or worse, laced with `Python lingo <https://www.fluentpython.com/lingo/>`_.

.. glossary::
    :sorted:

    *Protocol*
        Documentation in code, detailing the interface a Class supports via undocumented :term:`dunder methods`.
        They are named in *UpperCamelCaseP*, which is UpperCamelCase with a big P at the end which stands for *Protocol* and avoids tedious naming conflicts.

    dunder methods
        Special methods (also known as magic methods) whose names are surrounded by double underscores (dunders).
        The best know example is Python's Class constructor ``__init__(self, **kwargs)``.

    CRUD
        Create, Read, Update, Delete. The usual lifecycle for software objects stored on a filesystem.

    folder
        directory.

    RMSE
        Root Mean Square Error.

    SD
        Standard Deviation.

    ``pl``
    ``Pl``
        The polars package ``polars``.

    pl.DataFrame
        A sophisticated and flexible ``.csv`` viewer, used to store and manipulate Tables in RomCom.
        For a full description, see the `polars documentation <https://docs.pola.rs/user-guide/concepts/data-types-and-structures/#dataframe>`_.

    ``np``
    ``Np``
        The NumPy package ``numpy``.

    ``tc``
    ``Tc``
        The PyTorch package ``torch``.

    MetaData
        ``Mapping[str, Any]``.

    Table
        Consult :doc:`base <base>`.

    Categorical Input
    Discrete Input
        An input variable whose value is one of a finite set of discrete values of Type ``int | str | bool``,
        which will be interpreted as ``str`` for generality

    model
        An imprecise term for a Class which furnishes core functionality.
