#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2027 Robert A. Milton
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

""" Abstract and concrete BaseClasses for RomCom models."""
import pylab as pl

from .definitions import *
from shutil import copyfile, copytree, rmtree
from json import load, dump


class Store(ABC):
    """ Base Class for any stored Class. Users are not expected to SubClass this Class directly."""

    ext: str = ''
    """Class attribute specifying the file extension terminating ``self.path``. 
    Override if and only if the derived Class must be stored in a file.
    Otherwise, ``cls.ext == ''`` and the derived Class is stored in a folder."""

    class NameP(Protocol):
        """``str(self) = str(self.path.name)`` and ``repr(self) = str(self.path)``. """

    class CreateP(CreateP):
        """ ``Store.create(path)`` destroys everything in its ``path``. """

    class ReadP(ReadP):
        """ ``Store.__init__(path)`` must be overridden. """

    class UpdateP(UpdateP):
        """ ``Store.__call__(**updates)`` must be overridden. """

    class DeleteP(DeleteP):
        """ ``Store.delete(path)`` destroys everything in its ``path``. """

    class CopyP(CopyP):
        """ ``Store.copy(src, dst)`` deletes everything in ``dst`` before copying everything in ``src``. """

    @property
    def path(self) -> Path:
        """ The Path to this Store, without ``cls.ext``.
        File extension is internal, meaning ``self._path = self.path + cls.ext``."""
        return self._path.with_suffix('') if self.ext else self._path

    def __repr__(self) -> str:
        """ The Path to this Store."""
        return str(self.path)

    def __str__(self) -> str:
        """ The name of this Store - i.e. ``self.path.name``."""
        return str(self.path.name)

    @abstractmethod
    def __call__(self, **updates: Any) -> Self:
        """ Update and store ``self``.

        Args:
            **updates: Updates to ``self``.

        Returns: ``self``.
        """
        raise NotImplementedError()

    @abstractmethod
    def __init__(self, path: PathLike, **kwargs: Any):
        """ Store ``path`` in ``self._path``.

        Overrides should call ``super(Store).__init__(path)`` as a matter of priority.

        Args:
            path: The Path to ``self``. Do not include an extension.
        """
        self._path = self.extAppend(self.mkdir(path))

    @classmethod
    def extAppend(cls, path: PathLike) -> Path:
        """ Append ``cls.ext`` to ``path.name``.

        Args:
            path: The path to append ``cls.ext`` to.

        Returns: ``Path(path)`` with ``cls.ext`` appended.
        """
        path = Path(path)
        if cls.ext:
            path = path.with_name(path.name + cls.ext)
        return path

    @classmethod
    def mkdir(cls, path: PathLike) -> Path:
        """ Create ``path.parent``, with a subfolder ``path`` if ``cls.ext == ''``.

        Args:
            path: The folder to create, or a child file of the folder to create.

        Returns: ``path``.
        """
        path = Path(path)
        if cls.ext:
            path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        else:
            path.mkdir(mode=0o777, parents=True, exist_ok=True)
        return path

    @classmethod
    @abstractmethod
    def create(cls, path: PathLike, **kwargs: Any) -> Path:
        """ Create a folder (and its parents) if it doesn't already exist.

        Overrides should create and return an instance of ``cls``.

        Args:
            path: Where to create the folder. If ``cls.ext != ''``, the parent folder of ``path`` is created.

        Returns:
            ``path`` with extension ``f'.{cls.ext}'``.

        Raises:
            FileExistsError: If attempting to overwrite a file with a folder.
        """
        return cls.mkdir(path)

    @classmethod
    @abstractmethod
    def copy(cls, src: PathLike, dst: PathLike) -> Path:
        """ Copy ``src`` to ``dst``, overwriting only files in common.

        Overrides should copy an instance of ``cls`` called ``src`` to ``Store.create(dst)``
        and return the copy.

        Args:
            src: The source Path, which must be a folder or a file.
            dst: The destination Path, which may or may not exist.
            
        Returns: ``dst``.
        
        Raises:
            FileNotFoundError: If ``src`` does not exist.
            FileExistsError: If attempting to overwrite a file with a folder.
        """
        src, dst = Path(src), Path(dst)
        if src.is_dir():
            copytree(src=src, dst=dst, dirs_exist_ok=True)
        else:
            copyfile(src, dst)
        return dst

    @classmethod
    def delete(cls, path: Path) -> Path:
        """ Delete any file or folder at ``path``.

        Args:
            path: The Path to delete.
            
        Returns: ``path``, which no longer exists.
        """
        path = cls.extAppend(path)
        if path.is_dir():
            rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=False)
        return path


class Meta(Store, dict):
    """ Concrete Class encapsulating metadata stored in a ``.json`` file.
    The place to store kwargs and options of all Types.
    """

    ext: str = '.json'  #: ext: Class attribute specifying the file extension of Meta instances.

    class IndexP(IndexP):
        """ ``self[key]``, ``len(self)`` are inherited from ``dict``, except that ``self[key]`` writes to file."""

    class EqualsP(EqualsP):
        """ ``self == other`` is inherited directly from ``dict``. """

    class CreateP(CreateP):
        pass

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        """ ``self(**updates)`` performs ``dict.update(**updates)`` (inherited), then writes to ``self.path``. """

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    def __call__(self, **updates: Any) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            **updates: Data to update ``self.data``.

        Returns: ``self``.
        """
        self.update(updates)
        with open(self._path, mode='w') as file:
            dump(self, file, indent=4)
        return self

    def __setitem__(self, key: str, value: Any):
        """ Act as a ``dict``, then write to ``.json``."""
        super().__setitem__(key, value)
        self()

    def __init__(self, path: PathLike, **data: Any):
        """ Construct ``self`` from a ``.json`` file or ``MetaData``.
        This is read *or* write, *never* both: ``path`` is read *only* if ``**data`` is absent.

        Args:
            path: The Path (file) to store ``self``. A ``.json`` extension is implicitly appended.
            **data: The ``MetaData`` to store. If absent, ``self`` is read from ``path``,
                otherwise ``self=dict(**data)`` is stored in ``path`` (which is overwritten if existing).
        """
        super(Meta, self).__init__(path)
        if data == {}:
            with open(self._path, mode='r') as file:
                data = load(file)
        super(Store, self).__init__(**data)
        self()

    @classmethod
    def create(cls, path: PathLike, **data: Any):
        """ Create a Meta at ``path``, overwriting.

        Args:
            path: The Path (file) to store ``self``, overwritten if existing.
                A ``.json`` extension is implicitly appended.
            **data: The ``MetaData`` to store.

        Returns: The Meta created.
        Raises: TypeError if ``data`` is empty.
        """
        if not data: raise ValueError("Meta cannot be created with no data")
        return cls(path, **data)

    @classmethod
    def copy(cls, src: Self, dst: PathLike) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source Meta.
            dst: The destination Path, overwritten if existing.
                A ``.json`` extension is implicitly appended.

        Returns: The Meta now stored at ``dst.json``.
        """
        return cls.create(dst, **src)


class Table(Store):
    """ Concrete Class encapsulating a DataFrame backed by a ``.csv`` file.

    This Class may be usefully overridden to provide bespoke read and write options for
    file operations. SubClasses should follow the template (copy and paste it)::

        class MyTable(Table):

            readOptions: MetaData = Table.readOptions | {'myOption': 'myValue'}
            \"\"\" File read options passed directly to
            `pl.read_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.read_csv.html#polars.read_csv>`__.\"\"\"

            writeOptions: MetaData = Table.writeOptions | {'myOption': 'myValue'}
            \"\"\" File write options passed directly to
            `pl.DataFrame.write_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.DataFrame.write_csv.html>`__.\"\"\"
    """

    ext: str = '.csv'   #: Class attribute specifying the file extension of Table objects. Defaults to ``.csv``.

    con: str = '│'
    """ Class attribute specifying the connector when collapsing column heads and categorical coords.
    Defaults to ``│``."""

    readOptions: MetaData = {}
    """ File read options passed directly to `pl.read_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.read_csv.html#polars.read_csv>`__."""

    writeOptions: MetaData = {'include_bom': True}
    """ File write options passed directly to `pl.DataFrame.write_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.DataFrame.write_csv.html>`__."""

    class IndexP(IndexP):
        """ ``self[columns]`` accesses Table columns by IndexLike ``index``. ``len(self)`` counts the columns."""

    class EqualsP(EqualsP):
        """ ``self == other`` compares ``self.pl``, ``self.np`` or ``self.tc`` matching the the type of ``other``. """

    class CreateP(CreateP):
        """ Creates a new Table at ``path`` from ``data: TableData | Table``. """

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        pass

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    @property
    def heads(self) -> list[str]:
        """ The column heads of ``self``, alias ``self.df.columns``."""
        return self._df.columns

    @heads.setter
    def heads(self, value: Iterable[str]):
        self._df.columns = list(value)
        self()

    @property
    def df(self) -> DataFrame:
        """ The DataFrame stored in ``self``."""
        return self._df

    @property
    def np(self) -> Np.Matrix:
        """ The ``Np.Matrix`` stored in ``self``."""
        return self._df.with_columns(pl.col(Category).cast(pl.Int32)).to_numpy()

    @property
    def tc(self) -> Tc.Matrix:
        """ The ``Tc.Matrix`` stored in ``self``."""
        return self._df.with_columns(pl.col(Category).cast(pl.Int32)).to_torch()

    def broadcast_to(self, target_shape: tuple[int, int], is_diagonal: bool = True) -> Self:
        """ Broadcast ``self``.

        Args:
            target_shape: The shape to broadcast to.
            is_diagonal: Whether to zero the off-diagonal elements of a square matrix.

        Returns: ``self``.

        Raises:
            IndexError: If broadcasting is impossible.
        """
        try:
            data = np.array(np.broadcast_to(self.np, target_shape))
        except ValueError:
            raise IndexError(f'{repr(self)} has shape {self._df.shape} '
                             f'which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            data = np.diag(np.diagonal(data))
        return self(data)

    def __len__(self) -> int:
        """ Counts the columns in ``self``. """
        return self._df.width

    def __getitem__(self, index: IndexLike) -> DataFrame:
        """ Indexer returns the column(s) named or sliced by ``index``. """
        return self._df[:, index]     # int or slice

    def __setitem__(self, index: IndexLike, columns: Table | TableData | tuple[Table | TableData, ...]):
        """ Indexer sets the Table (s) named or sliced by ``index``."""
        if isinstance(index, str):
            columns = self._df.with_columns(**{index : pl.lit(columns)})
        elif isinstance(index, int):
            columns = self._df.with_columns(**{self._df.columns[index] : pl.lit(columns)})
        elif isinstance(index, Iterable):
            if not (isinstance(columns, tuple) and len(columns) == len(index)):
                raise IndexError(f'Expected a tuple of {len(index)} tables, not {len(columns)}.')
            columns = self._df.with_columns(**{self.heads[i]: pl.lit(columns[i])
                                               for i in range(len(index))})
        elif isinstance(columns, tuple):
            return self.__setitem__(self.heads[index], columns)
        else:
            return NotImplemented
        self(columns)

    def __eq__(self, other: Self | TableData) -> bool:
        """ Equality of ``self`` and ``other``.

        Args:
            other: The other to compare with.

        Returns: Not implemented if ``other`` is not of admissible Type.
            Otherwise returns comparison with ``self.pl``, ``self.np`` or ``self.tc`` matching the type of ``other``.
        """
        match other:
            case Table():
                return self._df.equals(other._df)
            case DataFrame():
                return self._df.equals(other)
            case Np.Matrix():
                return np.array_equal(self.np, other)
            case Tc.Matrix():
                return tc.equal(self.tc, other)
            case _:
                return NotImplemented

    def __call__(self, update: Self | TableData | None = None) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            update: The data updates.

        Returns: ``self``.
        """
        if isinstance(update, Table):
            self._df = update._df
        elif isinstance(update, DataFrame):
            self._df = update
        else:
            self._df = DataFrame(update, orient='row')
        self._df = self._df.with_columns(pl.col(pl.Float16, pl.Float32, pl.Float64).cast(Float))
        self._df = self._df.with_columns(pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                                pl.Int128, pl.Boolean).cast(String))
        self._df = self._df.with_columns(pl.col(String).cast(Category))
        self._df.write_csv(self._path, **self.writeOptions)
        return self

    def __init__(self, path: PathLike, table: Self | DataFrame | None = None):
        """ Construct ``self`` from a ``.csv`` file or DataFrame.

        Args:
            path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
            table: The ``Table | DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                otherwise ``self`` is stored in ``path`` (which is overwritten if existing).
        """
        super().__init__(path)
        if table is None:
            self(pl.read_csv(self._path, **self.readOptions))
        else:
            self(table)

    @classmethod
    def create(cls, path: PathLike, data: Self | TableData, **kwargs: Any) -> Self:
        """ Create a Table at ``path``, overwriting.

        Args:
            path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            data: The table to store.
            **kwargs: KeywordArguments passed directly to `DataFrame()`_.

        Returns: The Table created.

        .. DataFrame(): https://docs.pola.rs/api/python/dev/reference/dataframe/index.html
        """
        data = DataFrame(data._df if isinstance(data, Table) else data, **kwargs)
        return cls(path, data)

    @classmethod
    def copy(cls, src: Self, dst: PathLike) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source Table.
            dst: The destination Path, overwritten if existing.
                A ``.csv`` extension is implicitly appended.

        Returns: The Table now stored at ``dst``.
        """
        Store.copy(src._path, cls.extAppend(dst))
        return cls(dst)

    @classmethod
    def conjoinHeads(cls, src: PathLike, dst: PathLike, headcount: int = 2) -> Self:
        """ Collapse multi-level headers in ``src`` to single-level ``dst.heads``,
        overwriting ``dst`` with a Table.
        Collapse is top down, so a 3-level header ``(a,b,c)`` becomes the single head ``a│b│c``.
        The first column is presumed to be an index. Any other column with empty header levels is dropped.

        Args:
            src: The source Path. A ``.csv`` extension is implicitly appended.
            dst: The destination ``Table.path``, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            headcount: Counts the column heads (header rows) in ``src.csv``.
        Returns: The Table now stored at ``dst``.
        """
        src = cls.extAppend(src)
        heads = pl.read_csv(src, **(cls.readOptions | {'has_header': False, 'n_rows': headcount})
                            ).fill_null(2*cls.con)
        src = pl.read_csv(src, **(cls.readOptions | {'has_header': False, 'skip_rows': headcount}))
        heads = [cls.con + cls.con.join(map(str, heads[head].to_list())) + cls.con for head in heads.columns]
        src = DataFrame(src, heads).drop([col for col in heads[1:] if 2*cls.con in col])
        src.columns = [head[1:-1] for head in src.columns]
        return cls.create(dst, src)

    @classmethod
    def unjoinHeads(cls, src: Table, dst: PathLike) -> Path:
        """ Explode ``src.heads`` into multi-level headed  ``dst.csv``, overwriting.
        Explosion is from the left, so ``a│b│c`` becomes the 3-level header ``(a,b,c)``.
        The first column is presumed to be an index.
        Every other column must produce the same headcount (number of levels).

        Args:
            src: The source Table.
            dst: The destination Path, overwritten if existing. A ``.csv`` extension is implicitly appended.
        Returns: ``dst``, now containing the unjoined ``dst.csv``.
        Raises: IndexError if ``src.heads`` cannot be unjoined due to inconsistent headcounts (levels).
        """
        heads = [head.split(cls.con) for head in src.heads]
        if len(heads) > 1:
            headcount = {len(head) for head in heads[1:]}
            if len(headcount) > 1:
                raise IndexError(f'Cannot unjoin heads: Expected all heads except the first '
                                 f'to have the same number of levels, but got {headcount}.')
            heads[0] += [None] * (headcount.pop() - len(heads[0]))
        heads = DataFrame(heads)
        with open(cls.extAppend(dst), "w", newline="", encoding="utf-8") as file:
            heads.write_csv(file, **(cls.writeOptions | {'include_header': False}))
            src.df.write_csv(file, **(cls.writeOptions | {'include_header': False}))
        return Path(dst)


class DataBase(Store):
    """ ``NamedTables(NamedTuple)`` in a folder alongside Meta. Abstract BaseClass for any model.

    *DataBase* SubClasses must be implemented according to the template (copy and paste it)::

        class MyDataBase(DataBase):                                                                                 [EDIT]

            class NamedTables(NamedTuple):

                names[i]: Table | TableData = defaults[names[i]]                                                    [EDIT]
                \"\"\" Normally a DataFrame. If no default is appropriate, use the TableType.\"\"\"                 [EDIT]
                ...

                def __call__(self, name: str) -> Table | TableData | MetaData:
                    \"\"\" Returns the Table named ``name``.\"\"\"
                    return getattr(self, name)

            Tables: NamedTables[type[Table], ...] = NamedTables(**{name: Table for name in NamedTables._fields})    [EDIT]
            \"\"\" The ``NamedTables`` of TableTypes, to communicate ``readOptions, writeOptions``.\"\"\"

            defaultMeta: MetaData = {'Tables': {name: TableType.__name__
                                                for name, TableType in Tables._asdict().items()}, }                 [EDIT]
            \"\"\" Class default ``self.meta``.\"\"\"
    """

    class NamedTables(NamedTuple):
        """ Must be overridden. """
        NotImplemented: Table | TableData = Table

        def __call__(self, name: str) -> Table | TableData | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)


    Tables: NamedTables[type[Table], ...] = NamedTables(**{name: Table for name in NamedTables._fields})
    """ Class attribute of the form ``NamedTables(**{names[i]: TableTypes[i], ...})``, 
    where ``TableTypes[i]`` is a SubClass of Table. Must be overridden."""

    defaultMeta: MetaData = {'Tables': {name: TableType.__name__ for name, TableType in Tables._asdict().items()}}
    """ Class attribute. Should be overridden."""

    class IndexP(IndexP):
        """ ``self[index]`` accesses ``NamedTables`` by IndexLike ``index``.  ``len(self)`` counts the Tables."""

    class EqualsP(EqualsP):
        """ ``self == other`` compares Meta and ``tables`` between two DataBases. """

    class CreateP(CreateP):
        pass

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        """ ``self(**tables)`` updates and writes ``NamedTables`` (``self.meta(**updates)`` updates Meta."""

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    @property
    def tables(self) -> NamedTables:
        """ The ``NamedTables`` currently in ``self``."""
        return self._tables

    @property
    def meta(self) -> Meta:
        """ The Meta currently in ``self``."""
        return self._meta

    def __eq__(self, other: Any) -> bool:
        """ Equality of meta, namedTables and names()."""
        if isinstance(other, DataBase):
            return (self.names() == other.names() and self.meta == other.meta
                    and self._tables == other.tables)
        return NotImplemented

    def __len__(self) -> int:
        """ Counts the Tables in ``self``. """
        return len(self._tables)

    def __getitem__(self, index: IndexLike) -> Table | tuple[Table, ...]:
        """ Indexer returns the Table (s) named or sliced by ``index``. """
        if isinstance(index, str):
            return self._tables(index)
        elif isinstance(index, Iterable):
            return tuple(self[i] for i in index)
        else:
            return self._tables[index]     # int or slice

    def __setitem__(self, index: IndexLike, tables: Table | TableData | tuple[Table | TableData, ...]):
        """ Indexer sets the Table (s) named or sliced by ``index``."""
        if isinstance(index, str):
            tables = {index: tables}
        elif isinstance(index, int):
            tables = {index: tables}
        elif isinstance(index, Iterable):
            if not (isinstance(tables, tuple) and len(tables) == len(index)):
                raise IndexError(f'Expected a tuple of {len(index)} tables, not {len(tables)}.')
            index = tuple((self.index()[i] if isinstance(i, int) else i for i in index))
            tables = {index[i]: tables[i] for i in range(len(index))}
        elif isinstance(tables, tuple):
            return self.__setitem__(self.names()[index], tables)
        else:
            return NotImplemented
        self(**tables)

    def __call__(self, **tables: Table | TableData) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            path: Optionally, an update to ``self.path``, overwritten if existing.
            **tables: Updates to ``self`` in the form ``names[i]=Table[i], ...``.

        Returns: ``self``.
        """
        for name, table in tables.items():
            self._tables(name)(table)
        return self

    def __init__(self, path: PathLike, **tables: Table | DataFrame):
        """ Read the *DataBase* in ``path``.
        Reading is lazy: If ``names[i]`` occurs in ``**tables`` it's Table is not read, just updated.
        Overrides must call ``super(DataBase).__init__(path, **tables)`` as a matter of priority.

        Args:
            path: The Path to read from.
            **tables: Tables to update those read, in the form ``names[i]=tables[i], ...``.

        Raises:
            FileNotFoundError: If ``path`` lacks ``self.meta`` or any member of
                ``self.Tables.names`` not mentioned in ``**tables``.
        """
        super().__init__(path)
        try:
            self._meta = Meta(self._meta_in(path))
            self._tables = self.NamedTables(**{name:
                                                    TableType.create(self._path / name, tables[name])
                                                    if name in tables and tables[name] is not None
                                                    else TableType(self._path / name)
                                               for name, TableType in self.Tables._asdict().items()})
        except FileNotFoundError as error:
            print(f'DataBase "{self}" is trying to read a non-existent Table. Did your script mean to call '
                  f'{type(self).__qualname__}.create("{str(self)}") '
                  f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error


    @classmethod    # Class Property
    def names(cls) -> tuple[str, ...]:
        """ ``(names[i], ...)`` of table names for this ``Tables`` Class."""
        return cls.NamedTables._fields

    @classmethod    # Class Property
    def defaults(cls) -> dict[str, DataFrame]:
        """ ``{names[i]: DataFrame[i], ...}`` of default tables for this ``Tables`` Class."""
        return cls.NamedTables._field_defaults

    @classmethod
    def create(cls, path: PathLike, **tables_and_meta: Table | DataFrame | MetaData) -> Self:
        """ Create a *DataBase* in ``path``.

        Args:
            path: The folder to store the *DataBase* in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **tables_and_meta: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

        Returns: The *DataBase* created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMeta | (tables_and_meta.pop('meta', {}))))
        return cls(path, **(cls.defaults() | tables_and_meta))

    @classmethod
    def copy(cls, src: Self, dst: PathLike) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source *DataBase*.
            dst: The destination Path, which may or may not exist.

        Returns: The *DataBase* now stored in ``dst``.
        """
        return cls.create(dst, meta=src.meta, **src._tables._asdict())

    @classmethod
    def delete(cls, path: PathLike, ignoreErrors: bool=False) -> Path:
        """ Delete all *DataBase* files in ``path``, retaining ``path`` and any other files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: Path to the *DataBase* to delete.
            ignoreErrors: Whether to raise any ``FileNotFoundError`` s encountered.
        Returns: ``path``, which still exists.
        Raises: FileNotFoundError if ``path`` is not a folder, regardless of ``ignoreErrors``.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f'Path {path} is not a folder.')
        try:
            Meta.delete(cls._meta_in(path))
            for name in cls.names():
                Table.delete(path / name)
            if not tuple(path.iterdir()):
                path.rmdir()
        except FileNotFoundError as error:
            if not ignoreErrors:
                raise error
        return path

    @staticmethod
    def _meta_in(path: PathLike) -> Path:
        return Path(path) / 'meta'
