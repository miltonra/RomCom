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

    @cached_property
    def path(self) -> Path:
        """ The Path to this Store, without ``cls.ext``.
        File extension is internal, meaning ``self._path = self.path + cls.ext``."""
        return self._path.with_name(self._path.name[:-len(self.ext)]) if self.ext else self._path

    def __repr__(self) -> str:
        """ The Path to this Store."""
        return str(self.path)

    def __str__(self) -> str:
        """ The name of this Store - i.e. ``self.path.name``."""
        return str(self.path.name)

    def __format__(self, format_spec: str) -> str:
        """ ``str(self) if format_spec == '!s' else repr(self)`` ."""
        return str(self) if format_spec == '!s' else repr(self)

    @abstractmethod
    def __call__(self, **updates: Any) -> Self:
        """ Update and store ``self``.

        Args:
            **updates: Updates to ``self``.

        Returns: ``self``.
        """
        raise NotImplementedError(f'__call__(self, **updates: Any) -> Self for {type(self)}')

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
            path.unlink(missing_ok=True)
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
        Raises: AssertionError if ``data`` is empty.
        """
        assert data, f'Meta for {path} cannot be created with no data'
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

            readOptions: MetaData = Table.readOptions | {'myOption': myValue}

            writeOptions: MetaData = Table.writeOptions | {'myOption': myValue}
    """

    ext: str = '.csv'   #: Class attribute specifying the file extension of Table objects. Defaults to ``.csv``.

    con: str = '│'
    """ Class attribute specifying the connector when collapsing multi-heads and Categories.
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
        """ Creates a new Table at ``path`` from ``tableData: TableData | Table``. """

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        pass

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    @property
    def schema(self) -> dict[str, type[Category | Float]]:
        """ The schema of ``self``, alias ``self.df.schema``."""
        return self._df.schema

    @property
    def head(self) -> list[str]:
        """ The head of ``self``, alias ``self.df.columns``."""
        return self._df.columns

    @head.setter
    def head(self, value: Iterable[str]):
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

        Raises: IndexError if broadcasting is impossible.
        """
        try:
            tableData = np.array(np.broadcast_to(self.np, target_shape))
        except ValueError:
            raise IndexError(f'{self} has shape {self._df.shape} '
                             f'which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            tableData = np.diag(np.diagonal(tableData))
        return self(tableData)

    def __len__(self) -> int:
        """ Counts the columns in ``self``. """
        return self._df.width

    def __getitem__(self, index: IndexLike) -> DataFrame:
        """ Indexer returns the column(s) named or sliced by ``index``. """
        if isinstance(index, str):
            index = [_head for _head in self.head if _head.startswith(index)]
        return self._df[:, index]     # int or slice

    def __setitem__(self, index: IndexLike, columns: Np.Matrix | Tc.Matrix | tuple[Np.Vector | Tc.Vector, ...]):
        """ Indexer sets the Table (s) named or sliced by ``index``."""
        match index:
            case str():
                index = [_head for _head in self.head if _head.startswith(index)]
                assert len(index) > 0, f'{self} has no column head starting with {index}.'
                if len(index) == 1:
                    columns = self._df.with_columns(**{index[0]: pl.lit(columns)})
                else:
                    return self.__setitem__(index, columns)
            case int():
                columns = self._df.with_columns(**{self._df.columns[index] : pl.lit(columns)})
            case Iterable():
                if not isinstance(columns, tuple):
                    columns = tuple(map(tuple,columns.T))
                assert len(columns) == len(index), \
                    f'{self} expected a tuple of {len(index)} tables, not {len(columns)}.'
                columns = self._df.with_columns(**{self.head[i]: pl.lit(columns[i]) for i in range(len(index))})
            case slice():
                return self.__setitem__(self.head[index], columns)
            case _:
                raise NotImplementedError(f'{self} does not support indexing by {index}.')
        self(columns)

    def __eq__(self, other: TableData | Self) -> bool:
        """ Equality of ``self`` and ``other``.

        Args:
            other: The other to compare with.

        Returns: Not implemented if ``other`` is not of admissible Type.
            Otherwise returns comparison with ``self.pl``, ``self.np`` or ``self.tc``
            matching the type of ``other``.
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

    def __call__(self, update: TableData | Self | None = None, write_csv: bool = True) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            update: The tableData updates.
            write_csv: Whether to write the updated ``self.df`` to ``.csv``.

        Returns: ``self``.
        Raises: AssertionError if ``self.schema`` fails verification.
        """
        match update:
            case None:
                if write_csv: self._df.write_csv(self._path, **self.writeOptions)
                return self
            case Table():
                self._df = update._df
            case DataFrame():
                self._df = update
            case _:
                self._df = DataFrame(update, orient='row')
        percentages = [_head for _head, _type in self._df.schema.items() if _type == String
                       and self._df.select(pl.col(_head).drop_nulls().str.ends_with('%').all()).item()]
        self._df = self._df.with_columns(pl.col(percentages).str.head(-1).cast(Float, strict=False) / 100)
        self._df = self._df.with_columns(pl.col(*Floats).cast(Float))
        self._df = self._df.with_columns(pl.col(pl.Boolean, *Ints).cast(String))
        self._df = self._df.with_columns(pl.col(String).cast(Category))
        if write_csv: self._df.write_csv(self._path, **self.writeOptions)
        return self

    def __init__(self, path: PathLike, table: DataFrame | Self | None = None):
        """ Construct ``self`` from a ``.csv`` file or DataFrame.

        Args:
            path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
            table: The ``DataFrame | Table`` to store. If ``None``, ``self`` is read from ``path``,
                otherwise ``self`` is stored in ``path`` (which is overwritten if existing).
        """
        super().__init__(path)
        if table is None:
            self(pl.read_csv(self._path, **self.readOptions))
        else:
            self(table)

    @classmethod
    def create(cls, path: PathLike, df: DataFrame | Self) -> Self:
        """ Create a Table at ``path``, overwriting.

        Args:
            path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            df: The DataFrame to store.

        Returns: The Table created.
        Raises: AssertionError if ``df`` is unacceptable.
        """
        return cls(path, getattr(df, 'df', df))

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
        """ Conjoin multiple heads in ``src`` to single head ``dst.head``,
        overwriting ``dst`` with a Table.
        Conjoining is top down, so the triple head ``(a,b,c)`` becomes the single head ``a│b│c``.
        The first column is presumed to be an index. Any other column with an empty head level is dropped.

        Args:
            src: The source Path. A ``.csv`` extension is implicitly appended.
            dst: The destination ``Table.path``, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            headcount: Counts the number of heads in ``src.csv``.
        Returns: The Table now stored at ``dst``.
        """
        src = cls.extAppend(src)
        head = pl.read_csv(src, **(cls.readOptions | {'has_header': False,'n_rows': headcount})
                            ).fill_null('││')
        src = pl.read_csv(src, **(cls.readOptions | {'has_header': False, 'skip_rows': headcount}))
        head = ['│'.join(map(str, head[_head].to_list())).join(('│',)*2) for _head in head.columns]
        head[0] = '│' + head[0].lstrip('│')
        src = DataFrame(src, head).drop([col for col in head[1:] if '││' in col])
        src.columns = [_head[1:-1] for _head in src.columns]
        return cls.create(dst, src)

    @classmethod
    def unjoinHead(cls, src: Table, dst: PathLike) -> Path:
        """ Unjoin ``src.head`` into multi-headed ``dst.csv``, overwriting.
        Unjoining is from the left, so ``a│b│c`` becomes the triple head ``(a,b,c)``.
        The first column is presumed to be an index.
        Every other column must produce the same headcount (number of heads).

        Args:
            src: The source Table.
            dst: The destination Path, overwritten if existing. A ``.csv`` extension is implicitly appended.
        Returns: ``dst``, now containing the unjoined ``dst.csv``.
        Raises: AssertionError if ``src.head`` cannot be unjoined due to inconsistent headcount (rows).
        """
        head = [_head.split('│') for _head in src.head]
        if len(head) > 1:
            headcount = {len(_head) for _head in head[1:]}
            assert len(headcount) == 1, f'{src} has incompatible headcounts {headcount}.'
            head[0] += [None] * (headcount.pop() - len(head[0]))
        head = DataFrame(head)
        with open(cls.extAppend(dst), "w", newline="", encoding="utf-8") as file:
            head.write_csv(file, **(cls.writeOptions | {'include_header': False}))
            src.df.write_csv(file, **(cls.writeOptions | {'include_header': False}))
        return Path(dst)


class DataBase(Store):
    """ ``Schema(NamedTuple)`` in a folder alongside Meta. Abstract BaseClass for any model.

    *DataBase* SubClasses must be implemented according to the template (copy and paste it)::

        class MyDataBase(DataBase):

            class Schema(NamedTuple):

                names[i]: TableData | Table = defaults[names[i]]
                \"\"\" Normally a DataFrame. If no value is appropriate, default to the TableType.\"\"\"
                ...

                def __call__(self, name: str) -> TableData | Table | MetaData:
                    \"\"\" Returns the Table named ``name``.\"\"\"
                    return getattr(self, name)

            schema: Schema[type[Table], ...] = Schema(**{name: Table for name in Schema._fields})
            \"\"\" The ``Schema`` of TableTypes, to communicate ``readOptions, writeOptions``.\"\"\"

            defaultMeta: MetaData = {'schema': {name: TableType.__name__
                                                for name, TableType in schema._asdict().items()}, }
            \"\"\" Class default ``self.meta``.\"\"\"
    """

    class Schema(NamedTuple):
        """ Must be overridden. """
        NotImplemented: TableData | Table = Table

        def __call__(self, name: str) -> TableData | Table | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)


    schema: Schema[type[Table], ...] = Schema(**{name: Table for name in Schema._fields})
    """ Class attribute of the form ``Schema(**{names[i]: TableTypes[i], ...})``, 
    where ``TableTypes[i]`` is a SubClass of Table. Must be overridden."""

    defaultMeta: MetaData = {'schema': {name: TableType.__name__ for name, TableType in schema._asdict().items()}}
    """ Class attribute. Should be overridden."""

    class IndexP(IndexP):
        """ ``self[index]`` accesses ``Schema`` by IndexLike ``index``.  ``len(self)`` counts the Tables."""

    class EqualsP(EqualsP):
        """ ``self == other`` compares Meta and ``tables`` between two DataBases. """

    class CreateP(CreateP):
        pass

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        """ ``self(**tables)`` updates and writes ``Schema`` (``self.meta(**updates)`` updates Meta."""

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    @property
    def tables(self) -> Schema:
        """ The ``Schema`` currently in ``self``."""
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
        match index:
            case str():
                return self._tables(index)
            case Iterable():
                return tuple(self[i] for i in index)
            case _:
                return self._tables[index]     # int or slice

    def __setitem__(self, index: IndexLike, tables: TableData | Table | tuple[TableData | Table, ...]):
        """ Indexer sets the Table (s) named or sliced by ``index``."""
        match index:
            case str() | int ():
                tables = {index: tables}
            case Iterable():
                assert isinstance(tables, tuple) and len(tables) == len(index), \
                    f'{self} expected a tuple of {len(index)} tables, not {len(tables)}.'
                index = tuple((self.index()[i] if isinstance(i, int) else i for i in index))
                tables = {index[i]: tables[i] for i in range(len(index))}
            case slice():
                return self.__setitem__(self.names()[index], tables)
            case _:
                raise NotImplementedError(f'{self} does not support indexing by {index}.')
        self(**tables)

    def __call__(self, **tables: TableData | Table) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            path: Optionally, an update to ``self.path``, overwritten if existing.
            **tables: Updates to ``self`` in the form ``names[i]=Table[i], ...``.

        Returns: ``self``.
        """
        for name, table in tables.items():
            self._tables(name)(table)
        return self

    def __init__(self, path: PathLike, **tables: DataFrame | Table):
        """ Read the *DataBase* in ``path``.
        Reading is lazy: If ``names[i]`` occurs in ``**tables`` it's Table is not read, just updated.
        Overrides must call ``super(DataBase).__init__(path, **tables)`` as a matter of priority.

        Args:
            path: The Path to read from.
            **tables: Tables to update those read, in the form ``names[i]=tables[i], ...``.

        Raises:
            FileNotFoundError: If ``path`` lacks ``self.meta`` or any member of
                ``self.names`` not mentioned in ``**tables``.
        """
        super().__init__(path)
        try:
            self._meta = Meta(self._meta_in(path))
            self._tables = self.Schema(**{name:
                                                    TableType.create(self._path / name, tables[name])
                                                    if name in tables and tables[name] is not None
                                                    else TableType(self._path / name)
                                               for name, TableType in self.schema._asdict().items()})
        except FileNotFoundError as error:
            print(f'DataBase "{self}" is trying to read a non-existent Table. Did your script mean to call '
                  f'{type(self).__qualname__}.create("{str(self)}") '
                  f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error


    @classmethod    # Class Property
    def names(cls) -> tuple[str, ...]:
        """ ``(names[i], ...)`` of table names for this ``Tables`` Class."""
        return cls.Schema._fields

    @classmethod    # Class Property
    def defaults(cls) -> dict[str, DataFrame]:
        """ ``{names[i]: DataFrame[i], ...}`` of default tables for this ``Tables`` Class."""
        return cls.Schema._field_defaults

    @classmethod
    def create(cls, path: PathLike, **tableData_and_metaData: DataFrame | Table | MetaData) -> Self:
        """ Create a *DataBase* in ``path``.

        Args:
            path: The folder to store the *DataBase* in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **tableData_and_metaData: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

        Returns: The *DataBase* created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMeta | (tableData_and_metaData.pop('meta', {}))))
        return cls(path, **(cls.defaults() | tableData_and_metaData))

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
    def delete(cls, path: PathLike) -> Path:
        """ Delete all *DataBase* files in ``path``, retaining ``path`` and any other files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: Path to the *DataBase* to delete.
        Returns: ``path``, which still exists.
        Raises: AssertionError if ``path`` is not a folder.
        """
        path = Path(path)
        assert path.is_dir(), f'Path {path} is not a folder.'
        Meta.delete(cls._meta_in(path))
        for name in cls.names():
            Table.delete(path / name)
        if not tuple(path.iterdir()):
            path.rmdir()
        return path

    @staticmethod
    def _meta_in(path: PathLike) -> Path:
        return Path(path) / 'meta'
