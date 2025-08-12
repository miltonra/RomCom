#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2025 Robert A. Milton
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

""" Abstract and concrete base classes for RomCom models."""

from __future__ import annotations

from astroid.bases import UnionType

from .definitions import *

from shutil import copyfile, copytree, rmtree
from json import load, dump


PathLike: TypeAlias = Path | str
""" = ``Path | str``. ClassAttribute aliasing valid Types for specifying the ``path`` to a Store."""


class Store(ABC):
    """ Base class for any stored class. Users are not expected to subclass this class directly."""

    ext: str = ''
    """Class attribute specifying the file extension terminating ``self.path``. 
    Override if and only if the derived class must be stored in a file.
    Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder."""

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

    class StrReprP(StrReprP):
        pass

    @property
    def path(self) -> Path:
        """ The ``Path`` to this ``Store``, without ``cls.ext``.
        File extension is internal, meaning ``self._path = self.path + cls.ext``."""
        return self._path.with_suffix('') if self.ext else self._path

    def __repr__(self) -> str:
        """ The ``Path`` to this ``Store``."""
        return str(self.path)

    def __str__(self) -> str:
        """ The name of this ``Store`` - i.e. ``self.path.name``."""
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
            path: The ``Path`` to ``self``. Do not include an extension.
        """
        self._path = self.extAppend(path)

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
    def create(cls, path: PathLike, **kwargs: Any) -> Self | Path:
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
    def copy(cls, src: PathLike, dst: PathLike) -> Self | Path:
        """ Copy ``src`` to ``dst``, overwriting only files in common.

        Overrides should copy an instance of ``cls`` called ``src`` to ``Store.create(dst)``
        and return the copy.

        Args:
            src: The source ``Path``, which must be a folder or a file.
            dst: The destination ``Path``, which may or may not exist.
            
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
            path: The ``Path`` to delete.
            
        Returns: ``path``, which no longer exists.
        """
        path = cls.extAppend(path)
        if path.is_dir():
            rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=False)
        return path


MetaData: TypeAlias = dict[str, Any]
""" = ``dict[str, Any]``. Type for passing metadata such as options or ``**kwargs``. """


class Meta(Store, dict):
    """ Concrete class encapsulating metadata stored in a ``.json`` file."""

    ext: str = '.json'  #: ext: Class attribute specifying the file extension of Meta instances.

    class EqualityP(EqualityP):
        """ ``self == other`` is inherited directly from ``dict``. """

    class IndexP(IndexP):
        """ ``self[key]``, ``len(self)`` are inherited from ``dict``, except that ``self[key]`` writes to file."""

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

    def __setitem__(self, key, value):
        """ Act as a ``dict``, then write to ``.json``."""
        super().__setitem__(key, value)
        self()

    def __init__(self, path: PathLike, **data: Any):
        """ Construct ``self`` from a ``.json`` file or ``MetaData``.
        This is read *or* write, *never* both: ``path`` is read *only* if ``**data`` is absent.

        Args:
            path: The Path (file) to store ``self``. A ``.json`` extension is automatically appended.
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
        """ Create a ``Meta`` at ``path``, overwriting.

        Args:
            path: The ``Path`` (file) to store ``self``, overwritten if existing.
                A ``.json`` extension is automatically appended.
            **data: The ``MetaData`` to store.

        Returns: The ``Meta`` created.
        Raises: TypeError if ``data`` is empty.
        """
        if not data: raise ValueError("Meta cannot be created with no data")
        return cls(cls.mkdir(path), **data)

    @classmethod
    def copy(cls, src: Meta, dst: PathLike) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source ``Meta``.
            dst: The destination ``Path``, overwritten if existing.
                A ``.json`` extension is automatically appended.

        Returns: The ``Meta`` now stored at ``dst.json``.
        """
        return cls.create(dst, **src)


Matrix: TypeAlias = Pd.DataFrame | Np.Matrix | Tc.Matrix
""" = ``Pd.DataFrame | Np.Matrix | Tc.Matrix``. Types which a DataBase Table accepts."""


class Table(Store):
    """ Concrete class encapsulating a ``pd.DataFrame`` backed by a ``.csv`` file.

    This class may be usefully overridden to provide bespoke read and write options for
    file operations. Subclasses should follow the template (copy and paste it)::

        class MyTable(Table):

            readOptions: MetaData = Table.readOptions | {'myOption': 'myValue'}
            \"\"\" File read options passed directly to
            `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__.\"\"\"

            writeOptions: MetaData = Table.writeOptions | {'myOption': 'myValue'}
            \"\"\" File write options passed directly to
            `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`__.\"\"\"
    """

    ext: str = '.csv'   #: Class attribute specifying the file extension of Table objects.

    readOptions: MetaData = {'encoding': 'utf-8-sig', 'index_col': 0, 'header': 0}
    """ File read options passed directly to `pd.read_csv <https: //pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    writeOptions: MetaData = {'encoding': 'utf-8-sig'}
    """ File write options passed directly to `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`__."""

    class EqualityP(EqualityP):
        """ ``self == other`` compares ``self.pd``,``self.np`` or ``self.tc`` matching the the type of ``other``. """

    class CreateP(CreateP):
        """ Creates a new Table at ``path`` from ``data: Matrix | Table``. """

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        pass

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    class StrReprP(StrReprP):
        pass

    @property
    def pd(self) -> Pd.DataFrame:
        """ The ``Pd.DataFrame`` stored in ``self``."""
        return self._pd

    @property
    def np(self) -> Np.Matrix:
        """ The ``Np.Matrix`` stored in ``self``."""
        return self.pd.to_numpy()

    @property
    def tc(self) -> Tc.Matrix:
        """ The ``Tc.Matrix`` stored in ``self``."""
        return tc.from_numpy(self.np)

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
            raise IndexError(f'{repr(self)} has shape {self._pd.shape} '
                             f'which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            data = np.diag(np.diagonal(data))
        return self(data)

    def __eq__(self, other: Table | Matrix) -> bool:
        """ Equality of ``self`` and ``other``.

        Args:
            other: The other to compare with.

        Returns: Not implemented if ``other`` is not of admissible Type.
            Otherwise returns comparison with ``self.pd``, ``self.np`` or ``self.tc`` matching the type of ``other``.
        """
        match other:
            case Table():
                return self.pd.astype(str).equals(other.pd.astype(str))
            case pd.DataFrame():
                return self.pd.astype(str).equals(other.astype(str))
            case Np.Matrix():
                return np.array_equal(self.np.astype(str), other.astype(str))
            case Tc.Matrix():
                return tc.equal(self.tc, other)
            case _:
                return NotImplemented

    def __call__(self, update: Self | Matrix | None = None) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            update: The data updates.

        Returns: ``self``.
        """
        if isinstance(update, Table):
            self._pd = update.pd.copy(deep=True)
        elif isinstance(update, pd.DataFrame):
            self._pd = update.copy()
        elif isinstance(update, Np.Matrix):
            self._pd.iloc[:, :] = update
        elif isinstance(update, Tc.Matrix):
            self._pd.iloc[:, :] = update.numpy()
        self._pd.to_csv(self._path, **self.writeOptions)
        return self

    def __init__(self, path: PathLike, table: Self | Pd.DataFrame | None = None):
        """ Construct ``self`` from a ``.csv`` file or ``Pd.DataFrame``.

        Args:
            path: The ``Path`` (file) to store ``self``. A ``.csv`` extension is automatically appended.
            table: The ``Table | Pd.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                otherwise ``self`` is stored in ``path`` (which is overwritten if existing).
        """
        super().__init__(path)
        if table is None:
            self(pd.read_csv(self._path, **self.readOptions))
        else:
            self(table)
        if self._pd.columns.nlevels > 1:
            for n in range(self._pd.columns.nlevels):
                self._pd.columns = self._pd.columns.set_levels(self._pd.columns.levels[n].astype(str), level=n)  # Ensure column names are strings
        else:
            self._pd.columns = self._pd.columns.astype(str)

    @classmethod
    def create(cls, path: PathLike, data: Self | Matrix, **kwargs: Any) -> Self:
        """ Create a ``Table`` at ``path``, overwriting.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            data: The table to store.
            **kwargs: KeywordArguments passed directly to `pd.DataFrame(...)`_.

        Returns: The ``Table`` created.

        .. _pd.DataFrame(...): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        """
        data = pd.DataFrame(data.pd if isinstance(data, Table) else data, **kwargs)
        return cls(cls.mkdir(path), data)

    @classmethod
    def copy(cls, src: Self, dst: PathLike) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source ``Table``.
            dst: The destination ``Path``, overwritten if existing.
                A ``.csv`` extension is automatically appended.

        Returns: The ``Table`` now stored at ``dst``.
        """
        Store.copy(src._path, cls.extAppend(dst))
        return cls(dst)


class DataBase(Store):
    """ ``NamedTables(NamedTuple)`` in a folder alongside ``Meta``. Abstract base class for any model.

    ``DataBase`` SubClasses must be implemented according to the template (copy and paste it)::

        class MyDataBase(DataBase):

            class NamedTables(NamedTuple):

                names[i]: Table | Matrix = defaults[names[i]].pd
                \"\"\" Normally a ``pd.DataFrame``. If no default is appropriate, use the Table Type\"\"\"
                ...

                def __call__(self, name: str) -> Table | Matrix | MetaData:
                    \"\"\" Returns the Table named ``name``.\"\"\"
                    return getattr(self, name)

            Tables: NamedTables[type[Table], ...] = NamedTables()
            \"\"\" The ``NamedTables`` of Table Types, to communicate ``readOptions, writeOptions``.\"\"\"

            defaultMeta: MetaData = {'Tables': {name: TableType.__name__ for name, TableType in Tables._asdict().items()}}
            \"\"\" Class default ``self.meta``.\"\"\"
    """

    class NamedTables(NamedTuple):
        """ Must be overridden. """
        NotImplemented: Table | Matrix = Table

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)


    Tables: NamedTables[type[Table], ...] = NamedTables(**{name: Table for name in NamedTables._fields})
    """ Class attribute of the form ``NamedTables(**{names[i]: Type[i], ...})``, 
    where ``Type[i]`` is a subclass of ``Table``. Must be overridden."""

    defaultMeta: MetaData = {'Tables': {name: TableType.__name__ for name, TableType in Tables._asdict().items()}}
    """ Class attribute. Should be overridden."""

    class IndexP(IndexP):
        """ ``self[names]`` accesses ``NamedTables`` by ``str | int | Iterable | slice``. """

    class CreateP(CreateP):
        pass

    class ReadP(ReadP):
        pass

    class UpdateP(UpdateP):
        """ ``self(**tables) updates and writes ``NamedTables`` (``self.meta(**updates)`` updates ``Meta``."""

    class DeleteP(DeleteP):
        pass

    class CopyP(CopyP):
        pass

    class StrReprP(StrReprP):
        pass

    @property
    def tables(self) -> NamedTables:
        """ The ``NamedTables`` currently in ``self``."""
        return self._tables

    @property
    def meta(self) -> Meta:
        """ The ``Meta`` currently in ``self``."""
        return self._meta

    def __eq__(self, other: Any) -> bool:
        """ Equality of meta, namedTables and names()."""
        if isinstance(other, DataBase):
            return (self.names() == other.names() and self.meta == other.meta
                    and self._tables == other.tables)
        return NotImplemented

    def __len__(self) -> int:
        """ Counts the ``Table`` s in ``self``. """
        return len(self._tables)

    def __getitem__(self, names: str | int | Iterable[str | int] | slice) -> Table | tuple[Table, ...]:
        """ Indexer returns the ``Table`` (s) named or sliced by ``names``. """
        if isinstance(names, str):
            return self._tables(names)
        elif isinstance(names, Iterable):
            return tuple(self[named] for named in names)
        else:
            return self._tables[names]     # int or slice

    def __setitem__(self, names: str | int | Iterable[str | int] | slice, tables: Table | Matrix | tuple[Table | Matrix, ...]):
        """ Indexer sets the ``Table`` (s) named or sliced by ``names``."""
        if isinstance(names, str):
            tables = {names: tables}
        elif isinstance(names, int):
            tables = {names: tables}
        elif isinstance(names, Iterable):
            if not (isinstance(tables, tuple) and len(tables) == len(names)):
                raise IndexError(f'Expected a tuple of {len(names)} tables, not {len(tables)}.')
            names = tuple((self.names()[named] if isinstance(named, int) else named for named in names))
            tables = {names[i]: tables[i] for i in range(len(names))}
        elif isinstance(tables, tuple):
            return self.__setitem__(self.names()[names], tables)
        else:
            return NotImplemented
        self(**tables)

    def __call__(self, **tables: Table | Matrix) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            path: Optionally, an update to ``self.path``, overwritten if existing.
            **tables: Updates to ``self`` in the form ``names[i]=Table[i], ...``.

        Returns: ``self``.
        """
        for name, table in tables.items():
            self._tables(name)(table)
        return self

    def __init__(self, path: PathLike, **tables: Table | Pd.DataFrame):
        """ Read the ``DataBase`` in ``path``.
        Reading is lazy: If ``names[i]`` occurs in ``**tables`` it's ``Table`` is not read, just updated.
        Overrides must call ``super(DataBase).__init__(path, **tables)`` as a matter of priority.

        Args:
            path: The ``Path`` to read from.
            **tables: ``Table`` s to update those read, in the form ``names[i]=tables[i], ...``.

        Raises:
            FileNotFoundError: If ``path`` lacks ``self.meta`` or any member of
                ``self.Tables.names`` not mentioned in ``**tables``.
        """
        super().__init__(path)
        try:
            self._meta = Meta(self._meta_in(path))
            self._tables = self.NamedTables(**{name:
                                                        TableType.create(path / name, tables[name])
                                                        if name in tables and tables[name] is not None
                                                        else TableType(path / name)
                                               for name, TableType in self.Tables._asdict().items()})
        except FileNotFoundError as error:
            print(f'DataBase "{self}" is trying to read a non-existent Table. Did your script mean to call '
                  f'{type(self).__qualname__}.create("{str(self)}") '
                  f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error


    @classmethod    # Class Property
    def names(cls) -> tuple[str, ...]:
        """ ``(names[i], ...)`` of table names for this ``Tables`` class."""
        return cls.NamedTables._fields

    @classmethod    # Class Property
    def defaults(cls) -> dict[str, Pd.DataFrame]:
        """ ``{names[i]: Pd.DataFrame[i], ...}`` of default tables for this ``Tables`` class."""
        return cls.NamedTables._field_defaults

    @classmethod
    def create(cls, path: PathLike, **tables_and_meta: Table | Pd.DataFrame | MetaData) -> Self:
        """ Create a ``DataBase`` in ``path``.

        Args:
            path: The folder to store the ``DataBase`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **tables_and_meta: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

        Returns: The ``DataBase`` created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMeta | (tables_and_meta.pop('meta', {}))))
        return cls(path, **(cls.defaults() | tables_and_meta))

    @classmethod
    def copy(cls, src: Self, dst: PathLike) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source ``DataBase``.
            dst: The destination ``Path``, which may or may not exist.

        Returns: The ``DataBase`` now stored in ``dst``.
        """
        return cls.create(dst, meta=src.meta, **src._tables._asdict())

    @classmethod
    def delete(cls, path: PathLike, ignoreErrors: bool=False) -> Path:
        """ Delete all ``DataBase`` files in ``path``, retaining ``path`` and any other files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: ``Path`` to the ``DataBase`` to delete.
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
