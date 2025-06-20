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

""" Abstract and concrete base classes for RomCom Models."""

from __future__ import annotations

from .definitions import *

from shutil import copyfile, copytree, rmtree
from json import load, dump


MetaData = dict[str, Any]
"""Type for passing metadata as ``**kwargs``."""

Matrix = Union[Pd.DataFrame, Np.Matrix, Tc.Matrix]
"""Types which a DataBase Table accepts."""


class Store(ABC):
    """ Base class for any stored class. Users are not expected to subclass this class directly."""

    Path = Path | str
    """ Class attribute aliasing Types used to specify the ``path`` to a Store. Do not override."""

    ext: str = ''
    """Class attribute specifying the file extension terminating ``self.path``. 
    Override if and only if the derived class must be stored in a file.
    Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder."""

    @property
    def path(self) -> Path:
        """ The ``Path`` to this ``Store``, without ``cls.ext``.
        File extension is internal, meaning ``self._path = self._path + cls.ext``."""
        return self._path.with_suffix('') if self.ext else self._path

    def __repr__(self) -> str:
        """ The ``Path`` to this ``Store``.

        :meta public:
        """
        return str(self._path)

    def __str__(self) -> str:
        """ The ``Path`` to this ``Store``, abbreviated.

        :meta public:
        """
        return self._path.stem if self.ext else self._path.name

    @abstractmethod
    def __call__(self, **data) -> Self:
        """ Update and store ``self``.

        Args:
            **data: Data to update.

        Returns: ``self``.
        """
        raise NotImplementedError()

    @abstractmethod
    def __init__(self, path: Path):
        """ Construct ``self``.

        Overrides should call ``super(Store).__init__(path)`` as a matter of priority.
        Then they should read ``self`` from ``self._path`` or write ``self`` in ``self._path``.

        Args:
            path: The ``Path`` to ``self``. Do not include an extension.
        """
        self._path = self.mkdir(path)

    @classmethod
    def extAppend(cls, path: Path) -> Path:
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
    def mkdir(cls, path: Path) -> Path:
        """ Create ``path.parent``, with a subfolder ``path`` if ``cls.ext == ''``.

        Args:
            path: The folder to create, or a child file of the folder to create.

        Returns: ``Path(path)`` with ``cls.ext`` appended.
        """
        path = cls.extAppend(path)
        if cls.ext:
            path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        else:
            path.mkdir(mode=0o777, parents=True, exist_ok=True)
        return path

    @classmethod
    @abstractmethod
    def create(cls, path: Path) -> Self | Path:
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
    def copy(cls, src: Path, dst: Path) -> Self | Path:
        """ Copy ``src`` to ``dst``, overwriting only files in common.

        Overrides should copy an instance of ``cls`` called ``src`` to ``Store.create(dst)``,
        and return the copy.

        Args:
            src: The source ``Path``, which must be a folder or a file.
            dst: The destination ``Path``, which may or may not exist.
            
        Returns: ``dst``.
        
        Raises:
            FileNotFoundError: If ``src`` does not exist.
            FileExistsError: If attempting to overwrite a file with a folder.
        """
        src, dst = cls.extAppend(src), cls.mkdir(dst)
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


class Meta(Store, dict):
    """ Concrete class encapsulating metadata stored in a ``.json`` file."""

    ext: str = '.json'  #: ext: Class attribute specifying the file extension of Meta instances.

    def __call__(self, **data: Any) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            **data: Data to update ``self.data``.

        Returns: ``self``.
        """
        self.update(data)
        with open(self._path, mode='w') as file:
            dump(self, file, indent=4)
        return self

    def __setitem__(self, key, value):
        """ Indexer sets the ``value`` indexed by ``key``."""
        super().__setitem__(key, value)
        self()

    def __init__(self, path: Store.Path, **data: Any):
        """ Construct ``self`` from a ``.json`` file or ``MetaData``.

        Args:
            path: The Path (file) to store ``self``. A ``.json`` extension is automatically appended.
            **data: The ``MetaData`` to store. If absent, ``self.data`` is read from ``path``,
                otherwise ``self.data=data`` is stored in ``path`` (which is overwritten if existing).
        """
        super(Meta, self).__init__(path)
        if data == {}:
            with open(self._path, mode='r') as file:
                data = load(file)
        super(Store, self).__init__(**data)
        self()

    @classmethod
    def create(cls, path: Store.Path, **data: Any):
        """ Create a ``Meta`` at ``path``, overwriting.

        Args:
            path: The ``Path`` (file) to store ``self``, overwritten if existing.
                A ``.json`` extension is automatically appended.
            **data: The ``MetaData`` to store.

        Returns: The ``Meta`` created.
        """
        return cls(path, **data)

    @classmethod
    def copy(cls, src: Meta, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source ``Meta``.
            dst: The destination ``Path``, overwritten if existing.
                A ``.json`` extension is automatically appended.

        Returns: The ``Meta`` now stored at ``dst.json``.
        """
        return cls(dst, **src)


class Table(Store):
    """ Concrete class encapsulating a ``pd.DataFrame`` backed by a ``.csv`` file.

    This class may be usefully overridden to provide bespoke read and write options for
    file operations. Subclasses should follow the template (copy and paste it)::

        class MyTable(Table):

        class Options(NamedTuple):

            read: MetaData =  {'index_col': 0}  #: Read options passed to ``pd.read_csv``.
            write: MetaData =  {}   #: Write options passed to ``pd.DataFrame.to_csv``.

            @classmethod
            def default(cls) -> MetaData:
                \"\"\" Returns the default Options as ``cls.read | cls.write``.\"\"\"
                return cls._field_defaults['read'] | cls._field_defaults['write']
    """
    class Options(NamedTuple):

        read: MetaData =  {'index_col': 0}  #: Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   #: Write options passed to ``pd.DataFrame.to_csv``.

        @classmethod
        def default(cls) -> MetaData:
            """ Returns the default Options as ``cls.read | cls.write``."""
            return cls._field_defaults['read'] | cls._field_defaults['write']


    ext: str = '.csv'   #: Class attribute specifying the file extension of Table objects.

    writeOptions: list[str] = ['sep', 'na_rep', 'float_format']
    """ Class attribute listing kwargs which will be interpreted as write options. 
    All other kwargs are interpreted as read options. 
    To specify a separator, use ``delimiter`` as read option and ``sep`` as write option. """

    @property
    def options(self) -> MetaData:
        """ A ``dict of options for file operations involving ``self``.
        Any option not in ``Table.writeOptions`` is stored in ``self.options.read`` and passed to ``pd.read_csv``.
        Any option in ``Table.writeOptions`` is stored in ``self.options.write``
        and passed to ``pd.DataFrame.to_csv``.
        The setter updates via logical or ``|=``, so existing values are retained unless explicitly updated.
        """
        return self._options.read | self._options.write

    @options.setter
    def options(self, update: MetaData):
        write = {key: update.pop(key) for key in self.writeOptions if key in update}
        self._options._replace(read =self._options.read | update, write =self._options.write | write)

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
        """ The ``TF.Matrix`` stored in ``self``."""
        return tc.from_numpy(self.np)

    def broadcast_to(self, target_shape: Tuple[int, int], is_diagonal: bool = True) -> Self:
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

    def __call__(self, data: Self | Matrix | None, **options: Any) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            data: The data updates.
            **options: Updates ``self.options``, before storing ``self``.

        Returns: ``self``.
        """
        if isinstance(data, Table):
            self._pd = data.pd.copy()
        elif isinstance(data, pd.DataFrame):
            self._pd = data.copy()
        elif isinstance(data, Np.Matrix):
            self._pd.iloc[:, :] = data
        elif isinstance(data, Tc.Matrix):
            self._pd.iloc[:, :] = data.numpy()
        self.options = options
        self._pd.to_csv(self._path, **self._options.write)
        return self

    def __init__(self, path: Store.Path, data: Self | Pd.DataFrame | None = None, **options: Any):
        """ Construct ``self`` from a ``.csv`` file or ``Pd.DataFrame``.

        Args:
            path: The ``Path`` (file) to store ``self``. A ``.csv`` extension is automatically appended.
            data: The data to store. If ``None``, ``data`` is read from ``path``,
                otherwise ``data`` is stored in ``path`` (which is overwritten if existing).
            **metadata: Updates ``self.readMetaData`` if ``data is None``,
                otherwise updates ``self.writeMetaData``.
        """
        super().__init__(path)
        self._options = self.Options()
        self.options = options
        if data is None:
            self(pd.read_csv(self._path, **self._options.read))
        else:
            self(data)

    @classmethod
    def create(cls, path: Store.Path, data: Self | Matrix | None = None,
               index: Pd.Index | Np.Array = None, columns: Pd.Index | Np.Array = None,
               dtype: Np.DType | None = None, copy: bool | None = None, **metadata) -> Self:
        """ Create a ``Table`` at ``path``, overwriting.

        Args:
            path: The ``Path`` to store this DataTable, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            data: The data to store. If ``None``, a ``Pd.DataFrame`` is read from ``.csv``.
                See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            index: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            columns: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            dtype: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            copy: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            **metadata: MetaData passed to
                `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_
                or
                `pd.DataFrame.to_csv`_.

        Returns: The ``DataTable`` created.

        .. _pd.DataFrame.to_csv: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
        """
        return cls(path, pd.DataFrame(data.pd if isinstance(data, Table) else data, index, columns, dtype, copy),
                   **metadata)

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source ``DataTable``.
            dst: The destination ``Path``, overwritten if existing.
                A ``.csv`` extension is automatically appended.

        Returns: The ``DataTable`` now stored at ``dst.csv``.
        """
        return cls(dst, src.pd, **src.options)


class DataBase(Store):
    """ ``NamedTables(NamedTuple)`` in a folder alongside ``Meta``. Abstract base class for any model.

    ``DataBase`` subclasses must be implemented according to the template (copy and paste it)::

        class MyDataBase(DataBase):

            class NT(NamedTuple):

                names[i]: Table | Matrix | MetaData = pd.DataFrame(defaults[names[i]].pd)   #: Comment
                ...

                def __call__(self, name: str) -> Table | Matrix | MetaData:
                    \"\"\" Returns the Table named ``name``.\"\"\"
                    return getattr(self, name)


            options: NamedTables[MetaData] = NamedTables(**{name: table.options for name, table in {}.items()})
            \"\"\" Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
            Override as necessary for bespoke ``Table.options``.
            Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
            the remainder populate ``self[i].options.read``.\"\"\"

            defaultMetaData: MetaData = {'Tables': Tables.options._asdict()}
    """

    class NamedTables(NamedTuple):
        """ Must be overridden. """
        NotImplemented: Table | Matrix = pd.DataFrame(((f'Attribute type should be Table in '
                                                        f'any implementation.',),))  #: :meta private:

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)


    options: NamedTables[MetaData] = NamedTables(**{name: table.options for name, table in {}.items()})
    """ Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``. 
    Override as necessary for bespoke ``Table.options``.
    Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
    the remainder populate ``self[i].options.read``."""

    #: Class attribute. Should be overridden.
    defaultMetaData: MetaData = {'Tables': options._asdict()}

    @property
    def nt(self) -> NamedTables:
        """ The ``NamedTables`` currently in ``self``."""
        return self._nt

    @property
    def meta(self) -> Meta:
        """ The ``Meta`` currently in ``self``."""
        return self._meta

    def __len__(self) -> int:
        """ Counts the ``Table`` s in ``self``. """
        return len(self._nt)

    def __getitem__(self, name: str | slice) -> Table | Tuple[Table, ...]:
        """ Indexer returns the ``Table`` (s) named or sliced by ``name``. """
        return self._nt(name) if isinstance(name, str) else self._nt[name]

    def __setitem__(self, name: str | slice , tables: Table | Matrix | Tuple[Table | Matrix, ...]):
        """ Indexer sets the ``Table`` (s) named or sliced by ``name``."""
        if isinstance(name, str):
            tables = {name: tables}
        else:
            tables = {named: tables[i] for i, named in enumerate(self.names()[name])}
        self(**tables)

    def __call__(self, **tables: Table | Matrix) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            path: Optionally, an update to ``self.path``, overwritten if existing.
            **tables: Updates to ``self`` in the form ``names[i]=Table[i], ...``.

        Returns: ``self``.
        """
        for name, table in tables.items():
            self._nt(name)(table, **self.options(name))
        return self

    def __init__(self, path: Store.Path, **tables: Table | Pd.DataFrame):
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
            self._nt = self.NamedTables(**{name:
                                               Table(path / name, tables[name], **self.options(name))
                                               if name in tables and tables[name] is not None
                                               else Table(path / name, **self.options(name))
                                           for name in self.names()})
        except FileNotFoundError as error:
            print(f'DataBase "{self}" is trying to read a non-existent Table. Did your script mean to call '
                  f'{type(self).__qualname__}.create("{str(self)}") '
                  f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error


    @classmethod    # Class Property
    def names(cls) -> Tuple[str, ...]:
        """ ``(names[i], ...)`` of table names for this ``Tables`` class."""
        return cls.NamedTables._fields

    @classmethod    # Class Property
    def defaults(cls) -> Dict[str, Pd.DataFrame]:
        """ ``{names[i]: Pd.DataFrame[i], ...}`` of default tables for this ``Tables`` class."""
        return cls.NamedTables._field_defaults

    @classmethod
    def create(cls, path: Store.Path, **tables_and_meta: Table | Pd.DataFrame | MetaData) -> Self:
        """ Create a ``DataBase`` in ``path``.

        Args:
            path: The folder to store the ``DataBase`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **tables_and_meta: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

        Returns: The ``DataBase`` created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMetaData | (tables_and_meta.pop('meta', {}))))
        return cls(path, **(cls.defaults() | tables_and_meta))

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source ``DataBase``.
            dst: The destination ``Path``, which may or may not exist.

        Returns: The ``DataBase`` now stored in ``dst``.
        """
        return cls.create(dst, meta=src.meta, **src.nt._asdict())

    @classmethod
    def delete(cls, path: Store.Path) -> Path:
        """ Delete all ``DataBase`` files in ``path``, retaining ``path`` and any other files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: ``Path`` to the ``DataBase`` to delete.
        Returns: ``path``, which still exists.
        """
        path = Path(path)
        Meta.delete(cls._meta_in(path))
        for name in cls.names():
            Table.delete(path / name)
        return path

    @staticmethod
    def _meta_in(path: Store.Path) -> Path:
        return Path(path) / 'meta'
