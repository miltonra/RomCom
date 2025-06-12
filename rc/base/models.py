#  BSD 3-Clause License.
#
#  Copyright (c) 2019-2025 Robert A. Milton. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#  following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#  following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
#  promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Abstract and concrete base classes for RomCom Models."""

from __future__ import annotations

from shutil import copyfile, copytree, rmtree
from json import load, dump

from .definitions import *


MetaData = dict[str, Any]  #: Type for passing metadata as ``**kwargs``.
Matrix = Union[Pd.DataFrame, Np.Matrix, Tc.Matrix] #: Types which a DataBase Table accepts.


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
        """ The ``Path`` to this ``Store``."""
        return self._path

    def __repr__(self) -> str:
        """ The ``Path`` to this ``Store``.

        :meta public:
        """
        return str(self._path)

    def __str__(self) -> str:
        """ The ``Path`` to this ``Store``, abbreviated.

        :meta public:
        """
        return self._path.stem if self._path.is_file() else self._path.name

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
            path: The ``Path`` to ``self``. A ``cls.ext`` suffix is automatically appended.
        """
        self._path = self.mkdir(path)

    @classmethod
    def mkdir(cls, path: Path) -> Path:
        path = Path(path).with_suffix(cls.ext)
        if cls.ext == '':
            path.mkdir(mode=0o777, parents=True, exist_ok=True)
        else:
            path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        return path

    @classmethod
    @abstractmethod
    def create(cls, path: Path) -> Self | Path:
        """ Create a folder (and its parents) if it doesn't already exist.

        Overrides should create and return an instance of ``cls``.

        Args:
            path: Where to create the folder. If ``cls.ext != ''``, the parent folder of ``path`` is created.

        Returns:
            ``path.with_suffix(cls.ext)``.

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
        src, dst = Path(src).with_suffix(cls.ext), Path(dst).with_suffix(cls.ext)

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
        path = Path(path).with_suffix(cls.ext)
        if path.is_dir():
            rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
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

        Returns: The ``Meta`` now stored at ``dst.with_suffix('.json')``.
        """
        return cls(dst, **src)


class Table(Store):
    """ Concrete class encapsulating a ``Pd.DataFrame`` backed by a ``.csv`` file."""

    writeOptions: list[str] = ['sep', 'na_rep', 'float_format']
    """ Class attribute listing kwargs which will be interpreted as write options. 
    All other kwargs are interpreted as read options. 
    To specify a separator, use ``delimiter`` as read option and ``sep`` as write option. """

    class Options(NamedTuple):
        read: MetaData =  {'index_col': 0}  # Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   # Write options passed to ``pd.DataFrame.to_csv``.

    ext: str = '.csv'   #: Class attribute specifying the file extension of DataTable instances.

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
            **metadata: Updates ``self.readMetaData`` if ``data is None``, otherwise updates ``self.writeMetaData``.
        """
        super().__init__(path)
        self._options = self.Options()
        self.options = options
        if data is None:
            self(pd.read_csv(self.path, **self._options.read))
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
                `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.

        Returns: The ``DataTable`` created.
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

        Returns: The ``DataTable`` now stored at ``dst.with_suffix('.csv')``.
        """
        return cls(dst, src.pd, **src.options)


class Tables(Store):
    """ A ``NamedTuple`` of ``Table`` s in a folder. Abstract base class for any ``DataBase.Tables``.

    ``Tables`` subclasses must be implemented according to the template (copy and paste it)::

        class MyTables(Tables):

            class NT(NamedTuple):

                names[i]: Table | Matrix | MetaData = pd.DataFrame(defaults[names[i]].pd)
                ...

                def __call__(self, name: str) -> Table | Matrix | MetaData:
                    \"\"\" Returns the Table named ``name``.\"\"\"
                    return getattr(self, name)

            options: NT[MetaData] = NT(names[i] = data[i].options[i], ...)
    """
    class NT(NamedTuple):
        """ Must be overridden. """
        NotImplemented: Table | Matrix = pd.DataFrame(((f'Attribute type should be Table in '
                                                        f'any implementation.',),))  #: :meta private:

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)

    options: NT[MetaData] = NT(**{name: {} for name in NT._fields})
    """ Class attribute of the form ``NT(**{names[i]: options[i], ...})``. 
    Override as necessary for bespoke ``Table.options``.
    Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
    the remainder populate ``self[i].options.read``."""

    # TODO: Delete if unmissed
    # @property
    # def asDict(self) -> Dict[str, Table]:
    #     """ The ``self.NT(NamedTuple)`` of ``self`` as a ``dict``."""
    #     return self._nt._asdict()

    def __len__(self) -> int:
        """ Counts the ``Table`` s in ``self``. """
        return len(self._nt)

    def __getitem__(self, name: str | slice) -> Table | Tuple[Table, ...]:
        """ Indexer returns the ``Table`` (s) named or sliced by ``name``.
            The NamedTuple constituting ``self`` is ``self.NT(*self[:]``."""
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

    def __init__(self, path: Store.Path, **tables: Table | Matrix):
        """ Read the ``Tables`` in ``path``, updating ``**tables``.

        Reading is lazy: If ``names[i]`` occurs in ``**tables`` it's ``Table`` is not read, just updated.

        Args:
            path: The ``Path`` to read from, and update to.
            **tables: Updates to ``self``, in the form ``names[i]=tables[i], ...``.

        Raises:
            FileNotFoundError: If ``path`` lacks any member of ``cls.names()`` not mentioned in ``**data``.
        """
        super().__init__(path)
        try:
            self._nt = self.NT(**{name:
                                        Table(path / name, tables[name], **self.options(name))
                                        if name in tables and tables[name] is not None
                                        else Table(path / name, **self.options(name))
                                  for name in self.names()})
        except FileNotFoundError as error:
            print(f'Tables "{self}" is trying to read a non-existent Table. Did your script mean to call '
                  f'{type(self).__qualname__}.create("{str(self)}") '
                  f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error

    @classmethod    # Class Property
    def names(cls) -> Tuple[str, ...]:
        """ ``(names[i], ...)`` of table names for this ``Tables`` class."""
        return cls.NT._fields

    @classmethod    # Class Property
    def defaults(cls) -> Dict[str, Pd.DataFrame]:
        """ ``{names[i]: Pd.DataFrame[i], ...}`` of default tables for this ``Tables`` class."""
        return cls.NT._field_defaults

    @classmethod
    def create(cls, path: Store.Path, **tables: Table) -> Self:
        """ Create a ``Tables`` (folder) in ``path``.

        Args:
            path: The folder to store the ``Tables`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **tables: ``Table``s to update ``self.table_defaults``,
                in the form ``names[i]=Table[i], ...``.

        Returns: The ``Tables`` created.
        """
        return cls(path, **(cls.defaults() | tables))

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source ``Tables``.
            dst: The destination ``Path``, which may or may not exist.

        Returns: The ``Tables`` now stored at ``dst``.
        """
        return cls(dst, **src.tables_as_dict())

    @classmethod
    def delete(cls, path: Store.Path) -> Path:
        """ Delete ``cls.names[i].with_suffix(.csv)`` from ``path``,
        retaining ``path`` and any files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: ``Path`` to the ``Tables`` to delete.

        Returns: ``path``, which still exists.
        """
        path = Path(path)
        for table_name in cls.names():
            Table.delete(path / table_name)
        return path


class DataBase(Store):
    """ ``Tables`` with ``Meta``. Abstract base class for any model.

    ``DataBase`` subclasses must be implemented according to the template (copy and paste it)::

        class MyDataBase(DataBase):

            class Tables(Tables):

                class NT(NamedTuple):

                    names[i]: Table | Matrix | MetaData = pd.DataFrame(defaults[names[i]].pd)
                    ...

                    def __call__(self, name: str) -> Table | Matrix | MetaData:
                        \"\"\" Returns the Table named ``name``.\"\"\"
                        return getattr(self, name)

                options: NT[MetaData] = NT(names[i] = data[i].options[i], ...)

            defaultMetaData: MetaData = {'Tables': Tables.options._asdict()}
    """

    class Tables(Tables):
        """ Must be overridden."""
        class NT(NamedTuple):
            """ Must be overridden."""
            NotImplemented: Table | Matrix = pd.DataFrame(((f'Attribute type should be Table in '
                                                            f'any implementation.',),))  #: :meta private:

            def __call__(self, name: str) -> Table | Matrix | MetaData:
                """ Returns the Table named ``name``."""
                return getattr(self, name)

        options: NT[MetaData] = NT(**{name: {} for name in NT._fields})
        """ Class attribute of the form ``NT(**{names[i]: options[i], ...})``. 
        Override as necessary for bespoke ``Table.options``.
        Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``table[i].options.write``,
        the remainder populate ``table[i].options.read``."""

    #: Class attribute. Should be overridden.
    defaultMetaData: MetaData = {'Tables': Tables.options._asdict()}

    @property
    def meta(self) -> Meta:
        """ The ``Meta`` currently in ``self``."""
        return self._meta

    def __len__(self) -> int:
        """ Counts the ``Table`` s in ``self``. """
        return len(self._tables)

    def __getitem__(self, name: str | slice) -> Table:
        """ Indexer returns the ``Table`` (s) named or sliced by ``name``.
            The NamedTuple constituting ``self`` is ``self.NT(*self[:]``."""
        return self._tables[name]

    def __setitem__(self, name: str | slice , tables: Table | Matrix | Tuple[Table | Matrix, ...]):
        """ Indexer sets the ``Table`` (s) named or sliced by ``name``."""
        self._tables[name] = tables

    def __call__(self, **metadata: Any) -> Self:
        """ Optimize and update ``self``.

        Args:
            **metadata: Optimization ``MetaData``.

        Returns: ``self``
        """
        self._tables(**metadata)
        return self

    def __init__(self, path: Store.Path, **tables: Table | Pd.DataFrame):
        """ Read the ``DataBase`` in ``path``.

        Overrides must call ``super(DataBase).__init__(path, **tables)`` as a matter of priority.

        Args:
            path: The ``Path`` to read from.
            **tables: ``Table`` s to update those read, in the form ``names[i]=tables[i], ...``.
            
        Raises:
            FileNotFoundError: If ``path`` lacks ``self.meta`` or any member of
                ``self.Tables.names`` not mentioned in ``**data``.
        """
        super().__init__(path)
        try:
            self._meta = Meta(self._meta_in(path))
            self._tables = self.Tables(path, **tables)
        except FileNotFoundError as error:
            print(f'DataBase "{self}" is trying to read a non-existent file. '
                  f'Did your script mean to call {type(self).__qualname__}.create("{str(self)}") '
                  f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error

    @classmethod
    def create(cls, path: Store.Path, **tables_and_meta: Table | Pd.DataFrame | MetaData) -> Self:
        """ Create a ``DataBase`` in ``path``.

        Args:
            path: The folder to store the ``DataBase`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **tables_and_meta: Data to update ``cls.Tables.table_defaults``, in the form ``names[i]=tables[i]``,
                and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.
            
        Returns: The ``DataBase`` created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMetaData | (tables_and_meta.pop('meta', {}))))
        return cls(path, **(cls.Tables.defaults() | tables_and_meta))

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source ``DataBase``.
            dst: The destination ``Path``, which may or may not exist.

        Returns: The ``DataBase`` now stored in ``dst``.
        """
        return cls.create(dst, meta=src.meta.data, **src[:].asDict())

    @classmethod
    def delete(cls, path: Store.Path) -> Path:
        """ Delete all ``DataBase`` files in ``path``, retaining ``path`` and any other files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: ``Path`` to the ``DataBase`` to delete.
        Returns: ``path``, which still exists.
        """
        Meta.delete(cls._meta_in(path))
        cls.Tables.delete(path)
        return path

    @staticmethod
    def _meta_in(path: Store.Path) -> Path:
        return Path(path) / 'meta'
