#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2024 Robert A. Milton. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
# 
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Base and concrete classes for RomComma Models."""

from __future__ import annotations

from romcomma.base.definitions import *
from shutil import copyfile, copytree, rmtree
from json import load, dump


class Store(ABC):
    """ Base class for any stored class. Users are not expected to subclass this class directly.

    Attributes:
        ext: Class attribute specifying the file extension terminating ``self.path``. Override if and only if the derived class must be stored in a file.
            Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.
        Path: Class attribute aliasing Types used to specify the ``path`` to a Store. Do not override.
    """

    ext: str = ''
    Path = Path | str

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self) -> str:
        """ The Path ``self`` is stored in."""
        return str(self._path)

    def __str__(self) -> str:
        """ The Path ``self`` is stored in, abbreviated."""
        return self._path.stem if self._path.is_file() else self._path.name

    @abstractmethod
    def __call__(self, path: Path | None) -> Self:
        """ Update and store ``self``.

        Overrides should call ``super(Stored).__call__(path)`` as a matter of priority.
        Finally, they should store the class in ``self._path``.

        Args:
            path: The updated Path to store ``self`` in. A ``cls.ext`` suffix is appended.
        Returns: ``self``.
        """
        if path is not None:
            self._path = self._create(path)
        return self

    @abstractmethod
    def __init__(self, path: Path):
        """ Construct ``self``.

        Overrides should call ``super(Stored).__init__(path)`` as a matter of priority.
        Then they should read ``self`` from ``self._path`` or write ``self`` in ``self._path``.

        Args:
            path: The Path ``self`` is stored in. A ``cls.ext`` suffix is appended.
        """
        self._path = self._create(path)

    @classmethod
    def _create(cls, path: Path) -> Self | Path:
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
        Returns: ``path.with_suffix(cls.ext)``.
        Raises:
            FileExistsError: If attempting to overwrite a file with a folder.
        """
        return cls._create(path)

    @classmethod
    @abstractmethod
    def copy(cls, src: Path, dst: Path) -> Self | Path:
        """ Copy ``src`` to ``dst``, overwriting only files in common.

        Overrides should copy an instance of ``cls`` called ``src`` to ``Store.create(dst)``, and return the copy.

        Args:
            src: The source Path, which must be a folder or a file.
            dst: The destination Path, which may or may not exist.
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
            path: The Path to delete.
        Returns: ``path``, which no longer exists.
        """
        path = Path(path).with_suffix(cls.ext)
        if path.is_dir():
            rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        return path


class DataTable(Store):
    """ Concrete class encapsulating a pd.DataFrame backed by a .csv file.

    Attributes:
        Data: Class attribute aliasing data Types which a DataTable accepts.
        ext: Class attribute specifying the file extension of DataTable instances.
        read_options: Instance attribute Options passed to `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_.
            Defaults to ``{'index_col': 0}``.
        write_options: Instance attribute Options passed to `DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.
            Defaults to ``{}``.
    """

    Data = Union[PD.DataFrame, NP.Matrix, TF.Matrix]
    ext: str = '.csv'

    read_options: Options
    write_options: Options

    @property
    def pd(self) -> PD.DataFrame:
        return self._pd

    @property
    def np(self) -> NP.Matrix:
        return self.pd.to_numpy()

    @property
    def tf(self) -> TF.Matrix:
        return tf.convert_to_tensor(self.np)

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
            raise IndexError(f'{repr(self)} has shape {self._pd.shape} 'f' which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            data = np.diag(np.diagonal(data))
        return self(data)

    def __call__(self, data: Self | Data | None, path: Store.Path | None = None, **options: Any) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            data: The data updates.
            path: Optionally, an update to ``self.path``, overwritten if existing. A .csv extension is appended.
            **options: Options to update ``self.write_options``.
        Returns: ``self``.
        """
        super().__call__(path)
        if isinstance(data, DataTable):
            self._pd = data.pd.copy()
        elif isinstance(data, pd.DataFrame):
            self._pd = data.copy()
        elif isinstance(data, NP.Matrix):
            self._pd.iloc[:, :] = data
        elif isinstance(data, TF.Matrix):
            self._pd.iloc[:, :] = data.numpy()
        self.write_options |= options
        self._pd.to_csv(self._path, **self.write_options)
        return self

    def __init__(self, path: Store.Path, data: PD.DataFrame | None = None, **options: Any):
        """ Construct ``self`` from a .csv file or pd.DataFrame.

        Args:
            path: The Path (file) to store ``self``. A .csv extension is appended.
            data: The data to store. If ``None``, ``data`` is read from ``path``, otherwise ``data`` is stored in ``path`` (which is overwritten if existing).
            **options: Updates ``self.read_options`` if ``data is None``, otherwise updates ``self.write_options``.
        """
        super().__init__(path)
        self.read_options = {'index_col': 0}
        self.write_options = {}
        if data is None:
            self.read_options |= options
            self(pd.read_csv(self.path, **self.read_options))
        else:
            self.write_options |= options
            self(data, **self.write_options)

    @classmethod
    def create(cls, path: Store.Path, data: Self | Data | None = None,
               index: PD.Index | NP.ArrayLike = None, columns: PD.Index | NP.ArrayLike = None, dtype: NP.DType | None = None, copy: bool | None = None,
               **options) -> Self:
        """ Create a DataTable at ``path``, overwriting.

        Args:
            path: The Path to store this DataTable, overwritten if existing. A .csv extension is appended.
            data: The data to store. If ``None``, a pd.DataFrame is read from csv.
                See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            index: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            columns: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            dtype: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            copy: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            **options: Options passed to `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_
                or `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.
        Returns: The DataTable created.
        """
        return cls(path,
                   pd.DataFrame(data.pd if isinstance(data, DataTable) else data, index, columns, dtype, copy),
                   **options)

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source DataTable.
            dst: The destination Path, overwritten if existing. A .csv extension is appended.
        Returns: The DataTable now stored at ``dst.with_suffix('.csv')``.
        """
        return cls(dst, src.pd, **src.write_options)


class Meta(Store):
    """ Concrete class encapsulating metadata stored in a .json file.

    Attributes:
        Data: Class attribute alias for the Type holding metadata .
        ext: Class attribute specifying the file extension of Meta instances.
    """

    Data = dict[str, Any]
    ext: str = '.json'

    @property
    def data(self) -> Data:
        return self._data

    def __call__(self, path: Store.Path | None = None, **data: Any) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            path: Optionally, an update to ``self.path``, overwritten if existing. A .json extension is appended.
            **data: Data to update ``self.data``.
        Returns: ``self``.
        """
        super().__call__(path)
        self._data |= data
        with open(self._path, mode='w') as file:
            dump(self._data, file, indent=4)
        return self

    def __init__(self, path: Store.Path, **data: Any):
        """ Construct ``self`` from a .json file or dict.

        Args:
            path: The Path (file) to store ``self``. A .json extension is appended.
            **data: The metadata to store. If absent, ``self.data`` is read from ``path``, otherwise ``self.data=data`` is stored in ``path``
                (which is overwritten if existing).
        """
        super().__init__(path)
        if data == {}:
            with open(self._path, mode='r') as file:
                self._data = load(file)
        else:
            self._data = data
            self(path)

    @classmethod
    def create(cls, path: Store.Path, **data: Any):
        """ Create a Meta at ``path``, overwriting.

        Args:
            path: The Path (file) to store ``self``, overwritten if existing. A .json extension is appended.
            **data: The metadata to store.
        Returns: The Meta created.
        """
        return cls(path, **({'NotImplemented': 'in call to Meta.create()'} if data == {} else data) )

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source Meta.
            dst: The destination Path, overwritten if existing. A .json extension is appended.
        Returns: The Meta now stored at ``dst.with_suffix('.json')``.
        """
        return cls(dst, **src.data)


class DataBase(Store):
    """ A NamedTuple of DataTables in a folder. Base class for any Model DataBase. 
        *This class is abstract and must be subclassed*. Usage will raise AssertionErrors*.

    DataBase subclasses must be implemented according to the template (copy and paste it)::

        class MyDataBase(DataBase):
            class Tables(NamedTuple):
                table_names[i]: DataBase.Table = pd.DataFrame(table_defaults[i].pd)
            read_options: dict[str, Options] = {table_names[i]: DataTable.read_options[i]}
            write_options: dict[str, Options] = {table_names[i]: DataTable.write_options[i]}

    Attributes:
        Table: Class attribute aliasing `` DataTable | DataTable.Data``. Do not override.
        read_options: Class attribute ``dict`` of the form ``{table_names[i]: DataTable.read_options[i]}.
            Override as necessary for bespoke ``DataTable.read_options``.
        write_options: Class attribute ``dict`` of the form {table_names[i]: DataTable.write_options[i]}. 
            Override as necessary for bespoke ``DataTable.write_options``.
    Raises:
        AssertionError: Whenever used in a ``__debug__=True`` environment (i.e. whenever ``python`` is invoked without the ``-O`` or ``--O`` flags).
            This class is abstract, and must be subclassed.
    """

    Table = DataTable | DataTable.Data

    class Tables(NamedTuple):
        """ Must be overridden by a subclass of NamedTuple.

        Every attribute (table) must be of Type Database.Table and possess a default pd.DataFrame.
        No attribute (table) may be named ``path``, as this is reserved for the Path to the DataBase.

        Attributes:
            NotImplemented: A PD.DataFrame containing the words "Not Implemented".
        """
        NotImplemented: DataBase.Table = pd.DataFrame(data=(('Attribute type should be DataBase.Table in any implementation.',),))

    read_options: dict[str, Options] = {'Default for any DataTable is': {'index_col': 0}}
    write_options: dict[str, Options] = {'Default for any DataTable is': {}}

    @classmethod
    def DataNotImplementedError(cls) -> str:
        return (f'You must implement {cls.__qualname__}.Tables as a subclass of NamedTuple. Your code (minus docstrings) should read as follows.\n' +
                f'class {cls.__name__}(DataBase):\n    class Tables(NamedTuple):\n        table_names[i]: DataBase.Data = pd.DataFrame(table_defaults[i].pd)')

    @property
    def tables(self) -> Tables:
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        return self._tables

    def tables_as_dict(self) -> Dict[str, Table]:
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        return self._tables._asdict()

    def __call__(self, **tables_and_path: Table | Store.Path) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            path: Optionally, an update to ``self.path``, overwritten if existing.
            **tables_and_path: Data to update ``self.tables``, in the form ``table_name[i]=data[i]``, and/or ``self.path`` in the form ``path=[path]``.
        Returns: ``self``.
        """
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        all_tables = self.tables_as_dict()
        if path := tables_and_path.pop('path', None) is not None:
            super().__call__(path)
            all_tables = {name:
                              DataTable.create(path=self._path / name, data=tables_and_path.get(name, data), **self.write_options.get(name, {}))
                          for name, data in all_tables.items()}
        else:
            for name, data in tables_and_path.items():
                all_tables[name](data, **self.write_options.get(name, {}))
        self._tables = self.Tables(**all_tables)
        return self

    def __init__(self, path: Store.Path, **tables: Table):
        """ Read the DataBase in ``path``, updating ``**tables``.

        Reading is lazy: If a table appears in ``**tables`` it is not read.

        Args:
            path: The Path to read from, and update in.
            **tables: Data to update ``self.tables``, in the form ``table_name[i]=data[i]``.
        Raises:
            FileNotFoundError: If reading is performed and ``path`` is missing any member of ``self.Tables`` not included in ``**tables``.
        """
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        super().__init__(path)
        try:
            self._tables = self.Tables(**{name:
                                              DataTable(path=path / name, data=tables[name], **self.write_options.get(name, {})) if name in tables
                                              else DataTable(path / name, **self.read_options.get(name, {}))
                                          for name in self.table_names()})
        except FileNotFoundError as error:
            print(f'DataBase "{self}" is trying to read a non-existent DataTable. '
                                    f'Did your script mean to call {type(self).__qualname__}.create("{str(self)}") '
                                    f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error

    @classmethod    # Class Property
    def table_names(cls) -> Tuple[str, ...]:
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls.Tables._fields

    @classmethod    # Class Property
    def table_defaults(cls) -> Dict[str, PD.DataFrame]:
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls.Tables._field_defaults

    @classmethod
    def create(cls, path: Store.Path, **tables: Table) -> Self:
        """ Create a DataBase in ``path``.

        Args:
            path: The folder to store the DataBase in. Need not exist, existing DataTables will be overwritten if it does.
            **tables: Data to update ``self.table_defaults``, in the form ``table_name[i]=data[i]``.
        Returns: The DataBase created.
        """
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls(path, **(cls.table_defaults() | tables))

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source DataBase.
            dst: The destination Path, which may or may not exist.
        Returns: The DataBase now stored at ``dst``.
        """
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls(dst, **src.tables_as_dict())

    @classmethod
    def delete(cls, path: Store.Path) -> Path:
        """ Delete ``cls.table_names[i].with_suffix(.csv)`` from ``path``, retaining ``path`` and any files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: Path to the DataBase to delete.
        Returns: ``path``, which still exists.
        """
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        path = Path(path)
        for table_name in cls.table_names():
            DataTable.delete(path / table_name)
        return path


class Model(Store):
    """ A DataBase with Meta. Base class for any model.

    Attributes:
        defaultMetaData: Class attribute. Must be overridden.
    """

    defaultMetaData: Meta.Data = {}

    class DataBase(DataBase):
        """ Must be overridden by a subclass of DataBase."""
        class Tables(NamedTuple):
            """ Must be overridden by a subclass of NamedTuple.

            Attributes:
                NotImplemented: A DataBase.Table of
            """
            NotImplemented: DataBase.Table = pd.DataFrame(data=(('NotImplemented',),))

        read_options: dict[str, Options] = {'Default for any DataTable is': {'index_col': 0}}
        write_options: dict[str, Options] = {'Default for any DataTable is': {}}

    @property
    def data(self) -> DataBase:
        return self._database

    @property
    def meta(self) -> Meta:
        return self._meta

    @abstractmethod
    def __call__(self, **options: Any) -> Self:
        """ Optimize the Model.

        Args:
            **options: Optimization options.

        Returns: ``self``
        """

    def __init__(self, path: Store.Path, **data: DataBase.Table):
        """ Read the Model in ``path``.

        Overrides must call ``super(Model).__init__(path, **data)`` as a matter of priority.

        Args:
            path: The Path to read from.
            **data: Data to update those read, in the form ``table_name[i]=data[i]``.
        Raises:
            FileNotFoundError: If ``path`` is missing ``self.meta`` or any member of ``self.DataBase.Tables`` not included in ``**data``.
        """
        super().__init__(path)
        try:
            self._meta = Meta(self._meta_in(path))
            self._database = self.DataBase(path, **data)
        except FileNotFoundError as error:
            print(f'Model "{self}" is trying to read a non-existent file. '
                                    f'Did your script mean to call {type(self).__qualname__}.create("{str(self)}") '
                                    f'instead of {type(self).__qualname__}("{str(self)}")?')
            raise error

    @classmethod
    def create(cls, path: Store.Path, meta: Meta.Data | None = None, **data: DataBase.Table) -> Self:
        """ Create a Model in ``path``.

        Args:
            path: The folder to store the Model in. Need not exist, existing DataTables will be overwritten if it does.
            meta: Optional metadata to update the default.
            **data: Data to update ``cls.DataBase.table_defaults``, in the form ``table_name[i]=data[i]``.
        Returns: The Model created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMetaData | ({} if meta is None else meta)))
        return cls(path, **(cls.DataBase.table_defaults() | data))

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source Model.
            dst: The destination Path, which may or may not exist.
        Returns: The Model now stored at ``dst``.
        """
        return cls.create(dst, src.meta.data, **src.data.tables_as_dict())

    @classmethod
    def delete(cls, path: Store.Path) -> Path:
        """ Delete all Model files from ``path``, retaining ``path`` and any other files it contains.

        If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

        Args:
            path: Path to the Model to delete.
        Returns: ``path``, which still exists.
        """
        Meta.delete(cls._meta_in(path))
        cls.DataBase.delete(path)
        return path

    @staticmethod
    def _meta_in(path: Store.Path) -> Path:
        return Path(path) / 'meta'


class ToyDataBase(DataBase):
    """ Play with me."""
    class Tables(NamedTuple):
        """
        """
        data: DataBase.Table = pd.DataFrame(data=[[0, 0, 0]],
                                            columns=pd.MultiIndex.from_tuples((('Category', 'int'), ('Input', 'float'), ('Output', 'float'))))

    read_options: dict[str, Options] = {'data': {'header': [0, 1]}}
    write_options: dict[str, Options] = {'data': {'index': True}}


class ToyModel(Model):
    """ A DataBase with Meta. Base class for any model.

    Attributes:
        defaultMetaData: Class attribute. Must be overridden.
    """

    defaultMetaData: Meta.Data = {}

    class DataBase(DataBase):
        """ Must be overridden by a subclass of DataBase."""
        class Tables(NamedTuple):
            """ Must be overridden by a subclass of NamedTuple.

            Attributes:
                NotImplemented: A DataBase.Table of
            """
            data: DataBase.Table = pd.DataFrame(data=[[0, 0, 0]],
                                                columns=pd.MultiIndex.from_tuples((('Category', 'int'), ('Input', 'float'), ('Output', 'float'))))

        read_options: dict[str, Options] = {'data': {'header': [0, 1]}}
        write_options: dict[str, Options] = {'data': {}}

    def __call__(self, **options: Any) -> Self:
        self._database(**options)
        return self

