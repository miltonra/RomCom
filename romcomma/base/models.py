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

from romcomma.base.definitions import *
from shutil import copyfile, copytree, rmtree
from json import load, dump


class Store(ABC):
    """ Base class for any stored class. Users are not expected to subclass this class directly.

    Attributes:
        ext: Class attribute specifying the file extension terminating ``self.path``. Override if and only if the derived class must be stored in a file.
            Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.
        Path: Class attribute aliasing Types used to specify the ``path`` to this Store. Do not override.
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
            self._path = self.create(path)
        return self

    @abstractmethod
    def __init__(self, path: Path):
        """ Construct ``self``.

        Overrides should call ``super(Stored).__init__(path)`` as a matter of priority.
        Then they should read ``self`` from ``self._path`` or write ``self`` in ``self._path``.

        Args:
            path: The Path ``self`` is stored in. A ``cls.ext`` suffix is appended.
        """
        self._path = self.create(path)

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
        path = Path(path).with_suffix(cls.ext)
        if cls.ext == '':
            path.mkdir(mode=0o777, parents=True, exist_ok=True)
        else:
            path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        return path

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
            path: The path to delete.
        Returns: ``path``, which no longer exists.
        """
        path = Path(path)
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
        read_options: Instance attribute Kwargs passed to `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_.
            Defaults to ``{'index_col': 0}``.
        write_options: Instance attribute Kwargs passed to `DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.
            Defaults to ``{}``.
    """

    Data = Union[PD.DataFrame, NP.Matrix, TF.Matrix]
    ext: str = '.csv'

    read_options: Kwargs
    write_options: Kwargs

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

    def __call__(self, data: Self | Data | None, path: Store.Path | None = None, **kwargs: Any) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            data: The data updates.
            path: Optionally, an update to ``self.path``, overwritten if existing. A .csv extension is appended.
            **kwargs: Kwargs to update ``self.write_options``.
        Returns: ``self``.
        """
        super(Store).__call__(path)
        if isinstance(data, DataTable):
            self._pd = data.pd.copy()
        elif isinstance(data, pd.DataFrame):
            self._pd = data.copy()
        elif isinstance(data, NP.Matrix):
            self._pd.iloc[:, :] = data
        elif isinstance(data, TF.Matrix):
            self._pd.iloc[:, :] = data.numpy()
        self.write_options |= kwargs
        self._pd.to_csv(self._path, **self.write_options)
        return self

    def __init__(self, path: Store.Path, data: PD.DataFrame | None = None, **kwargs: Any):
        """ Construct ``self`` from a .csv file or pd.DataFrame.

        Args:
            path: The Path (file) to store ``self``. A .csv extension is appended.
            data: The data to store. If ``None``, ``data`` is read from ``path``, otherwise ``data`` is stored in ``path`` (which is overwritten if existing).
            **kwargs: Updates ``self.read_options`` if ``data is None``, otherwise updates ``self.write_options``.
        """
        super(Store).__init__(path)
        if data is None:
            self.read_options = {'index_col': 0} | kwargs
            self(pd.read_csv(self.path, **self.read_options))
        else:
            self.write_options = {} | kwargs
            self(data, **self.write_options)

    @classmethod
    def create(cls, path: Store.Path, data: Self | Data | None = None,
               index: PD.Index | NP.ArrayLike = None, columns: PD.Index | NP.ArrayLike = None, dtype: NP.DType | None = None, copy: bool | None = None,
               **kwargs) -> Self:
        """ Create a DataTable at ``path``, overwriting.

        Args:
            path: The Path to store this DataTable, overwritten if existing. A .csv extension is appended.
            data: The data to store. If ``None``, a pd.DataFrame is read from csv.
                See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            index: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            columns: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            dtype: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            copy: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            **kwargs: Kwargs passed straight to `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_
                or `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.
        Returns: The DataTable created.
        """
        return cls(path,
                   pd.DataFrame(data.pd if isinstance(data, DataTable) else data, index, columns, dtype, copy),
                   **kwargs)

    @classmethod
    def copy(cls, src: Self, dst: Path) -> Self:
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
        super(Store).__call__(path)
        self._data |= data
        with open(self._path, mode='w') as file:
            dump(self._data, file, indent=4)
        return self

    def __init__(self, path: Store.Path, **data: Any):
        """ Construct ``self`` from a .json file or dict.

        Args:
            path: The Path (file) to store ``self``. A .json extension is appended.
            **data: The metadata to store. If absent, ``self.data`` is read from ``path``, otherwise ``self.data = data`` is stored in ``path``
                (which is overwritten if existing).
        """
        super(Store).__init__(path)
        if data == {}:
            with open(self._path, mode='r') as file:
                self._data = load(file)
        else:
            self(**data)

    @classmethod
    def create(cls, path: Store.Path, **data: Any):
        """ Create a Meta at ``path``, overwriting.

        Args:
            path: The Path (file) to store ``self``, overwritten if existing. A .json extension is appended.
            **data: The metadata to store.
        Returns: The Meta created.
        """
        return cls(path, **data)

    @classmethod
    def copy(cls, src: Self, dst: Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source Meta.
            dst: The destination Path, overwritten if existing. A .json extension is appended.

        Returns: The Meta now stored at ``dst.with_suffix('.json')``.
        """
        return cls(dst, **src.data)


class DataBase(Store):
    """ A NamedTuple of DataTables in a folder. Base class for any Model DataBase, *this class is abstract and will raise AssertionErrors*.

    DataBase subclasses must be implemented according to the template::

        class MyDataBase(DataBase):
            class Tables(NamedTuple):
                table_names[i]: DataBase.Table = pd.DataFrame(table_defaults[i].pd)
            read_options: dict[str, type(DataTable.read_options)] = {table_names[i]: DataTable.read_options[i]}
            write_options: dict[str, type(DataTable.write_options)] = {table_names[i]: DataTable.write_options[i]}

    Attributes:
        Data: Class attribute aliasing `` DataTable | DataTable.Data``. Do not override.
        read_options: Class attribute ``dict`` of the form ``{table_names[i]: DataTable.read_options[i]}.
            Override as necessary for bespoke ``DataTable.read_options``.
        write_options: Class attribute ``dict`` of the form {table_names[i]: DataTable.write_options[i]}. Override as necessary.
            Override as necessary for bespoke ``DataTable.write_options``.
    Raises:
        AssertionError: Whenever used in a ``__debug__ = True`` environment (i.e. whenever ``python`` is invoked without the ``-O`` or ``--O`` flags).
    """

    Data = DataTable | DataTable.Data

    class Tables(NamedTuple):
        """ Must be overridden by a subclass of NamedTuple.

        Every Attribute (table) must be of type DataTable.Data and possess a default pd.Dataframe.

        Attributes:
            NotImplemented: A DataTable.Data containing the words "Not Implemented".
        """
        NotImplemented: DataTable | DataTable.Data = pd.DataFrame(data=(('DataBase.Tables must be overridden by a subclass of NamedTuple.',),))

    read_options: dict[str, Kwargs] = {}
    write_options: dict[str, Kwargs] = {}

    @classmethod
    def DataNotImplementedError(cls) -> str:
        return (f'You must implement {cls.__qualname__}.Tables as a subclass of NamedTuple. Your code (minus docstrings) should read as follows.\n' +
                f'class {cls.__name__}(DataBase):\n    class Tables(NamedTuple):\n        table_names[i]: DataBase.Data = pd.DataFrame(table_defaults[i].pd)')

    @property
    def tables(self) -> Data:
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        return self._tables

    def tables_as_dict(self) -> Dict[str, Any]:
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        return self._tables._asdict()

    def __call__(self, path: Path | None, **kwargs: DataTable | DataTable.Data) -> Self:
        """ Update this DataBase.

        Args:
            path: The new path to store this class.
            **kwargs: Data to update the existing tables, in the form table_name=data.

        Returns: ``self``, once path and data have been updated and stored.
        """
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        tables = self.tables_as_dict()
        if path is not None:
            super(Store).__call__(path)
            create = {name: DataTable.create(path=(self._path / name).with_suffix('.csv'), data=data, **self.write_options.get(name, {}))
                      for name, data in tables.items()}
            self._tables = self.Tables(**create)
        for name, data in kwargs.items():
            tables[name](data, **self.write_options.get(name, {}))
        return self

    def __init__(self, path: Store.Path | None, **kwargs: DataTable | DataTable.Data):
        """ Read the DataBase in ``path``.

        If ``**kwargs`` updates all tables no reading is performed.

        Args:
            path: A folder storing a DataBase. Must exist and contain all DataTables in ``cls.DataTables``.
            **kwargs: Data to update those read, in the form table_name=data.
        Returns: The DataBase stored in ``path``.
        Raises:
            FileNotFoundError: If reading is performed and ``path`` is missing any member of ``self.Data``, even one updated by ``**kwargs``.
        """
        assert self.Tables is not DataBase.Tables, type(self).DataNotImplementedError()
        if set(kwargs.keys()) == set(self.table_names()):
            self._tables = self.Tables()
        else:
            self._tables = self.Tables(**({name: DataTable((path / name).with_suffix('.csv'), **self.read_options.get(name, {}))
                                           for name in self.table_names()} | kwargs))
        self(path, **kwargs)

    @classmethod    # Class Property
    def table_names(cls) -> Tuple[str, ...]:
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls.Tables._fields

    @classmethod    # Class Property
    def table_defaults(cls) -> Dict[str, PD.DataFrame]:
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls.Tables._field_defaults

    @classmethod
    def create(cls, path: Store.Path | None, **kwargs: DataTable | DataTable.Data) -> Self:
        """ Create a DataBase in ``path``.

        Args:
            path: The folder to store the DataBase in. Need not exist, existing DataTables will be overwritten if it does.
            **kwargs: Data to update the defaults, in the form table_name=data.
        Returns: The DataBase created.
        """
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls(Store.create(path), **(cls.table_defaults() | kwargs))

    @classmethod
    def copy(cls, src: Self, dst: Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting any files in common.

        Args:
            src: The source DataBase.
            dst: The destination Path, which may or may not exist.

        Returns: The DataBase now stored at ``dst``.
        """
        assert cls.Tables is not DataBase.Tables, cls.DataNotImplementedError()
        return cls(Store.create(dst), **src.tables_as_dict())


class Model(ABC):
    """ A DataBase, metadata and a calibrate method. Base class for any model."""

    Meta = Dict[str, Any]

    _meta: Meta = {"Must be overridden.": 'With the default meta data for the derived class.'}

    class DataBase(DataBase):
        """ This is a placeholder which must be overridden in any implementation."""

    @property
    def database(self) -> DataBase:
        return self._database

    @property
    def meta(self) -> Meta:
        return self._meta

    def read_meta(self, **kwargs: Any) -> Meta:
        """ Read ``self.meta`` from csv.

        Args:
            **kwargs: a dict of ``self.meta`` updates, expressed as key=value.
        Returns: ``self.meta``.
        """
        with open(self._meta_json, mode='r') as file:
            self._meta = load(file) | kwargs
        return self._meta

    def write_meta(self, **kwargs: Any) -> Meta:
        """ Write ``self.meta`` to csv.

        Args:
            **kwargs: a dict of ``self.meta`` updates, expressed as key=value.
        Returns: ``self.meta``.
        """
        self._meta |= kwargs
        with open(self._meta_json, mode='w') as file:
            dump(self._meta, file, indent=8)
        return self._meta

    @abstractmethod
    def calibrate(self, method: str, **kwargs) -> Meta:
        if method != 'Not Implemented.':
            raise NotImplementedError('base.calibrate() must never be called.')
        else:   # Template code follows
            meta = self._meta | kwargs
            meta = (meta if meta is not None
                       else self.read_meta() if self._meta_json.exists() else self.META)
            meta.pop('result', default=None)
            meta = {**meta, 'result': 'OPTIMIZE HERE !!!'}
            self.write_meta(meta)
            self._database = self._database.replace('WITH OPTIMAL PARAMETERS!!!').write(self.folder)   # Remember to write optimization results.
        return meta

    # def copy(self, dst_folder: Path) -> Self:
    #     """ Returns a copy of this DataBase at ``dst_folder``, overwriting any DataBase tables at the destination."""
    #     return type(self)(dst_folder,(dst_folder, **self.asdict())

    def __repr__(self) -> str:
        """ Returns the folder path."""
        return str(self._folder)

    def __str__(self) -> str:
        """ Returns the folder name."""
        return self._folder.name

    @abstractmethod
    def __init__(self, folder: Path, is_read: bool = False, **kwargs: DataTable.Data):
        """ Model constructor, to be called by all subclasses as a matter of priority.

        Args:
            folder: The model file location.
            is_read: Tries to read data and meta from folder if True, else uses defaults.
            **kwargs: The model.database.tables=values to replace the default/read values in the form ``table=value``.
                Plus a ``dict`` of meta updates communicated as ``meta=dict``.
        """
        if is_read:
            self._folder = self.DataBase.create(folder)
            self._database = self.DataBase.read(self._folder)
        else:
            self._folder = self.DataBase.create(self.DataBase.delete(folder))
            self._database = self.DataBase(self._folder)
        self._meta_json = self._folder / "meta.json"
        self._meta = self.read_meta() if self._meta_json.exists() else self.META
        self._meta |= kwargs.pop('meta', {})
        self._database = self._database.replace(**kwargs)
        self._implementation = None


class ToyStore(Store):
    def __init__(self, folder: Path):
        self._path = folder


class Toy(DataBase):

    class Data(NamedTuple):

        data: DataTable.Data = pd.DataFrame([[0, 0, 0]],
                                            columns=pd.MultiIndex.from_tuples((('Category', 'int'), ('Input', 'float'), ('Output', 'float'))))
