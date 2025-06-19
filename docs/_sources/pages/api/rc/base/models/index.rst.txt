rc.base.models
==============

.. py:module:: rc.base.models

.. autoapi-nested-parse::

   Abstract and concrete base classes for RomCom Models.



Properties
-----------

.. autoapisummary::

   rc.base.models.MetaData
   rc.base.models.Matrix


Classes
-------

.. autoapisummary::

   rc.base.models.Store
   rc.base.models.Meta
   rc.base.models.Table
   rc.base.models.DataBase


Module Contents
---------------

.. py:data:: MetaData

   Type for passing metadata as ``**kwargs``.

.. py:data:: Matrix

   Types which a DataBase Table accepts.

.. py:class:: Store(path)

   Bases: :py:obj:`rc.base.definitions.ABC`

   .. autoapi-inheritance-diagram:: rc.base.models.Store
      :parts: 1


   Base class for any stored class. Users are not expected to subclass this class directly.


   .. py:attribute:: Path

      Class attribute aliasing Types used to specify the ``path`` to a Store. Do not override.


   .. py:attribute:: ext
      :type:  str
      :value: ''


      Class attribute specifying the file extension terminating ``self.path``.
      Override if and only if the derived class must be stored in a file.
      Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.


   .. py:property:: path
      :type: rc.base.definitions.Path


      The ``Path`` to this ``Store``, without ``cls.ext``.
      File extension is internal, meaning ``self._path = self._path + cls.ext``.


   .. py:method:: __repr__()

      The ``Path`` to this ``Store``.

      :meta public:



   .. py:method:: __str__()

      The ``Path`` to this ``Store``, abbreviated.

      :meta public:



   .. py:method:: __call__(**data)
      :abstractmethod:


      Update and store ``self``.

      :param \*\*data: Data to update.

      Returns: ``self``.



   .. py:method:: extAppend(path)
      :classmethod:


      Append ``cls.ext`` to ``path.name``.

      :param path: The path to append ``cls.ext`` to.

      Returns: ``Path(path)`` with ``cls.ext`` appended.



   .. py:method:: mkdir(path)
      :classmethod:


      Create ``path.parent``, with a subfolder ``path`` if ``cls.ext == ''``.

      :param path: The folder to create, or a child file of the folder to create.

      Returns: ``Path(path)`` with ``cls.ext`` appended.



   .. py:method:: create(path)
      :classmethod:

      :abstractmethod:


      Create a folder (and its parents) if it doesn't already exist.

      Overrides should create and return an instance of ``cls``.

      :param path: Where to create the folder. If ``cls.ext != ''``, the parent folder of ``path`` is created.

      :returns: ``path`` with extension ``f'.{cls.ext}'``.

      :raises FileExistsError: If attempting to overwrite a file with a folder.



   .. py:method:: copy(src, dst)
      :classmethod:

      :abstractmethod:


      Copy ``src`` to ``dst``, overwriting only files in common.

      Overrides should copy an instance of ``cls`` called ``src`` to ``Store.create(dst)``,
      and return the copy.

      :param src: The source ``Path``, which must be a folder or a file.
      :param dst: The destination ``Path``, which may or may not exist.

      Returns: ``dst``.

      :raises FileNotFoundError: If ``src`` does not exist.
      :raises FileExistsError: If attempting to overwrite a file with a folder.



   .. py:method:: delete(path)
      :classmethod:


      Delete any file or folder at ``path``.

      :param path: The ``Path`` to delete.

      Returns: ``path``, which no longer exists.



.. py:class:: Meta(path, **data)

   Bases: :py:obj:`Store`, :py:obj:`dict`

   .. autoapi-inheritance-diagram:: rc.base.models.Meta
      :parts: 1


   Concrete class encapsulating metadata stored in a ``.json`` file.


   .. py:attribute:: ext
      :type:  str
      :value: '.json'


      Class attribute specifying the file extension terminating ``self.path``.
      Override if and only if the derived class must be stored in a file.
      Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.


   .. py:method:: __call__(**data)

      Update and store ``self``, overwriting.

      :param \*\*data: Data to update ``self.data``.

      Returns: ``self``.



   .. py:method:: __setitem__(key, value)

      Indexer sets the ``value`` indexed by ``key``.



   .. py:method:: create(path, **data)
      :classmethod:


      Create a ``Meta`` at ``path``, overwriting.

      :param path: The ``Path`` (file) to store ``self``, overwritten if existing.
                   A ``.json`` extension is automatically appended.
      :param \*\*data: The ``MetaData`` to store.

      Returns: The ``Meta`` created.



   .. py:method:: copy(src, dst)
      :classmethod:


      Copy ``src`` to ``dst``, overwriting.

      :param src: The source ``Meta``.
      :param dst: The destination ``Path``, overwritten if existing.
                  A ``.json`` extension is automatically appended.

      Returns: The ``Meta`` now stored at ``dst.json``.



.. py:class:: Table(path, data = None, **options)

   Bases: :py:obj:`Store`

   .. autoapi-inheritance-diagram:: rc.base.models.Table
      :parts: 1


   Concrete class encapsulating a ``pd.DataFrame`` backed by a ``.csv`` file.

   This class may be usefully overridden to provide bespoke read and write options for
   file operations. Subclasses should follow the template (copy and paste it)::

       class MyTable(Table):

       class Options(NamedTuple):

           read: MetaData =  {'index_col': 0}  #: Read options passed to ``pd.read_csv``.
           write: MetaData =  {}   #: Write options passed to ``pd.DataFrame.to_csv``.

           @classmethod
           def default(cls) -> MetaData:
               """ Returns the default Options as ``cls.read | cls.write``."""
               return cls._field_defaults['read'] | cls._field_defaults['write']


   .. py:attribute:: ext
      :type:  str
      :value: '.csv'


      Class attribute specifying the file extension terminating ``self.path``.
      Override if and only if the derived class must be stored in a file.
      Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.


   .. py:attribute:: writeOptions
      :type:  list[str]
      :value: ['sep', 'na_rep', 'float_format']


      Class attribute listing kwargs which will be interpreted as write options.
      All other kwargs are interpreted as read options.
      To specify a separator, use ``delimiter`` as read option and ``sep`` as write option.


   .. py:property:: options
      :type: MetaData


      A ``dict of options for file operations involving ``self``.
      Any option not in ``Table.writeOptions`` is stored in ``self.options.read`` and passed to ``pd.read_csv``.
      Any option in ``Table.writeOptions`` is stored in ``self.options.write``
      and passed to ``pd.DataFrame.to_csv``.
      The setter updates via logical or ``|=``, so existing values are retained unless explicitly updated.


   .. py:property:: pd
      :type: rc.base.definitions.Pd.DataFrame


      The ``Pd.DataFrame`` stored in ``self``.


   .. py:property:: np
      :type: rc.base.definitions.Np.Matrix


      The ``Np.Matrix`` stored in ``self``.


   .. py:property:: tc
      :type: rc.base.definitions.Tc.Matrix


      The ``TF.Matrix`` stored in ``self``.


   .. py:method:: broadcast_to(target_shape, is_diagonal = True)

      Broadcast ``self``.

      :param target_shape: The shape to broadcast to.
      :param is_diagonal: Whether to zero the off-diagonal elements of a square matrix.

      Returns: ``self``.

      :raises IndexError: If broadcasting is impossible.



   .. py:method:: __call__(data, **options)

      Update and store ``self``, overwriting.

      :param data: The data updates.
      :param \*\*options: Updates ``self.options``, before storing ``self``.

      Returns: ``self``.



   .. py:method:: create(path, data = None, index = None, columns = None, dtype = None, copy = None, **metadata)
      :classmethod:


      Create a ``Table`` at ``path``, overwriting.

      :param path: The ``Path`` to store this DataTable, overwritten if existing.
                   A ``.csv`` extension is automatically appended.
      :param data: The data to store. If ``None``, a ``Pd.DataFrame`` is read from ``.csv``.
                   See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
      :param index: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
      :param columns: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
      :param dtype: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
      :param copy: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
      :param \*\*metadata: MetaData passed to
                           `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_
                           or
                           `pd.DataFrame.to_csv`_.

      Returns: The ``DataTable`` created.

      .. _pd.DataFrame.to_csv: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html



   .. py:method:: copy(src, dst)
      :classmethod:


      Copy ``src`` to ``dst``, overwriting.

      :param src: The source ``DataTable``.
      :param dst: The destination ``Path``, overwritten if existing.
                  A ``.csv`` extension is automatically appended.

      Returns: The ``DataTable`` now stored at ``dst.csv``.



.. py:class:: DataBase(path, **tables)

   Bases: :py:obj:`Store`

   .. autoapi-inheritance-diagram:: rc.base.models.DataBase
      :parts: 1


   ``NamedTables(NamedTuple)`` in a folder alongside ``Meta``. Abstract base class for any model.

   ``DataBase`` subclasses must be implemented according to the template (copy and paste it)::

       class MyDataBase(DataBase):

           class NT(NamedTuple):

               names[i]: Table | Matrix | MetaData = pd.DataFrame(defaults[names[i]].pd)   #: Comment
               ...

               def __call__(self, name: str) -> Table | Matrix | MetaData:
                   """ Returns the Table named ``name``."""
                   return getattr(self, name)


           options: NamedTables[MetaData] = NamedTables(**{name: table.options for name, table in {}.items()})
           """ Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
           Override as necessary for bespoke ``Table.options``.
           Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
           the remainder populate ``self[i].options.read``."""

           defaultMetaData: MetaData = {'Tables': Tables.options._asdict()}


   .. py:class:: NamedTables

      Bases: :py:obj:`rc.base.definitions.NamedTuple`

      .. autoapi-inheritance-diagram:: rc.base.models.DataBase.NamedTables
         :parts: 1


      Must be overridden.


      .. py:method:: __call__(name)

         Returns the Table named ``name``.




   .. py:attribute:: options
      :type:  DataBase.NamedTables[MetaData]

      Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
      Override as necessary for bespoke ``Table.options``.
      Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
      the remainder populate ``self[i].options.read``.


   .. py:property:: nt
      :type: NamedTables


      The ``NamedTables`` currently in ``self``.


   .. py:property:: meta
      :type: Meta


      The ``Meta`` currently in ``self``.


   .. py:method:: __len__()

      Counts the ``Table`` s in ``self``.



   .. py:method:: __getitem__(name)

      Indexer returns the ``Table`` (s) named or sliced by ``name``.



   .. py:method:: __setitem__(name, tables)

      Indexer sets the ``Table`` (s) named or sliced by ``name``.



   .. py:method:: __call__(**tables)

      Update and store ``self``, overwriting.

      :param path: Optionally, an update to ``self.path``, overwritten if existing.
      :param \*\*tables: Updates to ``self`` in the form ``names[i]=Table[i], ...``.

      Returns: ``self``.



   .. py:method:: names()
      :classmethod:


      ``(names[i], ...)`` of table names for this ``Tables`` class.



   .. py:method:: defaults()
      :classmethod:


      ``{names[i]: Pd.DataFrame[i], ...}`` of default tables for this ``Tables`` class.



   .. py:method:: create(path, **tables_and_meta)
      :classmethod:


      Create a ``DataBase`` in ``path``.

      :param path: The folder to store the ``DataBase`` in. Need not exist,
                   any existing ``Tables`` will be overwritten if it does.
      :param \*\*tables_and_meta: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                                  and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

      Returns: The ``DataBase`` created.



   .. py:method:: copy(src, dst)
      :classmethod:


      Copy ``src`` to ``dst``, overwriting any files in common.

      :param src: The source ``DataBase``.
      :param dst: The destination ``Path``, which may or may not exist.

      Returns: The ``DataBase`` now stored in ``dst``.



   .. py:method:: delete(path)
      :classmethod:


      Delete all ``DataBase`` files in ``path``, retaining ``path`` and any other files it contains.

      If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

      :param path: ``Path`` to the ``DataBase`` to delete.

      Returns: ``path``, which still exists.



