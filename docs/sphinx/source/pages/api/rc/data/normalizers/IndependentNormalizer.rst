rc.data.normalizers.IndependentNormalizer
=========================================


.. module:: rc.data.normalizers

.. toctree::
   :hidden:

   /pages/api/rc/data/normalizers/IndependentNormalizer.NamedTables
   /pages/api/rc/data/normalizers/IndependentNormalizer.Tables
   /pages/api/rc/data/normalizers/IndependentNormalizer.defaultMeta
   /pages/api/rc/data/normalizers/IndependentNormalizer.CreateP
   /pages/api/rc/data/normalizers/IndependentNormalizer.create
   /pages/api/rc/data/normalizers/IndependentNormalizer.IndexP
   /pages/api/rc/data/normalizers/IndependentNormalizer.EqualsP
   /pages/api/rc/data/normalizers/IndependentNormalizer.ReadP
   /pages/api/rc/data/normalizers/IndependentNormalizer.UpdateP
   /pages/api/rc/data/normalizers/IndependentNormalizer.DeleteP
   /pages/api/rc/data/normalizers/IndependentNormalizer.CopyP
   /pages/api/rc/data/normalizers/IndependentNormalizer.tables
   /pages/api/rc/data/normalizers/IndependentNormalizer.meta
   /pages/api/rc/data/normalizers/IndependentNormalizer.names
   /pages/api/rc/data/normalizers/IndependentNormalizer.defaults
   /pages/api/rc/data/normalizers/IndependentNormalizer.copy
   /pages/api/rc/data/normalizers/IndependentNormalizer.delete
   /pages/api/rc/data/normalizers/IndependentNormalizer.ext
   /pages/api/rc/data/normalizers/IndependentNormalizer.NameP
   /pages/api/rc/data/normalizers/IndependentNormalizer.path
   /pages/api/rc/data/normalizers/IndependentNormalizer.extAppend
   /pages/api/rc/data/normalizers/IndependentNormalizer.mkdir

.. py:class:: rc.data.normalizers.IndependentNormalizer(path, **tables)

   Bases: :py:obj:`rc.data.distributions.DataBase`

   .. autoapi-inheritance-diagram:: rc.data.normalizers.IndependentNormalizer
      :parts: 1


   ``NamedTables(NamedTuple)`` in a folder alongside Meta. Abstract BaseClass for any model.

   DataBase SubClasses must be implemented according to the template (copy and paste it)::

       class MyDataBase(DataBase):

           class NamedTables(NamedTuple):

               names[i]: Table | Matrix = defaults[names[i]].pl
               """ Normally a ``Pl.DataFrame``. If no default is appropriate, use the Table Type"""
               ...

               def __call__(self, name: str) -> Table | Matrix | MetaData:
                   """ Returns the Table named ``name``."""
                   return getattr(self, name)

           Tables: NamedTables[type[Table], ...] = NamedTables()
           """ The ``NamedTables`` of Table Types, to communicate ``readOptions, writeOptions``."""

           defaultMeta: MetaData = {'Tables': {name: TableType.__name__ for name, TableType in Tables._asdict().items()}}
           """ Class default ``self.meta``."""

   Read the DataBase in ``path``.
   Reading is lazy: If ``names[i]`` occurs in ``**tables`` it's Table is not read, just updated.
   Overrides must call ``super(DataBase).__init__(path, **tables)`` as a matter of priority.

   :param path: The Path to read from.
   :param \*\*tables: Tables to update those read, in the form ``names[i]=tables[i], ...``.

   :raises FileNotFoundError: If ``path`` lacks ``self.meta`` or any member of
       ``self.Tables.names`` not mentioned in ``**tables``.

Classes
-------

.. autoapisummary::

   rc.data.normalizers.IndependentNormalizer.NamedTables


Protocols
----------

.. autoapisummary::

   rc.data.normalizers.IndependentNormalizer.CreateP
   rc.data.normalizers.IndependentNormalizer.IndexP
   rc.data.normalizers.IndependentNormalizer.EqualsP
   rc.data.normalizers.IndependentNormalizer.ReadP
   rc.data.normalizers.IndependentNormalizer.UpdateP
   rc.data.normalizers.IndependentNormalizer.DeleteP
   rc.data.normalizers.IndependentNormalizer.CopyP
   rc.data.normalizers.IndependentNormalizer.NameP


Attributes
----------

.. autoapisummary::

   rc.data.normalizers.IndependentNormalizer.Tables
   rc.data.normalizers.IndependentNormalizer.defaultMeta
   rc.data.normalizers.IndependentNormalizer.ext


Properties
----------

.. autoapisummary::

   rc.data.normalizers.IndependentNormalizer.tables
   rc.data.normalizers.IndependentNormalizer.meta
   rc.data.normalizers.IndependentNormalizer.path


Methods
-------

.. autoapisummary::

   rc.data.normalizers.IndependentNormalizer.create
   rc.data.normalizers.IndependentNormalizer.names
   rc.data.normalizers.IndependentNormalizer.defaults
   rc.data.normalizers.IndependentNormalizer.copy
   rc.data.normalizers.IndependentNormalizer.delete
   rc.data.normalizers.IndependentNormalizer.extAppend
   rc.data.normalizers.IndependentNormalizer.mkdir


