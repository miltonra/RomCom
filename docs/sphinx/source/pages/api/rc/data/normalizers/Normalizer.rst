rc.data.normalizers.Normalizer
==============================


.. module:: rc.data.normalizers

.. toctree::
   :hidden:

   /pages/api/rc/data/normalizers/Normalizer.NamedTables
   /pages/api/rc/data/normalizers/Normalizer.CreateP
   /pages/api/rc/data/normalizers/Normalizer.create
   /pages/api/rc/data/normalizers/Normalizer.Tables
   /pages/api/rc/data/normalizers/Normalizer.defaultMeta
   /pages/api/rc/data/normalizers/Normalizer.IndexP
   /pages/api/rc/data/normalizers/Normalizer.EqualsP
   /pages/api/rc/data/normalizers/Normalizer.ReadP
   /pages/api/rc/data/normalizers/Normalizer.UpdateP
   /pages/api/rc/data/normalizers/Normalizer.DeleteP
   /pages/api/rc/data/normalizers/Normalizer.CopyP
   /pages/api/rc/data/normalizers/Normalizer.tables
   /pages/api/rc/data/normalizers/Normalizer.meta
   /pages/api/rc/data/normalizers/Normalizer.names
   /pages/api/rc/data/normalizers/Normalizer.defaults
   /pages/api/rc/data/normalizers/Normalizer.copy
   /pages/api/rc/data/normalizers/Normalizer.delete
   /pages/api/rc/data/normalizers/Normalizer.ext
   /pages/api/rc/data/normalizers/Normalizer.NameP
   /pages/api/rc/data/normalizers/Normalizer.path
   /pages/api/rc/data/normalizers/Normalizer.extAppend
   /pages/api/rc/data/normalizers/Normalizer.mkdir

.. py:class:: rc.data.normalizers.Normalizer(path, **tables)

   Bases: :py:obj:`IndependentNormalizer`

   .. autoapi-inheritance-diagram:: rc.data.normalizers.Normalizer
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

Protocols
----------

.. autoapisummary::

   rc.data.normalizers.Normalizer.CreateP
   rc.data.normalizers.Normalizer.IndexP
   rc.data.normalizers.Normalizer.EqualsP
   rc.data.normalizers.Normalizer.ReadP
   rc.data.normalizers.Normalizer.UpdateP
   rc.data.normalizers.Normalizer.DeleteP
   rc.data.normalizers.Normalizer.CopyP
   rc.data.normalizers.Normalizer.NameP


Attributes
----------

.. autoapisummary::

   rc.data.normalizers.Normalizer.NamedTables
   rc.data.normalizers.Normalizer.Tables
   rc.data.normalizers.Normalizer.defaultMeta
   rc.data.normalizers.Normalizer.ext


Properties
----------

.. autoapisummary::

   rc.data.normalizers.Normalizer.tables
   rc.data.normalizers.Normalizer.meta
   rc.data.normalizers.Normalizer.path


Methods
-------

.. autoapisummary::

   rc.data.normalizers.Normalizer.create
   rc.data.normalizers.Normalizer.names
   rc.data.normalizers.Normalizer.defaults
   rc.data.normalizers.Normalizer.copy
   rc.data.normalizers.Normalizer.delete
   rc.data.normalizers.Normalizer.extAppend
   rc.data.normalizers.Normalizer.mkdir


