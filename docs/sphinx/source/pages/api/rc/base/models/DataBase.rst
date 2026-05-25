rc.base.models.DataBase
=======================


.. module:: rc.base.models

.. toctree::
   :hidden:

   /pages/api/rc/base/models/DataBase.NamedTables
   /pages/api/rc/base/models/DataBase.Tables
   /pages/api/rc/base/models/DataBase.defaultMeta
   /pages/api/rc/base/models/DataBase.IndexP
   /pages/api/rc/base/models/DataBase.EqualsP
   /pages/api/rc/base/models/DataBase.CreateP
   /pages/api/rc/base/models/DataBase.ReadP
   /pages/api/rc/base/models/DataBase.UpdateP
   /pages/api/rc/base/models/DataBase.DeleteP
   /pages/api/rc/base/models/DataBase.CopyP
   /pages/api/rc/base/models/DataBase.tables
   /pages/api/rc/base/models/DataBase.meta
   /pages/api/rc/base/models/DataBase.names
   /pages/api/rc/base/models/DataBase.defaults
   /pages/api/rc/base/models/DataBase.create
   /pages/api/rc/base/models/DataBase.copy
   /pages/api/rc/base/models/DataBase.delete
   /pages/api/rc/base/models/DataBase.ext
   /pages/api/rc/base/models/DataBase.NameP
   /pages/api/rc/base/models/DataBase.path
   /pages/api/rc/base/models/DataBase.extAppend
   /pages/api/rc/base/models/DataBase.mkdir

.. py:class:: rc.base.models.DataBase(path, **tables)

   Bases: :py:obj:`Store`

   .. autoapi-inheritance-diagram:: rc.base.models.DataBase
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

   rc.base.models.DataBase.NamedTables


Protocols
----------

.. autoapisummary::

   rc.base.models.DataBase.IndexP
   rc.base.models.DataBase.EqualsP
   rc.base.models.DataBase.CreateP
   rc.base.models.DataBase.ReadP
   rc.base.models.DataBase.UpdateP
   rc.base.models.DataBase.DeleteP
   rc.base.models.DataBase.CopyP
   rc.base.models.DataBase.NameP


Attributes
----------

.. autoapisummary::

   rc.base.models.DataBase.Tables
   rc.base.models.DataBase.defaultMeta
   rc.base.models.DataBase.ext


Properties
----------

.. autoapisummary::

   rc.base.models.DataBase.tables
   rc.base.models.DataBase.meta
   rc.base.models.DataBase.path


Methods
-------

.. autoapisummary::

   rc.base.models.DataBase.names
   rc.base.models.DataBase.defaults
   rc.base.models.DataBase.create
   rc.base.models.DataBase.copy
   rc.base.models.DataBase.delete
   rc.base.models.DataBase.extAppend
   rc.base.models.DataBase.mkdir


