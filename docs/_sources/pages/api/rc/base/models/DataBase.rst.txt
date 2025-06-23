rc.base.models.DataBase
=======================

.. toctree::
   :hidden:

   /pages/api/rc/base/models/DataBase.NamedTables
   /pages/api/rc/base/models/DataBase.options
   /pages/api/rc/base/models/DataBase.Indexed
   /pages/api/rc/base/models/DataBase.Len
   /pages/api/rc/base/models/DataBase.CreateProtocol
   /pages/api/rc/base/models/DataBase.ReadProtocol
   /pages/api/rc/base/models/DataBase.UpdateProtocol
   /pages/api/rc/base/models/DataBase.DeleteProtocol
   /pages/api/rc/base/models/DataBase.CopyProtocol
   /pages/api/rc/base/models/DataBase.StrReprProtocol
   /pages/api/rc/base/models/DataBase.namedTables
   /pages/api/rc/base/models/DataBase.meta
   /pages/api/rc/base/models/DataBase.names
   /pages/api/rc/base/models/DataBase.defaults
   /pages/api/rc/base/models/DataBase.create
   /pages/api/rc/base/models/DataBase.copy
   /pages/api/rc/base/models/DataBase.delete
   /pages/api/rc/base/models/DataBase.Path
   /pages/api/rc/base/models/DataBase.ext
   /pages/api/rc/base/models/DataBase.path
   /pages/api/rc/base/models/DataBase.extAppend
   /pages/api/rc/base/models/DataBase.mkdir

.. py:class:: rc.base.models.DataBase(path, **tables)

   Bases: :py:obj:`Store`

   .. autoapi-inheritance-diagram:: rc.base.models.DataBase
      :parts: 1


   ``NamedTables(NamedTuple)`` in a folder alongside ``Meta``. Abstract base class for any model.

   ``DataBase`` subclasses must be implemented according to the template (copy and paste it)::

       class MyDataBase(DataBase):

           class NamedTables(NamedTuple):

               names[i]: Table | Matrix | MetaData = pd.DataFrame(defaults[names[i]].pd)   #: Comment
               ...

               def __call__(self, name: str) -> Table | Matrix | MetaData:
                   """ Returns the Table named ``name``."""
                   return getattr(self, name)


           options: NamedTables[MetaData] = NamedTables(**{name: table.options for name, table in {}.items()})
           """ Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
           Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
           the remainder populate ``self[i].options.read``. Must be overridden."""

           defaultMetaData: MetaData = {'Tables': options._asdict()}

Classes
-------

.. autoapisummary::

   rc.base.models.DataBase.NamedTables


Protocols
----------

.. autoapisummary::

   rc.base.models.DataBase.Indexed
   rc.base.models.DataBase.Len
   rc.base.models.DataBase.CreateProtocol
   rc.base.models.DataBase.ReadProtocol
   rc.base.models.DataBase.UpdateProtocol
   rc.base.models.DataBase.DeleteProtocol
   rc.base.models.DataBase.CopyProtocol
   rc.base.models.DataBase.StrReprProtocol


Attributes
----------

.. autoapisummary::

   rc.base.models.DataBase.options
   rc.base.models.DataBase.Path
   rc.base.models.DataBase.ext


Properties
----------

.. autoapisummary::

   rc.base.models.DataBase.namedTables
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


