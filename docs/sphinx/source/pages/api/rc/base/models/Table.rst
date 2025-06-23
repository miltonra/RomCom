rc.base.models.Table
====================

.. toctree::
   :hidden:

   /pages/api/rc/base/models/Table.ext
   /pages/api/rc/base/models/Table.writeOptions
   /pages/api/rc/base/models/Table.CreateProtocol
   /pages/api/rc/base/models/Table.ReadProtocol
   /pages/api/rc/base/models/Table.UpdateProtocol
   /pages/api/rc/base/models/Table.DeleteProtocol
   /pages/api/rc/base/models/Table.CopyProtocol
   /pages/api/rc/base/models/Table.StrReprProtocol
   /pages/api/rc/base/models/Table.options
   /pages/api/rc/base/models/Table.pd
   /pages/api/rc/base/models/Table.np
   /pages/api/rc/base/models/Table.tc
   /pages/api/rc/base/models/Table.broadcast_to
   /pages/api/rc/base/models/Table.create
   /pages/api/rc/base/models/Table.copy
   /pages/api/rc/base/models/Table.Path
   /pages/api/rc/base/models/Table.path
   /pages/api/rc/base/models/Table.extAppend
   /pages/api/rc/base/models/Table.mkdir
   /pages/api/rc/base/models/Table.delete

.. py:class:: rc.base.models.Table(path, data = None, **options)

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

Protocols
----------

.. autoapisummary::

   rc.base.models.Table.CreateProtocol
   rc.base.models.Table.ReadProtocol
   rc.base.models.Table.UpdateProtocol
   rc.base.models.Table.DeleteProtocol
   rc.base.models.Table.CopyProtocol
   rc.base.models.Table.StrReprProtocol


Attributes
----------

.. autoapisummary::

   rc.base.models.Table.ext
   rc.base.models.Table.writeOptions
   rc.base.models.Table.Path


Properties
----------

.. autoapisummary::

   rc.base.models.Table.options
   rc.base.models.Table.pd
   rc.base.models.Table.np
   rc.base.models.Table.tc
   rc.base.models.Table.path


Methods
-------

.. autoapisummary::

   rc.base.models.Table.broadcast_to
   rc.base.models.Table.create
   rc.base.models.Table.copy
   rc.base.models.Table.extAppend
   rc.base.models.Table.mkdir
   rc.base.models.Table.delete


