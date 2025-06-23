rc.base.models.Meta
===================

.. toctree::
   :hidden:

   /pages/api/rc/base/models/Meta.ext
   /pages/api/rc/base/models/Meta.Indexed
   /pages/api/rc/base/models/Meta.Len
   /pages/api/rc/base/models/Meta.CreateProtocol
   /pages/api/rc/base/models/Meta.ReadProtocol
   /pages/api/rc/base/models/Meta.UpdateProtocol
   /pages/api/rc/base/models/Meta.DeleteProtocol
   /pages/api/rc/base/models/Meta.CopyProtocol
   /pages/api/rc/base/models/Meta.create
   /pages/api/rc/base/models/Meta.copy
   /pages/api/rc/base/models/Meta.Path
   /pages/api/rc/base/models/Meta.StrReprProtocol
   /pages/api/rc/base/models/Meta.path
   /pages/api/rc/base/models/Meta.extAppend
   /pages/api/rc/base/models/Meta.mkdir
   /pages/api/rc/base/models/Meta.delete

.. py:class:: rc.base.models.Meta(path, **data)

   Bases: :py:obj:`Store`, :py:obj:`dict`

   .. autoapi-inheritance-diagram:: rc.base.models.Meta
      :parts: 1


   Concrete class encapsulating metadata stored in a ``.json`` file.

Protocols
----------

.. autoapisummary::

   rc.base.models.Meta.Indexed
   rc.base.models.Meta.Len
   rc.base.models.Meta.CreateProtocol
   rc.base.models.Meta.ReadProtocol
   rc.base.models.Meta.UpdateProtocol
   rc.base.models.Meta.DeleteProtocol
   rc.base.models.Meta.CopyProtocol
   rc.base.models.Meta.StrReprProtocol


Attributes
----------

.. autoapisummary::

   rc.base.models.Meta.ext
   rc.base.models.Meta.Path


Properties
----------

.. autoapisummary::

   rc.base.models.Meta.path


Methods
-------

.. autoapisummary::

   rc.base.models.Meta.create
   rc.base.models.Meta.copy
   rc.base.models.Meta.extAppend
   rc.base.models.Meta.mkdir
   rc.base.models.Meta.delete


