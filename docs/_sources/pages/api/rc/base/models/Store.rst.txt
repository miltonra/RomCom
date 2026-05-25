rc.base.models.Store
====================


.. module:: rc.base.models

.. toctree::
   :hidden:

   /pages/api/rc/base/models/Store.ext
   /pages/api/rc/base/models/Store.CreateP
   /pages/api/rc/base/models/Store.ReadP
   /pages/api/rc/base/models/Store.UpdateP
   /pages/api/rc/base/models/Store.DeleteP
   /pages/api/rc/base/models/Store.CopyP
   /pages/api/rc/base/models/Store.NameP
   /pages/api/rc/base/models/Store.path
   /pages/api/rc/base/models/Store.extAppend
   /pages/api/rc/base/models/Store.mkdir
   /pages/api/rc/base/models/Store.create
   /pages/api/rc/base/models/Store.copy
   /pages/api/rc/base/models/Store.delete

.. py:class:: rc.base.models.Store(path, **kwargs)

   Bases: :py:obj:`rc.base.definitions.ABC`

   .. autoapi-inheritance-diagram:: rc.base.models.Store
      :parts: 1


   Base Class for any stored Class. Users are not expected to SubClass this Class directly.

   Store ``path`` in ``self._path``.

   Overrides should call ``super(Store).__init__(path)`` as a matter of priority.

   :param path: The Path to ``self``. Do not include an extension.

Protocols
----------

.. autoapisummary::

   rc.base.models.Store.CreateP
   rc.base.models.Store.ReadP
   rc.base.models.Store.UpdateP
   rc.base.models.Store.DeleteP
   rc.base.models.Store.CopyP
   rc.base.models.Store.NameP


Attributes
----------

.. autoapisummary::

   rc.base.models.Store.ext


Properties
----------

.. autoapisummary::

   rc.base.models.Store.path


Methods
-------

.. autoapisummary::

   rc.base.models.Store.extAppend
   rc.base.models.Store.mkdir
   rc.base.models.Store.create
   rc.base.models.Store.copy
   rc.base.models.Store.delete


