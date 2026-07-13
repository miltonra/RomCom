:html_theme.sidebar_secondary.remove:

rc.base.models.Meta
===================


.. module:: rc.base.models

.. toctree::
   :hidden:

   /pages/api/rc/base/models/Meta.ext
   /pages/api/rc/base/models/Meta.IndexP
   /pages/api/rc/base/models/Meta.EqualsP
   /pages/api/rc/base/models/Meta.CreateP
   /pages/api/rc/base/models/Meta.ReadP
   /pages/api/rc/base/models/Meta.UpdateP
   /pages/api/rc/base/models/Meta.DeleteP
   /pages/api/rc/base/models/Meta.CopyP
   /pages/api/rc/base/models/Meta.create
   /pages/api/rc/base/models/Meta.copy
   /pages/api/rc/base/models/Meta.NameP
   /pages/api/rc/base/models/Meta.path
   /pages/api/rc/base/models/Meta.extAppend
   /pages/api/rc/base/models/Meta.mkdir
   /pages/api/rc/base/models/Meta.delete

.. py:class:: rc.base.models.Meta(path: rc.base.definitions.PathLike, **data: rc.base.definitions.Any)

   Bases: :py:obj:`Store`, :py:obj:`dict`

   .. autoapi-inheritance-diagram:: rc.base.models.Meta
      :parts: 1


   Concrete Class encapsulating metadata stored in a ``.json`` file.
   The place to store kwargs and options of all Types.

   Construct ``self`` from a ``.json`` file or ``MetaData``.
   This is read *or* write, *never* both: ``path`` is read *only* if ``**data`` is absent.

   :param path: The Path (file) to store ``self``. A ``.json`` extension is implicitly appended.
   :param \*\*data: The ``MetaData`` to store. If absent, ``self`` is read from ``path``,
                    otherwise ``self=dict(**data)`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.base.models.Meta.IndexP
   rc.base.models.Meta.EqualsP
   rc.base.models.Meta.CreateP
   rc.base.models.Meta.ReadP
   rc.base.models.Meta.UpdateP
   rc.base.models.Meta.DeleteP
   rc.base.models.Meta.CopyP
   rc.base.models.Meta.NameP


Attributes
----------

.. autoapisummary::

   rc.base.models.Meta.ext


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


