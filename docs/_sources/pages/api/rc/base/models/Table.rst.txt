:html_theme.sidebar_secondary.remove:

rc.base.models.Table
====================


.. module:: rc.base.models

.. toctree::
   :hidden:

   /pages/api/rc/base/models/Table.ext
   /pages/api/rc/base/models/Table.con
   /pages/api/rc/base/models/Table.readOptions
   /pages/api/rc/base/models/Table.writeOptions
   /pages/api/rc/base/models/Table.IndexP
   /pages/api/rc/base/models/Table.EqualsP
   /pages/api/rc/base/models/Table.CreateP
   /pages/api/rc/base/models/Table.ReadP
   /pages/api/rc/base/models/Table.UpdateP
   /pages/api/rc/base/models/Table.DeleteP
   /pages/api/rc/base/models/Table.CopyP
   /pages/api/rc/base/models/Table.schema
   /pages/api/rc/base/models/Table.head
   /pages/api/rc/base/models/Table.df
   /pages/api/rc/base/models/Table.np
   /pages/api/rc/base/models/Table.tc
   /pages/api/rc/base/models/Table.broadcast_to
   /pages/api/rc/base/models/Table.create
   /pages/api/rc/base/models/Table.copy
   /pages/api/rc/base/models/Table.conjoinHeads
   /pages/api/rc/base/models/Table.unjoinHead
   /pages/api/rc/base/models/Table.NameP
   /pages/api/rc/base/models/Table.path
   /pages/api/rc/base/models/Table.extAppend
   /pages/api/rc/base/models/Table.mkdir
   /pages/api/rc/base/models/Table.delete

.. py:class:: rc.base.models.Table(path: rc.base.definitions.PathLike, table: rc.base.definitions.DataFrame | rc.base.definitions.Self | None = None)

   Bases: :py:obj:`Store`

   .. autoapi-inheritance-diagram:: rc.base.models.Table
      :parts: 1


   Concrete Class encapsulating a DataFrame backed by a ``.csv`` file.

   This Class may be usefully overridden to provide bespoke read and write options for
   file operations. SubClasses should follow the template (copy and paste it)::

       class MyTable(Table):

           readOptions: MetaData = Table.readOptions | {'myOption': myValue}

           writeOptions: MetaData = Table.writeOptions | {'myOption': myValue}

   Construct ``self`` from a ``.csv`` file or DataFrame.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``DataFrame | Table`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.base.models.Table.IndexP
   rc.base.models.Table.EqualsP
   rc.base.models.Table.CreateP
   rc.base.models.Table.ReadP
   rc.base.models.Table.UpdateP
   rc.base.models.Table.DeleteP
   rc.base.models.Table.CopyP
   rc.base.models.Table.NameP


Attributes
----------

.. autoapisummary::

   rc.base.models.Table.ext
   rc.base.models.Table.con
   rc.base.models.Table.readOptions
   rc.base.models.Table.writeOptions


Properties
----------

.. autoapisummary::

   rc.base.models.Table.schema
   rc.base.models.Table.head
   rc.base.models.Table.df
   rc.base.models.Table.np
   rc.base.models.Table.tc
   rc.base.models.Table.path


Methods
-------

.. autoapisummary::

   rc.base.models.Table.broadcast_to
   rc.base.models.Table.create
   rc.base.models.Table.copy
   rc.base.models.Table.conjoinHeads
   rc.base.models.Table.unjoinHead
   rc.base.models.Table.extAppend
   rc.base.models.Table.mkdir
   rc.base.models.Table.delete


