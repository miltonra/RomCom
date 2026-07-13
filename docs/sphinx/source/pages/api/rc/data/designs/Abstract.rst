:html_theme.sidebar_secondary.remove:

rc.data.designs.Abstract
========================


.. module:: rc.data.designs

.. toctree::
   :hidden:

   /pages/api/rc/data/designs/Abstract.CreateP
   /pages/api/rc/data/designs/Abstract.axisTypes
   /pages/api/rc/data/designs/Abstract.AxisValidator
   /pages/api/rc/data/designs/Abstract.Slice
   /pages/api/rc/data/designs/Abstract.extPivot
   /pages/api/rc/data/designs/Abstract.isFat
   /pages/api/rc/data/designs/Abstract.M
   /pages/api/rc/data/designs/Abstract.J
   /pages/api/rc/data/designs/Abstract.L
   /pages/api/rc/data/designs/Abstract.ls
   /pages/api/rc/data/designs/Abstract.yPivot
   /pages/api/rc/data/designs/Abstract.create
   /pages/api/rc/data/designs/Abstract.ext
   /pages/api/rc/data/designs/Abstract.con
   /pages/api/rc/data/designs/Abstract.readOptions
   /pages/api/rc/data/designs/Abstract.writeOptions
   /pages/api/rc/data/designs/Abstract.IndexP
   /pages/api/rc/data/designs/Abstract.EqualsP
   /pages/api/rc/data/designs/Abstract.ReadP
   /pages/api/rc/data/designs/Abstract.UpdateP
   /pages/api/rc/data/designs/Abstract.DeleteP
   /pages/api/rc/data/designs/Abstract.CopyP
   /pages/api/rc/data/designs/Abstract.schema
   /pages/api/rc/data/designs/Abstract.head
   /pages/api/rc/data/designs/Abstract.df
   /pages/api/rc/data/designs/Abstract.np
   /pages/api/rc/data/designs/Abstract.tc
   /pages/api/rc/data/designs/Abstract.broadcast_to
   /pages/api/rc/data/designs/Abstract.copy
   /pages/api/rc/data/designs/Abstract.conjoinHeads
   /pages/api/rc/data/designs/Abstract.unjoinHead
   /pages/api/rc/data/designs/Abstract.NameP
   /pages/api/rc/data/designs/Abstract.path
   /pages/api/rc/data/designs/Abstract.extAppend
   /pages/api/rc/data/designs/Abstract.mkdir
   /pages/api/rc/data/designs/Abstract.delete

.. py:class:: rc.data.designs.Abstract(path: rc.base.definitions.PathLike, table: rc.base.definitions.DataFrame | rc.base.definitions.Self | None = None)

   Bases: :py:obj:`rc.base.Table`

   .. autoapi-inheritance-diagram:: rc.data.designs.Abstract
      :parts: 1


   Abstract scaffolding for Design and Measure, providing shared tabulation facilities for user data.

   Construct ``self`` from a ``.csv`` file or DataFrame.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``DataFrame | Table`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Classes
-------

.. autoapisummary::

   rc.data.designs.Abstract.AxisValidator
   rc.data.designs.Abstract.Slice


Protocols
----------

.. autoapisummary::

   rc.data.designs.Abstract.CreateP
   rc.data.designs.Abstract.IndexP
   rc.data.designs.Abstract.EqualsP
   rc.data.designs.Abstract.ReadP
   rc.data.designs.Abstract.UpdateP
   rc.data.designs.Abstract.DeleteP
   rc.data.designs.Abstract.CopyP
   rc.data.designs.Abstract.NameP


Attributes
----------

.. autoapisummary::

   rc.data.designs.Abstract.axisTypes
   rc.data.designs.Abstract.extPivot
   rc.data.designs.Abstract.ext
   rc.data.designs.Abstract.con
   rc.data.designs.Abstract.readOptions
   rc.data.designs.Abstract.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.designs.Abstract.isFat
   rc.data.designs.Abstract.M
   rc.data.designs.Abstract.J
   rc.data.designs.Abstract.L
   rc.data.designs.Abstract.ls
   rc.data.designs.Abstract.schema
   rc.data.designs.Abstract.head
   rc.data.designs.Abstract.df
   rc.data.designs.Abstract.np
   rc.data.designs.Abstract.tc
   rc.data.designs.Abstract.path


Methods
-------

.. autoapisummary::

   rc.data.designs.Abstract.yPivot
   rc.data.designs.Abstract.create
   rc.data.designs.Abstract.broadcast_to
   rc.data.designs.Abstract.copy
   rc.data.designs.Abstract.conjoinHeads
   rc.data.designs.Abstract.unjoinHead
   rc.data.designs.Abstract.extAppend
   rc.data.designs.Abstract.mkdir
   rc.data.designs.Abstract.delete


