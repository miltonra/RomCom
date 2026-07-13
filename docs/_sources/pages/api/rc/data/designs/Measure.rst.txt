:html_theme.sidebar_secondary.remove:

rc.data.designs.Measure
=======================


.. module:: rc.data.designs

.. toctree::
   :hidden:

   /pages/api/rc/data/designs/Measure.axisTypes
   /pages/api/rc/data/designs/Measure.by
   /pages/api/rc/data/designs/Measure.thin
   /pages/api/rc/data/designs/Measure.create
   /pages/api/rc/data/designs/Measure.CreateP
   /pages/api/rc/data/designs/Measure.AxisValidator
   /pages/api/rc/data/designs/Measure.Slice
   /pages/api/rc/data/designs/Measure.extPivot
   /pages/api/rc/data/designs/Measure.isFat
   /pages/api/rc/data/designs/Measure.M
   /pages/api/rc/data/designs/Measure.J
   /pages/api/rc/data/designs/Measure.L
   /pages/api/rc/data/designs/Measure.ls
   /pages/api/rc/data/designs/Measure.yPivot
   /pages/api/rc/data/designs/Measure.ext
   /pages/api/rc/data/designs/Measure.con
   /pages/api/rc/data/designs/Measure.readOptions
   /pages/api/rc/data/designs/Measure.writeOptions
   /pages/api/rc/data/designs/Measure.IndexP
   /pages/api/rc/data/designs/Measure.EqualsP
   /pages/api/rc/data/designs/Measure.ReadP
   /pages/api/rc/data/designs/Measure.UpdateP
   /pages/api/rc/data/designs/Measure.DeleteP
   /pages/api/rc/data/designs/Measure.CopyP
   /pages/api/rc/data/designs/Measure.schema
   /pages/api/rc/data/designs/Measure.head
   /pages/api/rc/data/designs/Measure.df
   /pages/api/rc/data/designs/Measure.np
   /pages/api/rc/data/designs/Measure.tc
   /pages/api/rc/data/designs/Measure.broadcast_to
   /pages/api/rc/data/designs/Measure.copy
   /pages/api/rc/data/designs/Measure.conjoinHeads
   /pages/api/rc/data/designs/Measure.unjoinHead
   /pages/api/rc/data/designs/Measure.NameP
   /pages/api/rc/data/designs/Measure.path
   /pages/api/rc/data/designs/Measure.extAppend
   /pages/api/rc/data/designs/Measure.mkdir
   /pages/api/rc/data/designs/Measure.delete

.. py:class:: rc.data.designs.Measure(path: rc.base.definitions.PathLike, table: rc.base.definitions.DataFrame | rc.base.definitions.Self | None = None)

   Bases: :py:obj:`Abstract`

   .. autoapi-inheritance-diagram:: rc.data.designs.Measure
      :parts: 1


   Measures a Design of user data.

   Construct ``self`` from a ``.csv`` file or DataFrame.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``DataFrame | Table`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Classes
-------

.. autoapisummary::

   rc.data.designs.Measure.AxisValidator
   rc.data.designs.Measure.Slice


Protocols
----------

.. autoapisummary::

   rc.data.designs.Measure.CreateP
   rc.data.designs.Measure.IndexP
   rc.data.designs.Measure.EqualsP
   rc.data.designs.Measure.ReadP
   rc.data.designs.Measure.UpdateP
   rc.data.designs.Measure.DeleteP
   rc.data.designs.Measure.CopyP
   rc.data.designs.Measure.NameP


Attributes
----------

.. autoapisummary::

   rc.data.designs.Measure.axisTypes
   rc.data.designs.Measure.by
   rc.data.designs.Measure.extPivot
   rc.data.designs.Measure.ext
   rc.data.designs.Measure.con
   rc.data.designs.Measure.readOptions
   rc.data.designs.Measure.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.designs.Measure.isFat
   rc.data.designs.Measure.M
   rc.data.designs.Measure.J
   rc.data.designs.Measure.L
   rc.data.designs.Measure.ls
   rc.data.designs.Measure.schema
   rc.data.designs.Measure.head
   rc.data.designs.Measure.df
   rc.data.designs.Measure.np
   rc.data.designs.Measure.tc
   rc.data.designs.Measure.path


Methods
-------

.. autoapisummary::

   rc.data.designs.Measure.thin
   rc.data.designs.Measure.create
   rc.data.designs.Measure.yPivot
   rc.data.designs.Measure.broadcast_to
   rc.data.designs.Measure.copy
   rc.data.designs.Measure.conjoinHeads
   rc.data.designs.Measure.unjoinHead
   rc.data.designs.Measure.extAppend
   rc.data.designs.Measure.mkdir
   rc.data.designs.Measure.delete


