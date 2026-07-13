:html_theme.sidebar_secondary.remove:

rc.data.designs.Design
======================


.. module:: rc.data.designs

.. toctree::
   :hidden:

   /pages/api/rc/data/designs/Design.CreateP
   /pages/api/rc/data/designs/Design.axisLexicon
   /pages/api/rc/data/designs/Design.axisTypes
   /pages/api/rc/data/designs/Design.create
   /pages/api/rc/data/designs/Design.AxisValidator
   /pages/api/rc/data/designs/Design.Slice
   /pages/api/rc/data/designs/Design.extPivot
   /pages/api/rc/data/designs/Design.isFat
   /pages/api/rc/data/designs/Design.M
   /pages/api/rc/data/designs/Design.J
   /pages/api/rc/data/designs/Design.L
   /pages/api/rc/data/designs/Design.ls
   /pages/api/rc/data/designs/Design.yPivot
   /pages/api/rc/data/designs/Design.ext
   /pages/api/rc/data/designs/Design.con
   /pages/api/rc/data/designs/Design.readOptions
   /pages/api/rc/data/designs/Design.writeOptions
   /pages/api/rc/data/designs/Design.IndexP
   /pages/api/rc/data/designs/Design.EqualsP
   /pages/api/rc/data/designs/Design.ReadP
   /pages/api/rc/data/designs/Design.UpdateP
   /pages/api/rc/data/designs/Design.DeleteP
   /pages/api/rc/data/designs/Design.CopyP
   /pages/api/rc/data/designs/Design.schema
   /pages/api/rc/data/designs/Design.head
   /pages/api/rc/data/designs/Design.df
   /pages/api/rc/data/designs/Design.np
   /pages/api/rc/data/designs/Design.tc
   /pages/api/rc/data/designs/Design.broadcast_to
   /pages/api/rc/data/designs/Design.copy
   /pages/api/rc/data/designs/Design.conjoinHeads
   /pages/api/rc/data/designs/Design.unjoinHead
   /pages/api/rc/data/designs/Design.NameP
   /pages/api/rc/data/designs/Design.path
   /pages/api/rc/data/designs/Design.extAppend
   /pages/api/rc/data/designs/Design.mkdir
   /pages/api/rc/data/designs/Design.delete

.. py:class:: rc.data.designs.Design(path: rc.base.definitions.PathLike, table: rc.base.definitions.DataFrame | rc.base.definitions.Self | None = None)

   Bases: :py:obj:`Abstract`

   .. autoapi-inheritance-diagram:: rc.data.designs.Design
      :parts: 1


   A Design of user data, tabulating continuous inputs, categorical inputs, and unpivoted outputs.

   Construct ``self`` from a ``.csv`` file or DataFrame.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``DataFrame | Table`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Classes
-------

.. autoapisummary::

   rc.data.designs.Design.AxisValidator
   rc.data.designs.Design.Slice


Protocols
----------

.. autoapisummary::

   rc.data.designs.Design.CreateP
   rc.data.designs.Design.IndexP
   rc.data.designs.Design.EqualsP
   rc.data.designs.Design.ReadP
   rc.data.designs.Design.UpdateP
   rc.data.designs.Design.DeleteP
   rc.data.designs.Design.CopyP
   rc.data.designs.Design.NameP


Attributes
----------

.. autoapisummary::

   rc.data.designs.Design.axisLexicon
   rc.data.designs.Design.axisTypes
   rc.data.designs.Design.extPivot
   rc.data.designs.Design.ext
   rc.data.designs.Design.con
   rc.data.designs.Design.readOptions
   rc.data.designs.Design.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.designs.Design.isFat
   rc.data.designs.Design.M
   rc.data.designs.Design.J
   rc.data.designs.Design.L
   rc.data.designs.Design.ls
   rc.data.designs.Design.schema
   rc.data.designs.Design.head
   rc.data.designs.Design.df
   rc.data.designs.Design.np
   rc.data.designs.Design.tc
   rc.data.designs.Design.path


Methods
-------

.. autoapisummary::

   rc.data.designs.Design.create
   rc.data.designs.Design.yPivot
   rc.data.designs.Design.broadcast_to
   rc.data.designs.Design.copy
   rc.data.designs.Design.conjoinHeads
   rc.data.designs.Design.unjoinHead
   rc.data.designs.Design.extAppend
   rc.data.designs.Design.mkdir
   rc.data.designs.Design.delete


