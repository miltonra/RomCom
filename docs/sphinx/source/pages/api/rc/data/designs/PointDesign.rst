rc.data.designs.PointDesign
===========================


.. module:: rc.data.designs

.. toctree::
   :hidden:

   /pages/api/rc/data/designs/PointDesign.create
   /pages/api/rc/data/designs/PointDesign.CreateP
   /pages/api/rc/data/designs/PointDesign.outputAxis
   /pages/api/rc/data/designs/PointDesign.validate
   /pages/api/rc/data/designs/PointDesign.ext
   /pages/api/rc/data/designs/PointDesign.con
   /pages/api/rc/data/designs/PointDesign.readOptions
   /pages/api/rc/data/designs/PointDesign.writeOptions
   /pages/api/rc/data/designs/PointDesign.IndexP
   /pages/api/rc/data/designs/PointDesign.EqualsP
   /pages/api/rc/data/designs/PointDesign.ReadP
   /pages/api/rc/data/designs/PointDesign.UpdateP
   /pages/api/rc/data/designs/PointDesign.DeleteP
   /pages/api/rc/data/designs/PointDesign.CopyP
   /pages/api/rc/data/designs/PointDesign.heads
   /pages/api/rc/data/designs/PointDesign.pl
   /pages/api/rc/data/designs/PointDesign.np
   /pages/api/rc/data/designs/PointDesign.tc
   /pages/api/rc/data/designs/PointDesign.broadcast_to
   /pages/api/rc/data/designs/PointDesign.copy
   /pages/api/rc/data/designs/PointDesign.conjoinHeads
   /pages/api/rc/data/designs/PointDesign.unjoinHeads
   /pages/api/rc/data/designs/PointDesign.NameP
   /pages/api/rc/data/designs/PointDesign.path
   /pages/api/rc/data/designs/PointDesign.extAppend
   /pages/api/rc/data/designs/PointDesign.mkdir
   /pages/api/rc/data/designs/PointDesign.delete

.. py:class:: rc.data.designs.PointDesign(path, table = None)

   Bases: :py:obj:`Design`

   .. autoapi-inheritance-diagram:: rc.data.designs.PointDesign
      :parts: 1


   The internal format of ``Design``, which is thin (has few columns), and only one header row.
   Categorical axes are concatenated into a single column of categorical points.

   Construct ``self`` from a ``.csv`` file or ``Pl.DataFrame``.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``Table | Pl.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.data.designs.PointDesign.CreateP
   rc.data.designs.PointDesign.IndexP
   rc.data.designs.PointDesign.EqualsP
   rc.data.designs.PointDesign.ReadP
   rc.data.designs.PointDesign.UpdateP
   rc.data.designs.PointDesign.DeleteP
   rc.data.designs.PointDesign.CopyP
   rc.data.designs.PointDesign.NameP


Attributes
----------

.. autoapisummary::

   rc.data.designs.PointDesign.outputAxis
   rc.data.designs.PointDesign.ext
   rc.data.designs.PointDesign.con
   rc.data.designs.PointDesign.readOptions
   rc.data.designs.PointDesign.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.designs.PointDesign.heads
   rc.data.designs.PointDesign.pl
   rc.data.designs.PointDesign.np
   rc.data.designs.PointDesign.tc
   rc.data.designs.PointDesign.path


Methods
-------

.. autoapisummary::

   rc.data.designs.PointDesign.create
   rc.data.designs.PointDesign.validate
   rc.data.designs.PointDesign.broadcast_to
   rc.data.designs.PointDesign.copy
   rc.data.designs.PointDesign.conjoinHeads
   rc.data.designs.PointDesign.unjoinHeads
   rc.data.designs.PointDesign.extAppend
   rc.data.designs.PointDesign.mkdir
   rc.data.designs.PointDesign.delete


