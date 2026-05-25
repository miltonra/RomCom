rc.data.designs.CoordDesign
===========================


.. module:: rc.data.designs

.. toctree::
   :hidden:

   /pages/api/rc/data/designs/CoordDesign.readOptions
   /pages/api/rc/data/designs/CoordDesign.create
   /pages/api/rc/data/designs/CoordDesign.CreateP
   /pages/api/rc/data/designs/CoordDesign.outputAxis
   /pages/api/rc/data/designs/CoordDesign.validate
   /pages/api/rc/data/designs/CoordDesign.ext
   /pages/api/rc/data/designs/CoordDesign.con
   /pages/api/rc/data/designs/CoordDesign.writeOptions
   /pages/api/rc/data/designs/CoordDesign.IndexP
   /pages/api/rc/data/designs/CoordDesign.EqualsP
   /pages/api/rc/data/designs/CoordDesign.ReadP
   /pages/api/rc/data/designs/CoordDesign.UpdateP
   /pages/api/rc/data/designs/CoordDesign.DeleteP
   /pages/api/rc/data/designs/CoordDesign.CopyP
   /pages/api/rc/data/designs/CoordDesign.heads
   /pages/api/rc/data/designs/CoordDesign.pl
   /pages/api/rc/data/designs/CoordDesign.np
   /pages/api/rc/data/designs/CoordDesign.tc
   /pages/api/rc/data/designs/CoordDesign.broadcast_to
   /pages/api/rc/data/designs/CoordDesign.copy
   /pages/api/rc/data/designs/CoordDesign.conjoinHeads
   /pages/api/rc/data/designs/CoordDesign.unjoinHeads
   /pages/api/rc/data/designs/CoordDesign.NameP
   /pages/api/rc/data/designs/CoordDesign.path
   /pages/api/rc/data/designs/CoordDesign.extAppend
   /pages/api/rc/data/designs/CoordDesign.mkdir
   /pages/api/rc/data/designs/CoordDesign.delete

.. py:class:: rc.data.designs.CoordDesign(path, table = None)

   Bases: :py:obj:`Design`

   .. autoapi-inheritance-diagram:: rc.data.designs.CoordDesign
      :parts: 1


   The familiar user format of a Design which has many axes (columns), and two header rows.
   The first header row contains the axisType, the second header row contains the axis.

   Construct ``self`` from a ``.csv`` file or ``Pl.DataFrame``.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``Table | Pl.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.data.designs.CoordDesign.CreateP
   rc.data.designs.CoordDesign.IndexP
   rc.data.designs.CoordDesign.EqualsP
   rc.data.designs.CoordDesign.ReadP
   rc.data.designs.CoordDesign.UpdateP
   rc.data.designs.CoordDesign.DeleteP
   rc.data.designs.CoordDesign.CopyP
   rc.data.designs.CoordDesign.NameP


Attributes
----------

.. autoapisummary::

   rc.data.designs.CoordDesign.readOptions
   rc.data.designs.CoordDesign.outputAxis
   rc.data.designs.CoordDesign.ext
   rc.data.designs.CoordDesign.con
   rc.data.designs.CoordDesign.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.designs.CoordDesign.heads
   rc.data.designs.CoordDesign.pl
   rc.data.designs.CoordDesign.np
   rc.data.designs.CoordDesign.tc
   rc.data.designs.CoordDesign.path


Methods
-------

.. autoapisummary::

   rc.data.designs.CoordDesign.create
   rc.data.designs.CoordDesign.validate
   rc.data.designs.CoordDesign.broadcast_to
   rc.data.designs.CoordDesign.copy
   rc.data.designs.CoordDesign.conjoinHeads
   rc.data.designs.CoordDesign.unjoinHeads
   rc.data.designs.CoordDesign.extAppend
   rc.data.designs.CoordDesign.mkdir
   rc.data.designs.CoordDesign.delete


