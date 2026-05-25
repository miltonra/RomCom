rc.data.distributions.CoordPDF
==============================


.. module:: rc.data.distributions

.. toctree::
   :hidden:

   /pages/api/rc/data/distributions/CoordPDF.uniform
   /pages/api/rc/data/distributions/CoordPDF.create
   /pages/api/rc/data/distributions/CoordPDF.readOptions
   /pages/api/rc/data/distributions/CoordPDF.CreateP
   /pages/api/rc/data/distributions/CoordPDF.compare
   /pages/api/rc/data/distributions/CoordPDF.ext
   /pages/api/rc/data/distributions/CoordPDF.con
   /pages/api/rc/data/distributions/CoordPDF.writeOptions
   /pages/api/rc/data/distributions/CoordPDF.IndexP
   /pages/api/rc/data/distributions/CoordPDF.EqualsP
   /pages/api/rc/data/distributions/CoordPDF.ReadP
   /pages/api/rc/data/distributions/CoordPDF.UpdateP
   /pages/api/rc/data/distributions/CoordPDF.DeleteP
   /pages/api/rc/data/distributions/CoordPDF.CopyP
   /pages/api/rc/data/distributions/CoordPDF.heads
   /pages/api/rc/data/distributions/CoordPDF.pl
   /pages/api/rc/data/distributions/CoordPDF.np
   /pages/api/rc/data/distributions/CoordPDF.tc
   /pages/api/rc/data/distributions/CoordPDF.broadcast_to
   /pages/api/rc/data/distributions/CoordPDF.copy
   /pages/api/rc/data/distributions/CoordPDF.conjoinHeads
   /pages/api/rc/data/distributions/CoordPDF.unjoinHeads
   /pages/api/rc/data/distributions/CoordPDF.NameP
   /pages/api/rc/data/distributions/CoordPDF.path
   /pages/api/rc/data/distributions/CoordPDF.extAppend
   /pages/api/rc/data/distributions/CoordPDF.mkdir
   /pages/api/rc/data/distributions/CoordPDF.delete

.. py:class:: rc.data.distributions.CoordPDF(path, table = None)

   Bases: :py:obj:`PDF`

   .. autoapi-inheritance-diagram:: rc.data.distributions.CoordPDF
      :parts: 1


   The Probability Density Function(s) of statistically independent categorical coords.

   Construct ``self`` from a ``.csv`` file or ``Pl.DataFrame``.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``Table | Pl.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.data.distributions.CoordPDF.CreateP
   rc.data.distributions.CoordPDF.IndexP
   rc.data.distributions.CoordPDF.EqualsP
   rc.data.distributions.CoordPDF.ReadP
   rc.data.distributions.CoordPDF.UpdateP
   rc.data.distributions.CoordPDF.DeleteP
   rc.data.distributions.CoordPDF.CopyP
   rc.data.distributions.CoordPDF.NameP


Attributes
----------

.. autoapisummary::

   rc.data.distributions.CoordPDF.readOptions
   rc.data.distributions.CoordPDF.ext
   rc.data.distributions.CoordPDF.con
   rc.data.distributions.CoordPDF.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.distributions.CoordPDF.heads
   rc.data.distributions.CoordPDF.pl
   rc.data.distributions.CoordPDF.np
   rc.data.distributions.CoordPDF.tc
   rc.data.distributions.CoordPDF.path


Methods
-------

.. autoapisummary::

   rc.data.distributions.CoordPDF.uniform
   rc.data.distributions.CoordPDF.create
   rc.data.distributions.CoordPDF.compare
   rc.data.distributions.CoordPDF.broadcast_to
   rc.data.distributions.CoordPDF.copy
   rc.data.distributions.CoordPDF.conjoinHeads
   rc.data.distributions.CoordPDF.unjoinHeads
   rc.data.distributions.CoordPDF.extAppend
   rc.data.distributions.CoordPDF.mkdir
   rc.data.distributions.CoordPDF.delete


