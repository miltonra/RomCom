rc.data.distributions.PointPDF
==============================


.. module:: rc.data.distributions

.. toctree::
   :hidden:

   /pages/api/rc/data/distributions/PointPDF.independent
   /pages/api/rc/data/distributions/PointPDF.create
   /pages/api/rc/data/distributions/PointPDF.readOptions
   /pages/api/rc/data/distributions/PointPDF.CreateP
   /pages/api/rc/data/distributions/PointPDF.compare
   /pages/api/rc/data/distributions/PointPDF.ext
   /pages/api/rc/data/distributions/PointPDF.con
   /pages/api/rc/data/distributions/PointPDF.writeOptions
   /pages/api/rc/data/distributions/PointPDF.IndexP
   /pages/api/rc/data/distributions/PointPDF.EqualsP
   /pages/api/rc/data/distributions/PointPDF.ReadP
   /pages/api/rc/data/distributions/PointPDF.UpdateP
   /pages/api/rc/data/distributions/PointPDF.DeleteP
   /pages/api/rc/data/distributions/PointPDF.CopyP
   /pages/api/rc/data/distributions/PointPDF.heads
   /pages/api/rc/data/distributions/PointPDF.pl
   /pages/api/rc/data/distributions/PointPDF.np
   /pages/api/rc/data/distributions/PointPDF.tc
   /pages/api/rc/data/distributions/PointPDF.broadcast_to
   /pages/api/rc/data/distributions/PointPDF.copy
   /pages/api/rc/data/distributions/PointPDF.conjoinHeads
   /pages/api/rc/data/distributions/PointPDF.unjoinHeads
   /pages/api/rc/data/distributions/PointPDF.NameP
   /pages/api/rc/data/distributions/PointPDF.path
   /pages/api/rc/data/distributions/PointPDF.extAppend
   /pages/api/rc/data/distributions/PointPDF.mkdir
   /pages/api/rc/data/distributions/PointPDF.delete

.. py:class:: rc.data.distributions.PointPDF(path, table = None)

   Bases: :py:obj:`PDF`

   .. autoapi-inheritance-diagram:: rc.data.distributions.PointPDF
      :parts: 1


   The Probability Density Function of categorical points.

   Construct ``self`` from a ``.csv`` file or ``Pl.DataFrame``.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``Table | Pl.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.data.distributions.PointPDF.CreateP
   rc.data.distributions.PointPDF.IndexP
   rc.data.distributions.PointPDF.EqualsP
   rc.data.distributions.PointPDF.ReadP
   rc.data.distributions.PointPDF.UpdateP
   rc.data.distributions.PointPDF.DeleteP
   rc.data.distributions.PointPDF.CopyP
   rc.data.distributions.PointPDF.NameP


Attributes
----------

.. autoapisummary::

   rc.data.distributions.PointPDF.readOptions
   rc.data.distributions.PointPDF.ext
   rc.data.distributions.PointPDF.con
   rc.data.distributions.PointPDF.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.distributions.PointPDF.heads
   rc.data.distributions.PointPDF.pl
   rc.data.distributions.PointPDF.np
   rc.data.distributions.PointPDF.tc
   rc.data.distributions.PointPDF.path


Methods
-------

.. autoapisummary::

   rc.data.distributions.PointPDF.independent
   rc.data.distributions.PointPDF.create
   rc.data.distributions.PointPDF.compare
   rc.data.distributions.PointPDF.broadcast_to
   rc.data.distributions.PointPDF.copy
   rc.data.distributions.PointPDF.conjoinHeads
   rc.data.distributions.PointPDF.unjoinHeads
   rc.data.distributions.PointPDF.extAppend
   rc.data.distributions.PointPDF.mkdir
   rc.data.distributions.PointPDF.delete


