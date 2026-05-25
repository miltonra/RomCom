rc.data.distributions.PDF
=========================


.. module:: rc.data.distributions

.. toctree::
   :hidden:

   /pages/api/rc/data/distributions/PDF.readOptions
   /pages/api/rc/data/distributions/PDF.CreateP
   /pages/api/rc/data/distributions/PDF.compare
   /pages/api/rc/data/distributions/PDF.create
   /pages/api/rc/data/distributions/PDF.ext
   /pages/api/rc/data/distributions/PDF.con
   /pages/api/rc/data/distributions/PDF.writeOptions
   /pages/api/rc/data/distributions/PDF.IndexP
   /pages/api/rc/data/distributions/PDF.EqualsP
   /pages/api/rc/data/distributions/PDF.ReadP
   /pages/api/rc/data/distributions/PDF.UpdateP
   /pages/api/rc/data/distributions/PDF.DeleteP
   /pages/api/rc/data/distributions/PDF.CopyP
   /pages/api/rc/data/distributions/PDF.heads
   /pages/api/rc/data/distributions/PDF.pl
   /pages/api/rc/data/distributions/PDF.np
   /pages/api/rc/data/distributions/PDF.tc
   /pages/api/rc/data/distributions/PDF.broadcast_to
   /pages/api/rc/data/distributions/PDF.copy
   /pages/api/rc/data/distributions/PDF.conjoinHeads
   /pages/api/rc/data/distributions/PDF.unjoinHeads
   /pages/api/rc/data/distributions/PDF.NameP
   /pages/api/rc/data/distributions/PDF.path
   /pages/api/rc/data/distributions/PDF.extAppend
   /pages/api/rc/data/distributions/PDF.mkdir
   /pages/api/rc/data/distributions/PDF.delete

.. py:class:: rc.data.distributions.PDF(path, table = None)

   Bases: :py:obj:`rc.data.designs.Table`

   .. autoapi-inheritance-diagram:: rc.data.distributions.PDF
      :parts: 1


   The Probability Density Function(s) of categorical coords or points.

   Construct ``self`` from a ``.csv`` file or ``Pl.DataFrame``.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``Table | Pl.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.data.distributions.PDF.CreateP
   rc.data.distributions.PDF.IndexP
   rc.data.distributions.PDF.EqualsP
   rc.data.distributions.PDF.ReadP
   rc.data.distributions.PDF.UpdateP
   rc.data.distributions.PDF.DeleteP
   rc.data.distributions.PDF.CopyP
   rc.data.distributions.PDF.NameP


Attributes
----------

.. autoapisummary::

   rc.data.distributions.PDF.readOptions
   rc.data.distributions.PDF.ext
   rc.data.distributions.PDF.con
   rc.data.distributions.PDF.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.distributions.PDF.heads
   rc.data.distributions.PDF.pl
   rc.data.distributions.PDF.np
   rc.data.distributions.PDF.tc
   rc.data.distributions.PDF.path


Methods
-------

.. autoapisummary::

   rc.data.distributions.PDF.compare
   rc.data.distributions.PDF.create
   rc.data.distributions.PDF.broadcast_to
   rc.data.distributions.PDF.copy
   rc.data.distributions.PDF.conjoinHeads
   rc.data.distributions.PDF.unjoinHeads
   rc.data.distributions.PDF.extAppend
   rc.data.distributions.PDF.mkdir
   rc.data.distributions.PDF.delete


