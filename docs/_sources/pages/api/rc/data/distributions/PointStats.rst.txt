rc.data.distributions.PointStats
================================


.. module:: rc.data.distributions

.. toctree::
   :hidden:

   /pages/api/rc/data/distributions/PointStats.readOptions
   /pages/api/rc/data/distributions/PointStats.CreateP
   /pages/api/rc/data/distributions/PointStats.create
   /pages/api/rc/data/distributions/PointStats.ext
   /pages/api/rc/data/distributions/PointStats.con
   /pages/api/rc/data/distributions/PointStats.writeOptions
   /pages/api/rc/data/distributions/PointStats.IndexP
   /pages/api/rc/data/distributions/PointStats.EqualsP
   /pages/api/rc/data/distributions/PointStats.ReadP
   /pages/api/rc/data/distributions/PointStats.UpdateP
   /pages/api/rc/data/distributions/PointStats.DeleteP
   /pages/api/rc/data/distributions/PointStats.CopyP
   /pages/api/rc/data/distributions/PointStats.heads
   /pages/api/rc/data/distributions/PointStats.pl
   /pages/api/rc/data/distributions/PointStats.np
   /pages/api/rc/data/distributions/PointStats.tc
   /pages/api/rc/data/distributions/PointStats.broadcast_to
   /pages/api/rc/data/distributions/PointStats.copy
   /pages/api/rc/data/distributions/PointStats.conjoinHeads
   /pages/api/rc/data/distributions/PointStats.unjoinHeads
   /pages/api/rc/data/distributions/PointStats.NameP
   /pages/api/rc/data/distributions/PointStats.path
   /pages/api/rc/data/distributions/PointStats.extAppend
   /pages/api/rc/data/distributions/PointStats.mkdir
   /pages/api/rc/data/distributions/PointStats.delete

.. py:class:: rc.data.distributions.PointStats(path, table = None)

   Bases: :py:obj:`rc.data.designs.Table`

   .. autoapi-inheritance-diagram:: rc.data.distributions.PointStats
      :parts: 1


   Concrete Class encapsulating a ``Pl.DataFrame`` backed by a ``.csv`` file.

   This Class may be usefully overridden to provide bespoke read and write options for
   file operations. SubClasses should follow the template (copy and paste it)::

       class MyTable(Table):

           readOptions: MetaData = Table.readOptions | {'myOption': 'myValue'}
           """ File read options passed directly to
           `Pl.read_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.read_csv.html#polars.read_csv>`__."""

           writeOptions: MetaData = Table.writeOptions | {'myOption': 'myValue'}
           """ File write options passed directly to
           `Pl.DataFrame.write_csv <https://docs.pola.rs/api/python/dev/reference/api/polars.DataFrame.write_csv.html>`__."""

   Construct ``self`` from a ``.csv`` file or ``Pl.DataFrame``.

   :param path: The Path (file) to store ``self``. A ``.csv`` extension is implicitly appended.
   :param table: The ``Table | Pl.DataFrame`` to store. If ``None``, ``self`` is read from ``path``,
                 otherwise ``self`` is stored in ``path`` (which is overwritten if existing).

Protocols
----------

.. autoapisummary::

   rc.data.distributions.PointStats.CreateP
   rc.data.distributions.PointStats.IndexP
   rc.data.distributions.PointStats.EqualsP
   rc.data.distributions.PointStats.ReadP
   rc.data.distributions.PointStats.UpdateP
   rc.data.distributions.PointStats.DeleteP
   rc.data.distributions.PointStats.CopyP
   rc.data.distributions.PointStats.NameP


Attributes
----------

.. autoapisummary::

   rc.data.distributions.PointStats.readOptions
   rc.data.distributions.PointStats.ext
   rc.data.distributions.PointStats.con
   rc.data.distributions.PointStats.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.distributions.PointStats.heads
   rc.data.distributions.PointStats.pl
   rc.data.distributions.PointStats.np
   rc.data.distributions.PointStats.tc
   rc.data.distributions.PointStats.path


Methods
-------

.. autoapisummary::

   rc.data.distributions.PointStats.create
   rc.data.distributions.PointStats.broadcast_to
   rc.data.distributions.PointStats.copy
   rc.data.distributions.PointStats.conjoinHeads
   rc.data.distributions.PointStats.unjoinHeads
   rc.data.distributions.PointStats.extAppend
   rc.data.distributions.PointStats.mkdir
   rc.data.distributions.PointStats.delete


