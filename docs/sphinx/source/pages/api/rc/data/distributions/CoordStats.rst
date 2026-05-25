rc.data.distributions.CoordStats
================================


.. module:: rc.data.distributions

.. toctree::
   :hidden:

   /pages/api/rc/data/distributions/CoordStats.readOptions
   /pages/api/rc/data/distributions/CoordStats.CreateP
   /pages/api/rc/data/distributions/CoordStats.diff
   /pages/api/rc/data/distributions/CoordStats.create
   /pages/api/rc/data/distributions/CoordStats.ext
   /pages/api/rc/data/distributions/CoordStats.con
   /pages/api/rc/data/distributions/CoordStats.writeOptions
   /pages/api/rc/data/distributions/CoordStats.IndexP
   /pages/api/rc/data/distributions/CoordStats.EqualsP
   /pages/api/rc/data/distributions/CoordStats.ReadP
   /pages/api/rc/data/distributions/CoordStats.UpdateP
   /pages/api/rc/data/distributions/CoordStats.DeleteP
   /pages/api/rc/data/distributions/CoordStats.CopyP
   /pages/api/rc/data/distributions/CoordStats.heads
   /pages/api/rc/data/distributions/CoordStats.pl
   /pages/api/rc/data/distributions/CoordStats.np
   /pages/api/rc/data/distributions/CoordStats.tc
   /pages/api/rc/data/distributions/CoordStats.broadcast_to
   /pages/api/rc/data/distributions/CoordStats.copy
   /pages/api/rc/data/distributions/CoordStats.conjoinHeads
   /pages/api/rc/data/distributions/CoordStats.unjoinHeads
   /pages/api/rc/data/distributions/CoordStats.NameP
   /pages/api/rc/data/distributions/CoordStats.path
   /pages/api/rc/data/distributions/CoordStats.extAppend
   /pages/api/rc/data/distributions/CoordStats.mkdir
   /pages/api/rc/data/distributions/CoordStats.delete

.. py:class:: rc.data.distributions.CoordStats(path, table = None)

   Bases: :py:obj:`rc.data.designs.Table`

   .. autoapi-inheritance-diagram:: rc.data.distributions.CoordStats
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

   rc.data.distributions.CoordStats.CreateP
   rc.data.distributions.CoordStats.IndexP
   rc.data.distributions.CoordStats.EqualsP
   rc.data.distributions.CoordStats.ReadP
   rc.data.distributions.CoordStats.UpdateP
   rc.data.distributions.CoordStats.DeleteP
   rc.data.distributions.CoordStats.CopyP
   rc.data.distributions.CoordStats.NameP


Attributes
----------

.. autoapisummary::

   rc.data.distributions.CoordStats.readOptions
   rc.data.distributions.CoordStats.ext
   rc.data.distributions.CoordStats.con
   rc.data.distributions.CoordStats.writeOptions


Properties
----------

.. autoapisummary::

   rc.data.distributions.CoordStats.heads
   rc.data.distributions.CoordStats.pl
   rc.data.distributions.CoordStats.np
   rc.data.distributions.CoordStats.tc
   rc.data.distributions.CoordStats.path


Methods
-------

.. autoapisummary::

   rc.data.distributions.CoordStats.diff
   rc.data.distributions.CoordStats.create
   rc.data.distributions.CoordStats.broadcast_to
   rc.data.distributions.CoordStats.copy
   rc.data.distributions.CoordStats.conjoinHeads
   rc.data.distributions.CoordStats.unjoinHeads
   rc.data.distributions.CoordStats.extAppend
   rc.data.distributions.CoordStats.mkdir
   rc.data.distributions.CoordStats.delete


