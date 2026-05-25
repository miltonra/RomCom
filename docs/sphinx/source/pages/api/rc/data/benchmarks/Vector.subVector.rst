rc.data.benchmarks.Vector.subVector
===================================

.. py:method:: rc.data.benchmarks.Vector.subVector(name, scalars)

   Create a subVector of ``self``.

   :param name: The name of the ``subVector``.
   :param scalars: The keys of the items of ``self`` to be included in subVector.

   Returns: A new instance of ``Vector`` named ``name`` containing the ``Scalars`` keyed ``scalars``.
       Effectively the pseudo-slice ``self[scalars]``.

