:html_theme.sidebar_secondary.remove:

rc.data.benchmarks.Vector
=========================


.. module:: rc.data.benchmarks

.. toctree::
   :hidden:

   /pages/api/rc/data/benchmarks/Vector.concat
   /pages/api/rc/data/benchmarks/Vector.meta
   /pages/api/rc/data/benchmarks/Vector.subVector

.. py:class:: rc.data.benchmarks.Vector(name: str, **scalars: Scalar)

   Bases: :py:obj:`dict`

   .. autoapi-inheritance-diagram:: rc.data.benchmarks.Vector
      :parts: 1


   A vector functon, which is little more than a named dictionary of Scalar functions,
   such that ``vector(x, **params)`` concatenates ``scalar(x, **params)``
   for each dictionary item ``key: Scalar``.

   Construct a vector function.

   :param name: The name of this Vector.
   :param \*\*scalars: The dict of Scalars comprising this Vector.

Properties
----------

.. autoapisummary::

   rc.data.benchmarks.Vector.meta


Methods
-------

.. autoapisummary::

   rc.data.benchmarks.Vector.concat
   rc.data.benchmarks.Vector.subVector


