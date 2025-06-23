rc.data.functions.Vector
========================

.. toctree::
   :hidden:

   /pages/api/rc/data/functions/Vector.concat
   /pages/api/rc/data/functions/Vector.meta
   /pages/api/rc/data/functions/Vector.subVector

.. py:class:: rc.data.functions.Vector(name, **kwargs)

   Bases: :py:obj:`dict`

   .. autoapi-inheritance-diagram:: rc.data.functions.Vector
      :parts: 1


   A vector functon, which is little more than a named dictionary of Scalar functions,
   such that ``vector(x, **kwargs)`` concatenates ``scalar(x, **kwargs)``
   for each dictionary item ``key: Scalar``.

Properties
----------

.. autoapisummary::

   rc.data.functions.Vector.meta


Methods
-------

.. autoapisummary::

   rc.data.functions.Vector.concat
   rc.data.functions.Vector.subVector


