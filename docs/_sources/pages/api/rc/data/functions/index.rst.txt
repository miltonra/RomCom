rc.data.functions
=================

.. py:module:: rc.data.functions

.. autoapi-nested-parse::

   Test functions, taken from `SALib <https://salib.readthedocs.io/en/latest/api/SALib.test_functions.html>`_.



Classes
-------

.. autoapisummary::

   rc.data.functions.Scalar
   rc.data.functions.Vector


Methods
---------

.. autoapisummary::

   rc.data.functions.linspace


Module Contents
---------------

.. py:class:: Scalar(call, loc, scale, m, **kwargs)

   A scalar function ``scalar`` such that ``scalar(x, kwargs)`` calls
   ``self.call(self.loc + self.scale * x[:, :self.m], **(self.kwargs | kwargs)``.


.. py:class:: Vector(name, **kwargs)

   Bases: :py:obj:`dict`

   .. autoapi-inheritance-diagram:: rc.data.functions.Vector
      :parts: 1


   A vector functon, which is little more than a named dictionary of Scalar functions,
   such that ``vector(x, **kwargs)`` concatenates ``scalar(x, **kwargs)``
   for each dictionary item ``key: Scalar``.


   .. py:method:: concat(name, vectors)
      :classmethod:


      Concatenate vectors.

      :param name: The name of the returned ``Vector``.
      :param vectors: A sequence of ``Vector`` functions to concatenate.

      Returns: The concatenation of ``vectors``, named ``name``.



   .. py:property:: meta
      :type: rc.base.Dict


      Meta data for providing to ``data.storage``.


   .. py:method:: subVector(name, scalars)

      Create a subVector of ``self``.

      :param name: The name of the ``subVector``.
      :param scalars: The keys of the items of ``self`` to be included in subVector.

      Returns: A new instance of ``Vector`` named ``name`` containing the ``Scalars`` keyed ``scalars``.
          Effectively the pseudo-slice ``self[scalars]``.



.. py:function:: linspace(start, stop, shape)

   A multi-dimensional version of ``np.linspace``, distributing values throughout ``shape``.

   :param start: Start value, which will be returned in ``linspace(...)[0,...,0]``.
   :param stop: Stop value, which will be returned in ``linspace(...)[-1,...,-1]``.
   :param shape: The ``linspace.shape`` to return.

   Returns: ``np.reshape(np.linspace(start, stop, int(np.prod(shape)), endpoint=True), newshape=shape)``.


