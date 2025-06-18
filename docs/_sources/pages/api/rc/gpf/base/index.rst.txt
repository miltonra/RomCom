rc.gpf.base
===========

.. py:module:: rc.gpf.base

.. autoapi-nested-parse::

   Contains extensions to gpflow.base.



Classes
-------

.. autoapisummary::

   rc.gpf.base.Variance


Module Contents
---------------

.. py:class:: Variance(value, name = 'Variance', cholesky_diagonal_lower_bound = CHOLESKY_DIAGONAL_LOWER_BOUND)

   Bases: :py:obj:`gpflow.Module`

   .. autoapi-inheritance-diagram:: rc.gpf.base.Variance
      :parts: 1


   A non-diagonal Variance Matrix.


   .. py:property:: shape
      :type: Tuple[int, int]


      Returns (L,L), which is the shape of self.value and self.cholesky.


   .. py:property:: cholesky
      :type: tensorflow.Tensor


      The (lower triangular) Cholesky decomposition of the covariance matrix.


   .. py:property:: value

      The covariance matrix, shape (L,L).


   .. py:property:: value_to_broadcast

      The covariance matrix, shape (L,1,L,1) ready to broadcast.


   .. py:method:: value_times_eye(N)

      The cartesian product variance[:L, :L] * eye[:N, :N], transposed.

      :param N: The dimension of the identity matrix we are multiplying by.

      Returns: An [:L, :N, :L, :N] Tensor, after transposition.



