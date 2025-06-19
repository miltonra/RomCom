rc.gpf.kernels
==============

.. py:module:: rc.gpf.kernels

.. autoapi-nested-parse::

   Contains extensions to gpflow.kernels.



Classes
-------

.. autoapisummary::

   rc.gpf.kernels.MOStationary
   rc.gpf.kernels.RBF


Module Contents
---------------

.. py:class:: MOStationary(variance, lengthscales, name='Kernel', active_dims=None)

   Bases: :py:obj:`gpflow.kernels.AnisotropicStationary`, :py:obj:`gpflow.kernels.Kernel`

   .. autoapi-inheritance-diagram:: rc.gpf.kernels.MOStationary
      :parts: 1


   Base class for stationary kernels, i.e. kernels that only
   depend on

       d = x - x'

   Derived classes should implement K_d(self, d): Returns the kernel evaluated
   on d, which is the pairwise difference matrix, scaled by the lengthscale
   parameter ℓ (i.e. [(X - X2ᵀ) / ℓ]). The last axis corresponds to the
   input dimension.


   .. py:property:: lengthscales_neat

      The kernel lengthscales as an (L,M) matrix.


   .. py:method:: K_diag(X)

      The kernel diagonal.

      :param X: An (N,M) Tensor.

      Returns: An (L, N, L, N) Tensor.



   .. py:method:: K_unit_variance(X, X2=None)

      The kernel with variance=ones(). This can be cached during optimisations where only the variance is trainable.

      :param X: An (n,M) Tensor.
      :param X2: An (N,M) Tensor.

      Returns: An (L,N,L,N) Tensor.



   .. py:method:: K_d_unit_variance(d)
      :abstractmethod:


      The kernel with variance=ones(). This can be cached during optimisations where only the variance is trainable.

      :param d: An (L,N,L,N,M) Tensor.

      Returns: An (L,N,L,N) Tensor.



   .. py:method:: K_d_apply_variance(K_d_unit_variance)

      Multiply the unit variance kernel by the kernel variance, and reshape.

      :param K_d_unit_variance: An (L,N,L,N) Tensor.

      Returns: An (LN,LN) Tensor



   .. py:method:: K_d(d)

      The kernel.

      :param d: An (L,N,L,N,M) Tensor.

      Returns: An (LN,LN) Tensor.



.. py:class:: RBF(variance, lengthscales, name='Kernel', active_dims=None)

   Bases: :py:obj:`MOStationary`

   .. autoapi-inheritance-diagram:: rc.gpf.kernels.RBF
      :parts: 1


   The radial basis function (RBF) or squared exponential kernel. The kernel equation is

       k(d) = σ² exp{-½ r²}

   where:
   r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
   σ²  is the variance parameter

   Functions drawn from a MOGP with this kernel are infinitely differentiable!


   .. py:method:: K_d_unit_variance(d)

      The kernel with variance=ones(). This can be cached during optimisations where only the variance is trainable.

      :param d: An (L,N,L,N,M) Tensor.

      Returns: An (L,N,L,N) Tensor.



