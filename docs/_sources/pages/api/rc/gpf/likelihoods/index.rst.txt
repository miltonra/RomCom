rc.gpf.likelihoods
==================

.. py:module:: rc.gpf.likelihoods

.. autoapi-nested-parse::

   Contains extensions to gpflow.likelihoods.



Classes
-------

.. autoapisummary::

   rc.gpf.likelihoods.MOGaussian


Module Contents
---------------

.. py:class:: MOGaussian(variance, **kwargs)

   Bases: :py:obj:`gpflow.likelihoods.QuadratureLikelihood`

   .. autoapi-inheritance-diagram:: rc.gpf.likelihoods.MOGaussian
      :parts: 1


   A non-diagonal, multivariate likelihood, extending gpflow. The code is the multivariate version of gf.likelihoods.Gaussian.

   The Gaussian likelihood is appropriate where uncertainties associated with
   the data are believed to follow a normal distribution, with constant
   variance.

   Very small uncertainties can lead to numerical instability during the
   optimization process. A lower bound of 1e-3 is therefore imposed on the
   likelihood Variance.cholesky_diagonal elements by default.


   .. py:method:: N(data)

      The number of samples in data, assuming the last 2 dimensions have been concatenated to LN.



   .. py:method:: split_axis_shape(data)

      Split the final data axis length LN into the pair (L,N).



