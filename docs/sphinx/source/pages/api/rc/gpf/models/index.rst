rc.gpf.models
=============

.. py:module:: rc.gpf.models

.. autoapi-nested-parse::

   Contains extensions to gpflow.models.



Classes
-------

.. autoapisummary::

   rc.gpf.models.MOGPR


Module Contents
---------------

.. py:class:: MOGPR(data, kernel, mean_function = None, noise_variance = 1.0)

   Bases: :py:obj:`gpflow.models.model.GPModel`, :py:obj:`gpflow.models.training_mixins.InternalDataTrainingLossMixin`

   .. autoapi-inheritance-diagram:: rc.gpf.models.MOGPR
      :parts: 1


   Gaussian Process Regression.

   This is a vanilla implementation of MOGP regression with a Gaussian
   likelihood.  Multiple columns of Y are treated independently.

   The log likelihood of this model is given by

   .. math::
      \log p(Y \,|\, \mathbf f) =
           \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

   To train the model, we maximise the log _marginal_ likelihood
   w.r.t. the likelihood variance and kernel hyperparameters theta.
   The marginal likelihood is found by integrating the likelihood
   over the prior, and has the form

   .. math::
      \log p(Y \,|\, \sigma_n, \theta) =
           \mathcal N(Y \,|\, 0, \mathbf{KXX} + \sigma_n^2 \mathbf{I})


   .. py:property:: M

      The input dimensionality.


   .. py:property:: L

      The output dimensionality.


   .. py:method:: log_marginal_likelihood()

      Computes the log marginal likelihood.

      .. math::
          \log p(Y | \theta).




   .. py:method:: predict_f(Xnew, full_cov = False, full_output_cov = False)

      This method computes predictions at X \in R^{N \x D} input points

      .. math::
          p(F* | Y)

      where F* are points on the MOGP at new data points, Y are noisy observations at training data points.
      Note that full_cov => full_output_cov (regardless of the ordinate given for full_output_cov), to avoid ambiguity.



