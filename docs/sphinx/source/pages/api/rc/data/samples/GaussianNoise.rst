rc.data.samples.GaussianNoise
=============================


.. module:: rc.data.samples

.. toctree::
   :hidden:

   /pages/api/rc/data/samples/GaussianNoise.Variance

.. py:class:: rc.data.samples.GaussianNoise(N, variance)

   Sample multivariate, zero-mean Gaussian noise.

   Generate N samples of L-dimensional Gaussian noise, sampled from :math:`\mathsf{N}[0,variance]`.

   :param N: Number of samples (rows).
   :param variance: (L,L) covariance matrix for homoskedastic noise.

Classes
-------

.. autoapisummary::

   rc.data.samples.GaussianNoise.Variance


