rc.data.samples.GaussianNoise.Variance
======================================


.. module:: rc.data.samples

.. toctree::
   :hidden:

   /pages/api/rc/data/samples/GaussianNoise.Variance.matrix
   /pages/api/rc/data/samples/GaussianNoise.Variance.meta

.. py:class:: rc.data.samples.GaussianNoise.Variance(L, magnitude, is_covariant = False, is_determined = True)

   An artificially generated (co)variance matrix for GaussianNoise, with a useful labelling scheme.

   Instantiate an (L,L) GaussianNoise (co)variance matrix.

   :param L: Output dimensionality.
   :param magnitude: The StdDev of noise.
   :param is_covariant: True to create a diagonal variance matrix.
   :param is_determined: False to create a random symmetric matrix.

Properties
----------

.. autoapisummary::

   rc.data.samples.GaussianNoise.Variance.matrix
   rc.data.samples.GaussianNoise.Variance.meta


