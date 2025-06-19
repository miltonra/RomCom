rc.gsa.base
===========

.. py:module:: rc.gsa.base

.. autoapi-nested-parse::

   Base Classes and basic functions underpinning GSA.



Classes
-------

.. autoapisummary::

   rc.gsa.base.Calibrator
   rc.gsa.base.Gaussian


Methods
---------

.. autoapisummary::

   rc.gsa.base.diag_det


Module Contents
---------------

.. py:function:: diag_det(tensor)

   Determinant of a diagonal tensor.

   :param tensor: Of shape ``[...,m]``. The last axis must contain the diagonal

   Returns: Tensor shaped ``[...]``.



.. py:class:: Calibrator

   Bases: :py:obj:`abc.ABC`

   .. autoapi-inheritance-diagram:: rc.gsa.base.Calibrator
      :parts: 1


   Interface to GSA calibrator


.. py:class:: Gaussian(mean, variance, is_variance_diagonal, ordinate = tf.constant(0, dtype=Float()), LBunch = 2)

   Encapsulates a Gaussian pdf. For numerical stability the 2 Pi factor is not included.


   .. py:property:: det
      :type: TF.Tensor


      The sqrt of the determinant of the Gaussian covariance.


   .. py:property:: pdf
      :type: TF.Tensor


      Calculate the Gaussian pdf from the output of Gaussian.


   .. py:method:: expand_dims(axes)

      Insert dimensions at the specified axes.

      :param axes: A sequence of dims to insert.

      Returns: ``self`` for chaining calls.



   .. py:method:: __truediv__(other)

      Divide this Gaussian pdf by denominator.

      :param other: The Gaussian to divide by.



