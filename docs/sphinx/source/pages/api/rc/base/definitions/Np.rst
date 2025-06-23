rc.base.definitions.Np
======================

.. py:class:: rc.base.definitions.Np

   Extended NumPy types and constants. This class should never be instantiated or subclassed.

   .. attribute:: DType

      ``np.dtype``.

   .. attribute:: Array

      ``np.ndarray``.

   .. attribute:: Tensor

      ``Array``.

   .. attribute:: Vector

      Column vector, first order Tensor ``.shape = (i,1)``.

   .. attribute:: CoVector = Tensor

      Row vector, first order Tensor ``.shape = (1,j)``.

   .. attribute:: Matrix = Tensor

      Second order Tensor ``.shape = (i,j)``.

