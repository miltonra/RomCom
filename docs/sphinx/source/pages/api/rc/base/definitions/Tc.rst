rc.base.definitions.Tc
======================

.. py:class:: rc.base.definitions.Tc

   Extended PyTorch types and constants. This class should never be instantiated or subclassed.

   .. attribute:: DType

      ``tc.dtype``.

   .. attribute:: Tensor

      ``tc.Tensor``.

   .. attribute:: Vector

      Column vector, first order Tensor ``.shape = (i,1)``.

   .. attribute:: CoVector = Tensor

      Row vector, first order Tensor ``.shape = (1,j)``.

   .. attribute:: Matrix = Tensor

      Second order Tensor ``.shape = (i,j)``.

   .. attribute:: BatchVector = Tensor

      Vector ``.shape = (...,i,1)``.

   .. attribute:: BatchCoVector = Tensor

      CoVector ``.shape = (...,1,j)``.

   .. attribute:: BatchMatrix = Tensor

      Matrix ``.shape = (...,i,j)``.

