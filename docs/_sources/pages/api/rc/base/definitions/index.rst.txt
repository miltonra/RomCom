rc.base.definitions
===================

.. py:module:: rc.base.definitions

.. autoapi-nested-parse::

   Type and constant definitions.

   All modules of RomCom ``import *`` from this module, so all types and constants in this module are referenced
   without adornment throughout RomCom.



Properties
-----------

.. autoapisummary::

   rc.base.definitions.zero


Classes
-------

.. autoapisummary::

   rc.base.definitions.Pd
   rc.base.definitions.Np
   rc.base.definitions.Tc


Module Contents
---------------

.. py:data:: zero
   :type:  float
   :value: 1e-64


   Tolerance when testing floats for equality.

.. py:class:: Pd

   Extended Pandas types and constants.

   .. attribute:: DataFrame

      pd.DataFrame.

   .. attribute:: Index

      pd.Index.

   .. attribute:: MultiIndex

      pd.MultiIndex.


.. py:class:: Np

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


.. py:class:: Tc

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


