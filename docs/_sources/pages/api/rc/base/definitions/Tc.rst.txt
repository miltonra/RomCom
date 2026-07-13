:html_theme.sidebar_secondary.remove:

rc.base.definitions.Tc
======================


.. module:: rc.base.definitions

.. py:class:: rc.base.definitions.Tc

   Extended PyTorch types and constants. This Class should never be instantiated or SubClassed.

   .. attribute:: DType = tc.dtype

      

   .. attribute:: Tensor = tc.Tensor

      

   .. attribute:: Vector = Tensor[i,1]

      

   .. attribute:: CoVector = Tensor[1,j]

      

   .. attribute:: Matrix = Tensor[i,j]

      

   .. attribute:: BatchVector = Tensor[...,i,1]

      

   .. attribute:: BatchCoVector = Tensor[...,1,j]

      

   .. attribute:: BatchMatrix = Tensor[...,i,j]

      

