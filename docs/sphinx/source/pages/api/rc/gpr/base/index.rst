rc.gpr.base
===========

.. py:module:: rc.gpr.base

.. autoapi-nested-parse::

   Basic facilities for GPR Models.



Classes
-------

.. autoapisummary::

   rc.gpr.base.Variance


Module Contents
---------------

.. py:class:: Variance(sd, SDfloor = 0, corr = None)

   Bases: :py:obj:`rc.base.models.tc.nn.Module`

   .. autoapi-inheritance-diagram:: rc.gpr.base.Variance
      :parts: 1


   A (Co)Variance Matrix, efficiently represented.


   .. py:method:: forward(is_cho = True)

      Alias for ``self.__call__()``. Do not call.

      :param is_cho: False to return the variance, True to return its Cholesky lower triangle.

      Returns: ``tc.cholesky(variance) if is_cho else variance``.
          In either case the result is shaped (...,L,L), or (...,L,1) if diagonal (not covariant).



   .. py:method:: create(variance, SDfloor = 0)
      :classmethod:


      Create a Variance object from a diagonal (...,L,1) CoVector or square (...,L,L) Matrix.

      :param variance: The (co)variance matrix to be stored. If a BatchedCovector is supplied, this
                       represents a  diagonal variance matrix of dimension ``variance.shape[-2]``.
      :param SDfloor: The lower bound applicable to the Standard Deviation.

      Returns: The Variance object efficiently representing ``variance``.



