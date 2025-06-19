rc.gsa.calibrators
==================

.. py:module:: rc.gsa.calibrators

.. autoapi-nested-parse::

   Contains the calculation of a single closed Sobol index without storing it.



Classes
-------

.. autoapisummary::

   rc.gsa.calibrators.ClosedSobol
   rc.gsa.calibrators.ClosedSobolWithError
   rc.gsa.calibrators.ClosedSobolWithRotation


Module Contents
---------------

.. py:class:: ClosedSobol(gp, **kwargs)

   Bases: :py:obj:`gf.Module`, :py:obj:`rc.gsa.base.Calibrator`

   .. autoapi-inheritance-diagram:: rc.gsa.calibrators.ClosedSobol
      :parts: 1


   Calculates closed Sobol Indices.


   .. py:property:: META
      :type: rc.base.definitions.Dict[str, rc.base.definitions.Any]

      :classmethod:


      Default calculation meta.

      Returns: An empty dictionary.


   .. py:method:: marginalize(m)

      Calculate everything.
      :param m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].

      Returns: The Sobol ClosedSobol of m.



.. py:class:: ClosedSobolWithError(gp, **kwargs)

   Bases: :py:obj:`ClosedSobol`

   .. autoapi-inheritance-diagram:: rc.gsa.calibrators.ClosedSobolWithError
      :parts: 1


   Calculates closed Sobol Indices with Errors.


   .. py:property:: META
      :type: rc.base.definitions.Dict[str, rc.base.definitions.Any]

      :classmethod:


      Default calculation meta. ``is_T_partial`` forces W[Mm] = W[MM] = 0.

      :returns: If True this effectively asserts the full ['M'] model is variance free, so WmM is not calculated or returned.
      :rtype: is_T_partial


   .. py:method:: marginalize(m)

      Calculate everything.
      :param m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].

      Returns: The Sobol ClosedSobol of m, with errors (T and W).



.. py:class:: ClosedSobolWithRotation(gp, **kwargs)

   Bases: :py:obj:`ClosedSobol`

   .. autoapi-inheritance-diagram:: rc.gsa.calibrators.ClosedSobolWithRotation
      :parts: 1


   Encapsulates the calculation of closed Sobol indices with a rotation U = Theta X.


