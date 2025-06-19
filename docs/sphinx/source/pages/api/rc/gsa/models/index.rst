rc.gsa.models
=============

.. py:module:: rc.gsa.models

.. autoapi-nested-parse::

   **User interface for undertaking GSA on a MOGP**



Classes
-------

.. autoapisummary::

   rc.gsa.models.GSA
   rc.gsa.models.Sobol


Module Contents
---------------

.. py:class:: GSA(gp, kind, m = -1, is_error_calculated = False, **kwargs)

   Bases: :py:obj:`rc.base.models.DataBase`

   .. autoapi-inheritance-diagram:: rc.gsa.models.GSA
      :parts: 1


   Class encapsulating a generic Sobol calculation.


   .. py:class:: Kind

      Bases: :py:obj:`enum.IntEnum`

      .. autoapi-inheritance-diagram:: rc.gsa.models.GSA.Kind
         :parts: 1


      Enum to specify the kind of Sobol index to calculate.



   .. py:property:: calibrator
      :type: rc.gsa.base.Calibrator

      :abstractmethod:


      The object to do the calculations, whose marginalise(m) method returns a dict of results.


   .. py:method:: calibrate(method = None, **kwargs)

      Perform a generic GSA calculation. This method should be overriden by specific subclasses, and called via ``super()`` as a matter of priority.

      :param method: Not used.

      Returns: The results of the calculation, as a labelled dictionary of tf.Tensors.



   .. py:attribute:: meta

      The ``Meta`` currently in ``self``.


.. py:class:: Sobol(gp, kind, m = -1, is_error_calculated = False, **kwargs)

   Bases: :py:obj:`GSA`

   .. autoapi-inheritance-diagram:: rc.gsa.models.Sobol
      :parts: 1


   Class encapsulating a generic Sobol calculation.


   .. py:class:: Data

      Bases: :py:obj:`rc.base.models.Tables`

      .. autoapi-inheritance-diagram:: rc.gsa.models.Sobol.Data
         :parts: 1


      The Data set of a GSA.


      .. py:property:: NamedTuple
         :type: rc.base.definitions.Type[rc.base.definitions.NamedTuple]

         :classmethod:


         The NamedTuple underpinning this Data set.



   .. py:property:: META
      :type: rc.base.definitions.Dict[str, rc.base.definitions.Any]

      :classmethod:


      Default calculation meta. ``is_T_partial`` forces ``WmM = 0``.


   .. py:property:: calibrator
      :type: rc.gsa.calibrators.ClosedSobol


      The object to do the calculations, whose marginalise(m) method returns a dict of results.
      :param gp: The GPR underpinning the GSA.
      :param is_error_calculated: Whether to calculate the standard error of the GSA
      :param \*\*kwargs: MetaData passed straight to the Calibrator.

      Returns: The Calibrator.


