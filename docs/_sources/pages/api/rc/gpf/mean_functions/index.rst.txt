rc.gpf.mean_functions
=====================

.. py:module:: rc.gpf.mean_functions

.. autoapi-nested-parse::

   Mean functions for gpf - i.e. Gaussian prior predictions.



Classes
-------

.. autoapisummary::

   rc.gpf.mean_functions.MOMeanFunction


Module Contents
---------------

.. py:class:: MOMeanFunction(output_dim, mean_functions = Zero())

   Bases: :py:obj:`gpflow.mean_functions.MeanFunction`

   .. autoapi-inheritance-diagram:: rc.gpf.mean_functions.MOMeanFunction
      :parts: 1


   Mean functions for MOGPR. Basically a wrapper for a Sequence of gpflow.mean_functions.MeanFunctions, one for each output_dim.
   These functions constitute the prior mean predictions f(x) in the absence of any training data.


   .. py:property:: output_dim

      Also known as L.


   .. py:property:: functions

      The sequence of functions defining this MOMeanFunction.


   .. py:method:: __call__(X)

      Given N samples in X, returns an output_dim * N vector of flatten(functions(X)).



