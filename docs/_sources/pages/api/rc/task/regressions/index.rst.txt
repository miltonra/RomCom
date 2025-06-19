rc.task.regressions
===================

.. py:module:: rc.task.regressions

.. autoapi-nested-parse::

   **Functionality for linear regression**



Methods
---------

.. autoapisummary::

   rc.task.regressions.gls


Module Contents
---------------

.. py:function:: gls(X, y, cov_y, is_through_origin = False)

   `Generalized Least Squares <https://en.wikipedia.org/wiki/Generalized_least_squares>`_ linear regression.

   :param X: An (N,M) matrix of regression variables
   :param y: An (N,1) vector of observations.
   :param cov_y: The (N,N) covariance matrix of observations ``y``.
   :param is_through_origin: True to constrain to ``y(0)=0``

   Returns: A pair consisting of the (M+1,1) -- or (M,1) if ``is_through_origin`` -- regression coefficients and their covariance matrix,
       where the intercept is the first regression coefficient -- or absent if ``is_through_origin``.



