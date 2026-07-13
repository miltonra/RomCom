:html_theme.sidebar_secondary.remove:

rc.data.samples.DOE.space_filling_test
======================================

.. py:method:: rc.data.samples.DOE.space_filling_test(X: rc.base.Np.Matrix, o: int) -> dict[str, float]
   :staticmethod:


   Test whether ``X`` is a space-filling design matrix,
   by finding the distance to the nearest point in ``X`` for ``o`` test points.

   :param X: An (N,M) design matrix.
   :param o: The number of test points used to assess whether ``X`` is a space-filling design.

   Returns: A dict of six measures: The theoretical hard upper bound, expected upper bound and
       expected lower bound for a perfectly space-filling design matrix,
       followed by the max, mean and SD of the distance-to-nearest-in-X over the o test points.

