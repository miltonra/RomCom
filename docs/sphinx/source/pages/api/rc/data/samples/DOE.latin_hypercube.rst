:html_theme.sidebar_secondary.remove:

rc.data.samples.DOE.latin_hypercube
===================================

.. py:method:: rc.data.samples.DOE.latin_hypercube(N: int, M: int, is_centered: bool = True, **kwargs)
   :staticmethod:


   Latin Hypercube DOE.

   :param N: The number of samples (rows).
   :param M: The of input dimensions (columns).
   :param is_centered: Boolean ordinate whether to centre each sample in its Latin Hypercube cell.
                       Default is False, which locates the sample randomly within its cell.
   :param kwargs: Passed directly to
                  `scipy.stats.qmc.LatinHypercube <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html>`_.

   Returns: An (N,M) matrix of N samples of dimension M.

