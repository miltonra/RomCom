rc.data.samples
===============

.. py:module:: rc.data.samples

.. autoapi-nested-parse::

   Functionality for Design of Experiments (DOE) and sampling.



Classes
-------

.. autoapisummary::

   rc.data.samples.DOE
   rc.data.samples.GaussianNoise
   rc.data.samples.Function


Methods
---------

.. autoapisummary::

   rc.data.samples.permute_axes
   rc.data.samples.PCA


Module Contents
---------------

.. py:class:: DOE

   Sampling methods for inputs.


   .. py:method:: latin_hypercube(N, M, is_centered = True, **kwargs)
      :staticmethod:


      Latin Hypercube DOE.

      :param N: The number of samples (rows).
      :param M: The of input dimensions (columns).
      :param is_centered: Boolean ordinate whether to centre each sample in its Latin Hypercube cell.
                          Default is False, which locates the sample randomly within its cell.
      :param kwargs: Passed directly to
                     `scipy.stats.qmc.LatinHypercube <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html>`_.

      Returns: An (N,M) matrix of N samples of dimension M.



   .. py:method:: full_factorial(N, M)
      :staticmethod:


      Full factorial DOE.

      :param N: The number of samples (rows).
      :param M: The of input dimensions (columns).

      Returns: An (N,M) matrix of N samples of dimension M.



   .. py:method:: space_filling_test(X, o)
      :staticmethod:


      Test whether ``X`` is a space-filling design matrix,
      by finding the distance to the nearest point in ``X`` for ``o`` test points.

      :param X: An (N,M) design matrix.
      :param o: The number of test points used to assess whether ``X`` is a space-filling design.

      Returns: A dict of six measures: The theoretical hard upper bound, expected upper bound and
          expected lower bound for a perfectly space-filling design matrix,
          followed by the max, mean and SD of the distance-to-nearest-in-X over the o test points.



.. py:class:: GaussianNoise(N, variance)

   Sample multivariate, zero-mean Gaussian noise.


   .. py:class:: Variance(L, magnitude, is_covariant = False, is_determined = True)

      An artificially generated (co)variance matrix for GaussianNoise, with a useful labelling scheme.


      .. py:property:: matrix
         :type: rc.base.Np.Matrix


         Variance as an (L,L) covariance matrix, suitable for constructing GaussianNoise.


      .. py:property:: meta
         :type: rc.base.Dict[str, rc.base.Any]


         Meta matching the initials in ``self.__format__()``.


      .. py:method:: __call__()

         Variance as an (L,L) covariance matrix, suitable for constructing GaussianNoise.
         The constructor generates the matrix (perhaps stochastically),
         so repeated calls to any methods produce identical Variance.



      .. py:method:: __format__(format_spec)

         The label for this Variance, to help name samples informatively.
         The description is ``d.`` (determined) or ``u.`` (undetermined),
         followed by ``v.`` (diagonal variance) or ``c.`` (non-diagonal covariance),
         followed by ``100 * self.magnitude:.2f``.




   .. py:method:: __call__(repo = None)

      Generate N samples of L-dimensional Gaussian noise, sampled from :math:`N[0,self.variance]`.
      The constructor generates the sample,
      so repeated calls to any method always refer to the same GaussianNoise.

      :param repo: An optional Repo which will have GaussianNoise added to Y in data.csv.

      Returns: An (N,L) noise matrix, where (L,L) is the shape of `self._variance`.



.. py:class:: Function(root, doe, function_vector, N, M, noise_variance, ext = None, overwrite_existing = False, **kwargs)

   Sample a ``task.function.Vector``.


   .. py:property:: repo
      :type: Repo


      The Repo containing the Function sample.


   .. py:method:: collection(sub_folder)

      Construct a Dict for task.results.Collect, with appropriate ``extra_columns``.

      :param folder: The folder under ``self.repo.folder`` housing the csvs to collect.

      Returns: The Dict for ``self.repo``.



   .. py:method:: un_rotate_folds()

      Create an un-rotated Fold in the Repo, with index ``K+1``.



.. py:function:: permute_axes(new_order)

   Provide a rotation matrix which reorders axes. Most use cases are to re-order input axes according to GSA.

   :param new_order: A Tuple or List containing a permutation of ``[0,...,M-1]``, for passing to ``np.transpose``.

   Returns: A rotation matrix which will reorder the axes to new_order. Returns ``None`` if ``new_order is None``.


.. py:function:: PCA(root, csv)

   Perform Principal Component Analysis on a Repo.

   :param root: The root folder.
   :param csv: The csv to read.
   :param normalization: An optional csv file to use for normalization.

   Returns: The folder written to, namely root``/PCA``


