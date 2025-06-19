rc.task.scripts
===============

.. py:module:: rc.task.scripts

.. autoapi-nested-parse::

   **User interface to GPR, GSA and ROM**



Methods
---------

.. autoapisummary::

   rc.task.scripts.gpr
   rc.task.scripts.gsa


Module Contents
---------------

.. py:function:: gpr(name, repo, is_read, is_covariant, is_isotropic, ignore_exceptions = False, kernel_parameters = None, likelihood_variance = None, is_calibrated = True, is_tested = True, **kwargs)

   Undertake GPR on a Fold, or recursively across the Folds in a Repo.

   :param name: The MOGP name.
   :param repo: A Fold to house the MOGP, or a Repo containing Folds to house the GPs.
   :param is_read: If True, MOGP kernel data and likelihood_variance are read from ``fold.folder/name``, otherwise defaults are used.
                   If None, the nearest ancestor MOGP in the independence/isotropy hierarchy is recursively constructed from its nearest ancestor MOGP if necessary,
                   then read and broadcast available.
   :param is_covariant: Whether the outputs are independent of each other or not. If None, independent is task then broadcast to task dependent.
   :param is_isotropic: Whether the kernel is isotropic. If None, isotropic is task, then broadcast to task anisotropic.
   :param ignore_exceptions: Whether to continue when the MOGP provider throws an exception.
   :param kernel_parameters: If not None, this replaces the Kernel specified by the MOGP default.
   :param likelihood_variance: If not None this replaces the likelihood_variance specified by the MOGP default.
   :param is_calibrated: Whether to is_calibrated each MOGP.
   :param is_tested: Whether to test_data each MOGP.
   :param kwargs: A Dict of implementation-dependent passes straight to MOGP.Optimize().

   :returns: A list of the names of the GPs which have been constructed. The MOGP.Data are ``task.results.Aggregated`` over folds

   :raises FileNotFoundError: If repo is not a Fold, and contains no Folds.


.. py:function:: gsa(name, repo, is_covariant, is_isotropic, kinds = GSA.ALL_KINDS, m = -1, ignore_exceptions = False, is_error_calculated = False, **kwargs)

   Undertake GSA on a Fold, or recursively across the Folds in a Repo.

   :param name: The GSA name.
   :param repo: A Fold to house the GSA, or a Repo containing Folds to house the GSAs.
   :param is_covariant: Whether each output is independent of the other outputs. None results in variant (independent) followed by covariant (dependent).
   :param is_isotropic: Whether the kernel is isotropic. If None, isotropic is task, then broadcast to task anisotropic.
   :param kinds: Kind of index to calculate - first_order, closed or total. A Sequence of Kinds will be task consecutively.
   :param is_error_calculated: Whether to calculate variances (errors) on the Sobol indices.
                               The calculation of error is memory intensive, so leave this flag as False unless you are sure you need errors.
                               Furthermore, errors will only be calculated if the kernel of the GP has diagonal variance F.
   :param m: The dimensionality of the reduced model. For a single calculation it is required that ``0 < m < gp.M``.
             Any m outside this range results the Sobol index of each kind being calculated for all ``m in range(1, M+1)``.
   :param ignore_exceptions: Whether to ignore exceptions (e.g. file not found) when they are encountered, or halt.
   :param kwargs: A Dict of gsa calculation options, which updates the default gsa.undertake.calculation.META.

   :raises FileNotFoundError: If repo is not a Fold, and contains no Folds.

   :returns: A list of the calculation names which have been task, relative to repo.folder.


