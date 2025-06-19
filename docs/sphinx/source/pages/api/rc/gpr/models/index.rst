rc.gpr.models
=============

.. py:module:: rc.gpr.models

.. autoapi-nested-parse::

   Contains the MOGP class implementing Gaussian Process Regression.



Classes
-------

.. autoapisummary::

   rc.gpr.models.Likelihood
   rc.gpr.models.GPR
   rc.gpr.models.MOGP


Module Contents
---------------

.. py:class:: Likelihood(parent, read_data = False, **kwargs)

   Bases: :py:obj:`rc.base.models.DataBase`

   .. autoapi-inheritance-diagram:: rc.gpr.models.Likelihood
      :parts: 1


   ``NamedTables(NamedTuple)`` in a folder alongside ``Meta``. Abstract base class for any model.

   ``DataBase`` subclasses must be implemented according to the template (copy and paste it)::

       class MyDataBase(DataBase):

           class NT(NamedTuple):

               names[i]: Table | Matrix | MetaData = pd.DataFrame(defaults[names[i]].pd)   #: Comment
               ...

               def __call__(self, name: str) -> Table | Matrix | MetaData:
                   """ Returns the Table named ``name``."""
                   return getattr(self, name)


           options: NamedTables[MetaData] = NamedTables(**{name: table.options for name, table in {}.items()})
           """ Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
           Override as necessary for bespoke ``Table.options``.
           Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
           the remainder populate ``self[i].options.read``."""

           defaultMetaData: MetaData = {'Tables': Tables.options._asdict()}


   .. py:class:: Data

      Bases: :py:obj:`rc.base.models.Tables`

      .. autoapi-inheritance-diagram:: rc.gpr.models.Likelihood.Data
         :parts: 1


      The Data set of a MOGP.


      .. py:property:: NamedTuple
         :type: rc.base.definitions.Type[rc.base.definitions.NamedTuple]

         :classmethod:


         The NamedTuple underpinning this Data set.



   .. py:method:: calibrate(**kwargs)

      Merely sets the trainable data.



.. py:class:: GPR(name, fold, is_read, is_covariant, is_isotropic, kernel_parameters = None, likelihood_variance = None)

   Bases: :py:obj:`rc.base.models.DataBase`

   .. autoapi-inheritance-diagram:: rc.gpr.models.GPR
      :parts: 1


   Interface to a Gaussian Process.


   .. py:class:: Data

      Bases: :py:obj:`rc.base.models.Tables`

      .. autoapi-inheritance-diagram:: rc.gpr.models.GPR.Data
         :parts: 1


      The Data set of a MOGP.


      .. py:property:: NamedTuple
         :type: rc.base.definitions.Type[rc.base.definitions.NamedTuple]

         :classmethod:


         The NamedTuple underpinning this Data set.



   .. py:property:: META
      :type: rc.base.definitions.Dict[str, rc.base.definitions.Any]

      :classmethod:

      :abstractmethod:


      Hyper-parameter optimizer meta


   .. py:property:: KERNEL_FOLDER_NAME
      :type: str

      :classmethod:


      The name of the folder where kernel data are stored.


   .. py:property:: fold
      :type: rc.data.models.Fold


      The parent fold.


   .. py:property:: implementation
      :type: rc.base.definitions.Tuple[rc.base.definitions.Any, Ellipsis]

      :abstractmethod:


      The implementation of this MOGP in GPFlow.
      If ``noise_variance.shape == (1,L)`` an L-tuple of kernels is returned.
      If ``noise_variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.


   .. py:property:: L
      :type: int


      The output (Y) dimensionality.


   .. py:property:: M
      :type: int


      The input (X) dimensionality.


   .. py:property:: N
      :type: int


      The the number of training samples.


   .. py:property:: X
      :type: rc.base.definitions.Any

      :abstractmethod:


      The implementation training inputs.


   .. py:property:: Y
      :type: rc.base.definitions.Any

      :abstractmethod:


      The implementation training outputs.


   .. py:property:: K_cho
      :type: rc.base.definitions.Union[NP.Matrix, TF.Tensor]

      :abstractmethod:


      The Cholesky decomposition of the LNxLN noisy kernel(X, X) + likelihood.variance. Shape is (LN, LN) if self.kernel.is_covariant, else (L,N,N).


   .. py:property:: K_inv_Y
      :type: rc.base.definitions.Union[NP.Matrix, TF.Tensor]

      :abstractmethod:


      The LN-Vector, which pre-multiplied by the LoxLN kernel k(x, X) gives the Lo-Vector predictive mean f(x).
      Shape is (L,1,N).
      Returns: ChoSolve(self.K_cho, self.Y)


   .. py:method:: predict(x, y_instead_of_f = True)
      :abstractmethod:


      Predicts the response to input X.

      :param x: An (o, M) design Matrix of inputs.
      :param y_instead_of_f: True to include noise in the variance of the result.

      Returns: The distribution of y or f, as a pair (mean (o, L) Matrix, std (o, L) Matrix).



   .. py:method:: predict_df(x, y_instead_of_f = True, is_normalized = True)

      Predicts the response to input X.

      :param x: An (o, M) design Matrix of inputs.
      :param y_instead_of_f: True to include noise in the variance of the result.
      :param is_normalized: Whether the results are normalized or not.

      Returns: The distribution of y or f, as a dataframe with M+L+L columns of the form (X, Mean, Predictive Std).



   .. py:method:: predict_gradient(x, y_instead_of_f = True)
      :abstractmethod:


      Predicts the gradient GP dy/dx (or df/dx) where ``self`` is the GP for y(x).

      :param x: An (o, M) design Matrix of inputs.
      :param y_instead_of_f: True to include noise in the variance of the result.

      Returns: The distribution of dy/dx or df/dx, as a pair (mean (o, L, M), cov (o, L, M, O, l, m)) if ``self.likelihood.is_covariant``,
          else (mean (o, L, M), cov (o, O, L, M)).



   .. py:method:: test()

      Tests the MOGP on the test data in self._fold.test_data. Test results comprise three values for each output at each sample:
      The mean prediction, the std error of prediction and the Z score of prediction (i.e. error of prediction scaled by std error of prediction).

      Returns: The test_data results as a DataTable backed by MOGP.test_result_csv.



   .. py:method:: broadcast_parameters(is_covariant, is_isotropic)

      Broadcast the data of the MOGP (including kernels) to higher dimensions.
      Shrinkage raises errors, unchanged dimensions silently do nothing.

      :param is_covariant: Whether the outputs will be treated as dependent.
      :param is_isotropic: Whether to restrict the kernel to be isotropic.

      Returns: ``self``, for chaining calls.



.. py:class:: MOGP(name, fold, is_read, is_covariant, is_isotropic, kernel_parameters = None, likelihood_variance = None)

   Bases: :py:obj:`GPR`

   .. autoapi-inheritance-diagram:: rc.gpr.models.MOGP
      :parts: 1


   Implementation of a Gaussian Process.


   .. py:property:: META
      :type: rc.base.definitions.Dict[str, rc.base.definitions.Any]

      :classmethod:


      Hyper-parameter optimizer meta


   .. py:property:: implementation
      :type: rc.base.definitions.Tuple[rc.base.definitions.Any, Ellipsis]


      The implementation of this MOGP in GPFlow.
      If ``noise_variance.shape == (1,L)`` an L-tuple of kernels is returned.
      If ``noise_variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.


   .. py:method:: calibrate(method = 'L-BFGS-B', **kwargs)

      Optimize the MOGP hyper-data.

      :param method: The optimization algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
      :param kwargs: A Dict of implementation-dependent optimizer meta, following the format of GPR.META.
                     MetaData for the kernel should be passed as kernel={see kernel.META for format}.
                     MetaData for the likelihood should be passed as likelihood={see likelihood.META for format}.



   .. py:method:: predict(X, y_instead_of_f = True)

      Predicts the response to input X.

      :param x: An (o, M) design Matrix of inputs.
      :param y_instead_of_f: True to include noise in the variance of the result.

      Returns: The distribution of y or f, as a pair (mean (o, L) Matrix, std (o, L) Matrix).



   .. py:method:: predict_gradient(x, y_instead_of_f = True)

      Predicts the gradient GP dy/dx (or df/dx) where ``self`` is the GP for y(x).

      :param x: An (o, M) design Matrix of inputs.
      :param y_instead_of_f: True to include noise in the variance of the result.

      Returns: The distribution of dy/dx or df/dx, as a pair (mean (o, L, M), cov (o, L, M, O, l, m)) if ``self.likelihood.is_covariant``,
          else (mean (o, L, M), cov (o, O, L, M)).



   .. py:property:: X
      :type: TF.Matrix


      The implementation training inputs as an (N,M) design matrix.


   .. py:property:: Y
      :type: TF.Matrix


      The implementation training outputs as an (N,L) design matrix.


   .. py:property:: K_cho
      :type: TF.Tensor


      The Cholesky decomposition of the LNxLN noisy kernel(X, X) + likelihood.variance. Shape is (LN, LN) if self.kernel.is_covariant, else (L,N,N).


   .. py:property:: K_inv_Y
      :type: TF.Tensor


      The LN-Vector, which pre-multiplied by the LoxLN kernel k(x, X) gives the Lo-Vector predictive mean f(x).
      Shape is (L,1,N).
      Returns: ChoSolve(self.K_cho, self.Y)


   .. py:method:: check_K_inv_Y(x)

      FOR TESTING PURPOSES ONLY. Should return 0 Vector (to within numerical error tolerance).

      :param x: An (o, M) matrix of inputs.

      Returns: Should return zeros((Lo)) (to within numerical error tolerance).



