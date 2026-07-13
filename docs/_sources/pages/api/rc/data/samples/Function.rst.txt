:html_theme.sidebar_secondary.remove:

rc.data.samples.Function
========================


.. module:: rc.data.samples

.. toctree::
   :hidden:

   /pages/api/rc/data/samples/Function.repo
   /pages/api/rc/data/samples/Function.collection
   /pages/api/rc/data/samples/Function.un_rotate_folds

.. py:class:: rc.data.samples.Function(root: rc.base.Path | str, doe: DOE, function_vector: functions.Vector, N: int, M: int, noise_variance: GaussianNoise, ext: str | None = None, overwrite_existing: bool = False, **kwargs: rc.base.Any)

   Sample a ``task.function.Vector``.

   Construct a Repo by sampling a function over a DOE.

   :param root: The folder under which the Repo will sit.
   :param doe: An experimental design for the sample inputs.
   :param function_vector: A vector function.
   :param N: The number of samples (rows) in the sample.
   :param M: The input dimensionality (columns).
   :param noise_magnitude: The (L,L) homoskedastic ``GaussianNoise.Variance``.
   :param ext: Unless None, the repo name is suffixed by ``.[ext]``.
   :param overwrite_existing: Whether to overwrite an existing Repo.
   :param \*\*kwargs: MetaData passed straight to doe.

Properties
----------

.. autoapisummary::

   rc.data.samples.Function.repo


Methods
-------

.. autoapisummary::

   rc.data.samples.Function.collection
   rc.data.samples.Function.un_rotate_folds


