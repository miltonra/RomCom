:html_theme.sidebar_secondary.remove:

rc.data.benchmarks.Scalar
=========================


.. module:: rc.data.benchmarks

.. py:class:: rc.data.benchmarks.Scalar(salib: rc.base.Callable[rc.base.Np.Matrix, float], loc: rc.base.Np.Vector, scale: rc.base.Np.Vector, m: int, **params: float | rc.base.Np.Array | rc.base.Sequence[float | rc.base.Np.Array])

   A scalar function ``scalar`` such that ``scalar(x, params)`` calls
   ``self.salib(self.loc + self.scale * x[:, :self.m], **(self.params | params)``.

   A scalar function, which calls ``call(loc + scale * x[:, :m], **params)``.

   :param salib: The SALib function called.
   :param loc: Input offset.
   :param scale: Input scale.
   :param m: The number of input dimensions.
   :param \*\*params: Function data applied to call.

