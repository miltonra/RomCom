:html_theme.sidebar_secondary.remove:

rc.data.benchmarks.linspace
===========================

.. py:function:: rc.data.benchmarks.linspace(start: float, stop: float, shape: rc.base.Sequence[int]) -> rc.base.Np.Matrix

   A multi-dimensional version of ``np.linspace``, distributing values throughout ``shape``.

   :param start: Start value, which will be returned in ``linspace(...)[0,...,0]``.
   :param stop: Stop value, which will be returned in ``linspace(...)[-1,...,-1]``.
   :param shape: The ``linspace.shape`` to return.

   Returns: ``np.reshape(np.linspace(start, stop, int(np.prod(shape)), endpoint=True), shape)``.

