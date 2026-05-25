rc.data.benchmarks.categorized
==============================

.. py:function:: rc.data.benchmarks.categorized(x, f, p)

   A single-output test function, categorized by ``f`` and ``p``.

   :param x: ``(n, m)`` design matrix of continuous inputs.
   :param f: The function category: 0 for ``ishigami``, 1 for ``sobolG``, 2 for ``oakley2004``.
   :param p: The parameter category: 0 for ``standard/weak5_2/lin``, 1 for ``balanced/strong5_2/quad``,
             2 for ``sin/strong5_4/rev``.

   Returns: The categorized function value evaluated at ``x,f,p``.

