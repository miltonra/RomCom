rc.data.functions.Scalar
========================

.. py:class:: rc.data.functions.Scalar(call, loc, scale, m, **kwargs)

   A scalar function ``scalar`` such that ``scalar(x, kwargs)`` calls
   ``self.call(self.loc + self.scale * x[:, :self.m], **(self.kwargs | kwargs)``.

