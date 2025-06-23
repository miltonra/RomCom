rc.data.samples.permute_axes
============================

.. py:function:: rc.data.samples.permute_axes(new_order)

   Provide a rotation matrix which reorders axes. Most use cases are to re-order input axes according to GSA.

   :param new_order: A Tuple or List containing a permutation of ``[0,...,M-1]``, for passing to ``np.transpose``.

   Returns: A rotation matrix which will reorder the axes to new_order. Returns ``None`` if ``new_order is None``.

