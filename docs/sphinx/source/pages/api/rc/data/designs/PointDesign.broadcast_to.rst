rc.data.designs.PointDesign.broadcast_to
========================================

.. py:method:: rc.data.designs.PointDesign.broadcast_to(target_shape, is_diagonal = True)

   Broadcast ``self``.

   :param target_shape: The shape to broadcast to.
   :param is_diagonal: Whether to zero the off-diagonal elements of a square matrix.

   Returns: ``self``.

   :raises IndexError: If broadcasting is impossible.

