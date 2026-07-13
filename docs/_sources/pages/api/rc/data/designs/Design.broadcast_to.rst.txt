:html_theme.sidebar_secondary.remove:

rc.data.designs.Design.broadcast_to
===================================

.. py:method:: rc.data.designs.Design.broadcast_to(target_shape: tuple[int, int], is_diagonal: bool = True) -> rc.base.definitions.Self

   Broadcast ``self``.

   :param target_shape: The shape to broadcast to.
   :param is_diagonal: Whether to zero the off-diagonal elements of a square matrix.

   Returns: ``self``.

   Raises: IndexError if broadcasting is impossible.

