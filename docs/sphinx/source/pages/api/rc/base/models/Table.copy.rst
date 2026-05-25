rc.base.models.Table.copy
=========================

.. py:method:: rc.base.models.Table.copy(src, dst)
   :classmethod:


   Copy ``src`` to ``dst``, overwriting.

   :param src: The source Table.
   :param dst: The destination Path, overwritten if existing.
               A ``.csv`` extension is implicitly appended.

   Returns: The Table now stored at ``dst``.

