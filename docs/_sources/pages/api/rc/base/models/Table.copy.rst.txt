rc.base.models.Table.copy
=========================

.. py:method:: rc.base.models.Table.copy(src, dst)
   :classmethod:


   Copy ``src`` to ``dst``, overwriting.

   :param src: The source ``DataTable``.
   :param dst: The destination ``Path``, overwritten if existing.
               A ``.csv`` extension is automatically appended.

   Returns: The ``DataTable`` now stored at ``dst.csv``.

