rc.base.models.Meta.copy
========================

.. py:method:: rc.base.models.Meta.copy(src, dst)
   :classmethod:


   Copy ``src`` to ``dst``, overwriting.

   :param src: The source Meta.
   :param dst: The destination Path, overwritten if existing.
               A ``.json`` extension is implicitly appended.

   Returns: The Meta now stored at ``dst.json``.

