rc.data.designs.CoordDesign.unjoinHeads
=======================================

.. py:method:: rc.data.designs.CoordDesign.unjoinHeads(src, dst)
   :classmethod:


   Explode ``src.heads`` into multi-level headed  ``dst.csv``, overwriting.
   Explosion is from the left, so ``a│b│c`` becomes the 3-level header ``(a,b,c)``.

   :param src: The source Table.
   :param dst: The destination Path, overwritten if existing. A ``.csv`` extension is implicitly appended.

   Returns: ``dst``, now containing the unjoined ``dst.csv``.

