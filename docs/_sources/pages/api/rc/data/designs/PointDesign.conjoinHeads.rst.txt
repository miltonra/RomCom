rc.data.designs.PointDesign.conjoinHeads
========================================

.. py:method:: rc.data.designs.PointDesign.conjoinHeads(src, dst, headcount = 2)
   :classmethod:


   Collapse multi-level headers in ``src`` to single-level ``dst.heads``,
   overwriting ``dst`` with a Table.
   Collapse is top down, so a 3-level header ``(a,b,c)`` becomes the single head ``a│b│c``.
   The first column is presumed to be an index. Any other column with empty header levels is dropped.

   :param src: The source Path. A ``.csv`` extension is implicitly appended.
   :param dst: The destination ``Table.path``, overwritten if existing.
               A ``.csv`` extension is implicitly appended.
   :param headcount: Counts the column heads (header rows) in ``src.csv``.

   Returns: The Table now stored at ``dst``.

