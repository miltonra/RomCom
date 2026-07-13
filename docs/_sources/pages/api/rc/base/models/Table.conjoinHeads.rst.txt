:html_theme.sidebar_secondary.remove:

rc.base.models.Table.conjoinHeads
=================================

.. py:method:: rc.base.models.Table.conjoinHeads(src: rc.base.definitions.PathLike, dst: rc.base.definitions.PathLike, headcount: int = 2) -> rc.base.definitions.Self
   :classmethod:


   Conjoin multiple heads in ``src`` to single head ``dst.head``,
   overwriting ``dst`` with a Table.
   Conjoining is top down, so the triple head ``(a,b,c)`` becomes the single head ``a│b│c``.
   The first column is presumed to be an index. Any other column with an empty head level is dropped.

   :param src: The source Path. A ``.csv`` extension is implicitly appended.
   :param dst: The destination ``Table.path``, overwritten if existing.
               A ``.csv`` extension is implicitly appended.
   :param headcount: Counts the number of heads in ``src.csv``.

   Returns: The Table now stored at ``dst``.

