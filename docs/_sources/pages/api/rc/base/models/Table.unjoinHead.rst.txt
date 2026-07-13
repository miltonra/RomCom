:html_theme.sidebar_secondary.remove:

rc.base.models.Table.unjoinHead
===============================

.. py:method:: rc.base.models.Table.unjoinHead(src: Table, dst: rc.base.definitions.PathLike) -> rc.base.definitions.Path
   :classmethod:


   Unjoin ``src.head`` into multi-headed ``dst.csv``, overwriting.
   Unjoining is from the left, so ``a│b│c`` becomes the triple head ``(a,b,c)``.
   The first column is presumed to be an index.
   Every other column must produce the same headcount (number of heads).

   :param src: The source Table.
   :param dst: The destination Path, overwritten if existing. A ``.csv`` extension is implicitly appended.

   Returns: ``dst``, now containing the unjoined ``dst.csv``.
   Raises: AssertionError if ``src.head`` cannot be unjoined due to inconsistent headcount (rows).

