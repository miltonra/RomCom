:html_theme.sidebar_secondary.remove:

rc.data.designs.Design.copy
===========================

.. py:method:: rc.data.designs.Design.copy(src: rc.base.definitions.Self, dst: rc.base.definitions.PathLike) -> rc.base.definitions.Self
   :classmethod:


   Copy ``src`` to ``dst``, overwriting.

   :param src: The source Table.
   :param dst: The destination Path, overwritten if existing.
               A ``.csv`` extension is implicitly appended.

   Returns: The Table now stored at ``dst``.

