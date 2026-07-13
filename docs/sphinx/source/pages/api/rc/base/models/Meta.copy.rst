:html_theme.sidebar_secondary.remove:

rc.base.models.Meta.copy
========================

.. py:method:: rc.base.models.Meta.copy(src: rc.base.definitions.Self, dst: rc.base.definitions.PathLike) -> rc.base.definitions.Self
   :classmethod:


   Copy ``src`` to ``dst``, overwriting.

   :param src: The source Meta.
   :param dst: The destination Path, overwritten if existing.
               A ``.json`` extension is implicitly appended.

   Returns: The Meta now stored at ``dst.json``.

