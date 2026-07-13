:html_theme.sidebar_secondary.remove:

rc.base.models.Table.extAppend
==============================

.. py:method:: rc.base.models.Table.extAppend(path: rc.base.definitions.PathLike) -> rc.base.definitions.Path
   :classmethod:


   Append ``cls.ext`` to ``path.name``.

   :param path: The path to append ``cls.ext`` to.

   Returns: ``Path(path)`` with ``cls.ext`` appended.

