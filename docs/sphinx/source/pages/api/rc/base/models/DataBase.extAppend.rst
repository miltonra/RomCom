:html_theme.sidebar_secondary.remove:

rc.base.models.DataBase.extAppend
=================================

.. py:method:: rc.base.models.DataBase.extAppend(path: rc.base.definitions.PathLike) -> rc.base.definitions.Path
   :classmethod:


   Append ``cls.ext`` to ``path.name``.

   :param path: The path to append ``cls.ext`` to.

   Returns: ``Path(path)`` with ``cls.ext`` appended.

