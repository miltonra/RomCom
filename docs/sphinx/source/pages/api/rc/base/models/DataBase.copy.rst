:html_theme.sidebar_secondary.remove:

rc.base.models.DataBase.copy
============================

.. py:method:: rc.base.models.DataBase.copy(src: rc.base.definitions.Self, dst: rc.base.definitions.PathLike) -> rc.base.definitions.Self
   :classmethod:


   Copy ``src`` to ``dst``, overwriting any files in common.

   :param src: The source *DataBase*.
   :param dst: The destination Path, which may or may not exist.

   Returns: The *DataBase* now stored in ``dst``.

