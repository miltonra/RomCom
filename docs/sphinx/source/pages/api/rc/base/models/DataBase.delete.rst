:html_theme.sidebar_secondary.remove:

rc.base.models.DataBase.delete
==============================

.. py:method:: rc.base.models.DataBase.delete(path: rc.base.definitions.PathLike) -> rc.base.definitions.Path
   :classmethod:


   Delete all *DataBase* files in ``path``, retaining ``path`` and any other files it contains.

   If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

   :param path: Path to the *DataBase* to delete.

   Returns: ``path``, which still exists.
   Raises: AssertionError if ``path`` is not a folder.

