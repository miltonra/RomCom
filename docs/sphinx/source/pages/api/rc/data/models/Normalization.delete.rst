rc.data.models.Normalization.delete
===================================

.. py:method:: rc.data.models.Normalization.delete(path)
   :classmethod:


   Delete all ``DataBase`` files in ``path``, retaining ``path`` and any other files it contains.

   If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

   :param path: ``Path`` to the ``DataBase`` to delete.

   Returns: ``path``, which still exists.

