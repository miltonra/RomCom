rc.data.normalizers.Normalizer.delete
=====================================

.. py:method:: rc.data.normalizers.Normalizer.delete(path, ignoreErrors = False)
   :classmethod:


   Delete all DataBase files in ``path``, retaining ``path`` and any other files it contains.

   If you wish to delete ``path`` entirely, use ``Store.delete(path)`` instead.

   :param path: Path to the DataBase to delete.
   :param ignoreErrors: Whether to raise any ``FileNotFoundError`` s encountered.

   Returns: ``path``, which still exists.
   Raises: FileNotFoundError if ``path`` is not a folder, regardless of ``ignoreErrors``.

