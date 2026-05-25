rc.base.models.Store.copy
=========================

.. py:method:: rc.base.models.Store.copy(src, dst)
   :classmethod:

   :abstractmethod:


   Copy ``src`` to ``dst``, overwriting only files in common.

   Overrides should copy an instance of ``cls`` called ``src`` to ``Store.create(dst)``
   and return the copy.

   :param src: The source Path, which must be a folder or a file.
   :param dst: The destination Path, which may or may not exist.

   Returns: ``dst``.

   :raises FileNotFoundError: If ``src`` does not exist.
   :raises FileExistsError: If attempting to overwrite a file with a folder.

