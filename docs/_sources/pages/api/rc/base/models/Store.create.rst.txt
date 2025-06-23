rc.base.models.Store.create
===========================

.. py:method:: rc.base.models.Store.create(path)
   :classmethod:

   :abstractmethod:


   Create a folder (and its parents) if it doesn't already exist.

   Overrides should create and return an instance of ``cls``.

   :param path: Where to create the folder. If ``cls.ext != ''``, the parent folder of ``path`` is created.

   :returns: ``path`` with extension ``f'.{cls.ext}'``.

   :raises FileExistsError: If attempting to overwrite a file with a folder.

