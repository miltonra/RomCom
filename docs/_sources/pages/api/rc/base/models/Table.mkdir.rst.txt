:html_theme.sidebar_secondary.remove:

rc.base.models.Table.mkdir
==========================

.. py:method:: rc.base.models.Table.mkdir(path: rc.base.definitions.PathLike) -> rc.base.definitions.Path
   :classmethod:


   Create ``path.parent``, with a subfolder ``path`` if ``cls.ext == ''``.

   :param path: The folder to create, or a child file of the folder to create.

   Returns: ``path``.

