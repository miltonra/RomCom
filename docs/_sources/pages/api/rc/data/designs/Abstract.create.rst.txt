:html_theme.sidebar_secondary.remove:

rc.data.designs.Abstract.create
===============================

.. py:method:: rc.data.designs.Abstract.create(path: rc.base.PathLike, df: rc.base.DataFrame | rc.base.Table, isFat: bool = False) -> rc.base.Self
   :classmethod:

   :abstractmethod:


   Create a Table at ``path``, overwriting.

   :param path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
   :param df: The DataFrame to store.

   Returns: The Table created.
   Raises: AssertionError if ``df`` is unacceptable.

