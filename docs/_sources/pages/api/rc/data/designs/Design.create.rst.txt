:html_theme.sidebar_secondary.remove:

rc.data.designs.Design.create
=============================

.. py:method:: rc.data.designs.Design.create(path: rc.base.PathLike, df: rc.base.DataFrame | rc.base.Table, isFat: bool = False) -> rc.base.Self
   :classmethod:


   Create a Table at ``path``, overwriting.

   :param path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
   :param df: The DataFrame to store.
   :param isFat: Whether to  create a fat Design.

   Returns: The Table created.
   Raises: AssertionError if ``df`` is unacceptable.

