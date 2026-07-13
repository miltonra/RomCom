:html_theme.sidebar_secondary.remove:

rc.base.models.Table.create
===========================

.. py:method:: rc.base.models.Table.create(path: rc.base.definitions.PathLike, df: rc.base.definitions.DataFrame | rc.base.definitions.Self) -> rc.base.definitions.Self
   :classmethod:


   Create a Table at ``path``, overwriting.

   :param path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
   :param df: The DataFrame to store.

   Returns: The Table created.
   Raises: AssertionError if ``df`` is unacceptable.

