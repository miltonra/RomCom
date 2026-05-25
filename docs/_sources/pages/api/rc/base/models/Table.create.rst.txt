rc.base.models.Table.create
===========================

.. py:method:: rc.base.models.Table.create(path, data, **kwargs)
   :classmethod:


   Create a Table at ``path``, overwriting.

   :param path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
   :param data: The table to store.
   :param \*\*kwargs: KeywordArguments passed directly to `Pl.DataFrame(...)`_.

   Returns: The Table created.

   .. Pl.DataFrame(...): https://docs.pola.rs/api/python/dev/reference/dataframe/index.html

