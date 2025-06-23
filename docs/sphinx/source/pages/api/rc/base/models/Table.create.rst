rc.base.models.Table.create
===========================

.. py:method:: rc.base.models.Table.create(path, data = None, index = None, columns = None, dtype = None, copy = None, **metadata)
   :classmethod:


   Create a ``Table`` at ``path``, overwriting.

   :param path: The ``Path`` to store this DataTable, overwritten if existing.
                A ``.csv`` extension is automatically appended.
   :param data: The data to store. If ``None``, a ``Pd.DataFrame`` is read from ``.csv``.
                See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
   :param index: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
   :param columns: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
   :param dtype: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
   :param copy: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
   :param \*\*metadata: MetaData passed to
                        `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_
                        or
                        `pd.DataFrame.to_csv`_.

   Returns: The ``DataTable`` created.

   .. _pd.DataFrame.to_csv: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html

