rc.base.models.DataBase.create
==============================

.. py:method:: rc.base.models.DataBase.create(path, **tables_and_meta)
   :classmethod:


   Create a ``DataBase`` in ``path``.

   :param path: The folder to store the ``DataBase`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
   :param \*\*tables_and_meta: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                               and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

   Returns: The ``DataBase`` created.

