:html_theme.sidebar_secondary.remove:

rc.base.models.DataBase.create
==============================

.. py:method:: rc.base.models.DataBase.create(path: rc.base.definitions.PathLike, **tableData_and_metaData: rc.base.definitions.DataFrame | Table | rc.base.definitions.MetaData) -> rc.base.definitions.Self
   :classmethod:


   Create a *DataBase* in ``path``.

   :param path: The folder to store the *DataBase* in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
   :param \*\*tableData_and_metaData: Data to update ``cls.defaults()``, in the form ``names[i]=tables[i]``,
                                      and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

   Returns: The *DataBase* created.

