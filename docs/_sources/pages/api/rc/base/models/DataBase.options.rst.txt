rc.base.models.DataBase.options
===============================

.. py:attribute:: rc.base.models.DataBase.options
   :type:  DataBase.NamedTables[MetaData]

   Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
   Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
   the remainder populate ``self[i].options.read``. Must be overridden.
