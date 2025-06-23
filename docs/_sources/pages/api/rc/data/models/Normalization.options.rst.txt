rc.data.models.Normalization.options
====================================

.. py:attribute:: rc.data.models.Normalization.options
   :type:  Normalization.NamedTables[rc.base.MetaData]

   Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
   Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
   the remainder populate ``self[i].options.read``. Must be overridden.
