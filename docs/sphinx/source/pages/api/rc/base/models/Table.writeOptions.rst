rc.base.models.Table.writeOptions
=================================

.. py:attribute:: rc.base.models.Table.writeOptions
   :type:  list[str]
   :value: ['sep', 'na_rep', 'float_format']


   Class attribute listing kwargs which will be interpreted as write options.
   All other kwargs are interpreted as read options.
   To specify a separator, use ``delimiter`` as read option and ``sep`` as write option.
