rc.data.models.DesignMatrix.options
===================================

.. py:property:: rc.data.models.DesignMatrix.options
   :type: MetaData


   A ``dict of options for file operations involving ``self``.
   Any option not in ``Table.writeOptions`` is stored in ``self.options.read`` and passed to ``pd.read_csv``.
   Any option in ``Table.writeOptions`` is stored in ``self.options.write``
   and passed to ``pd.DataFrame.to_csv``.
   The setter updates via logical or ``|=``, so existing values are retained unless explicitly updated.
