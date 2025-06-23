rc.data.models.NormalDesignMatrix.ext
=====================================

.. py:attribute:: rc.data.models.NormalDesignMatrix.ext
   :type:  str
   :value: '.csv'


   Class attribute specifying the file extension terminating ``self.path``.
   Override if and only if the derived class must be stored in a file.
   Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.
