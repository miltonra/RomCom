rc.base.models.Meta.ext
=======================

.. py:attribute:: rc.base.models.Meta.ext
   :type:  str
   :value: '.json'


   Class attribute specifying the file extension terminating ``self.path``.
   Override if and only if the derived class must be stored in a file.
   Otherwise, ``cls.ext == ''`` and the derived class is stored in a folder.
