rc.data.normalizers.Normalizer.ext
==================================

.. py:attribute:: rc.data.normalizers.Normalizer.ext
   :type:  str
   :value: ''


   Class attribute specifying the file extension terminating ``self.path``.
   Override if and only if the derived Class must be stored in a file.
   Otherwise, ``cls.ext == ''`` and the derived Class is stored in a folder.
