rc.data.designs.CoordDesign.ext
===============================

.. py:attribute:: rc.data.designs.CoordDesign.ext
   :type:  str
   :value: '.csv'


   Class attribute specifying the file extension terminating ``self.path``.
   Override if and only if the derived Class must be stored in a file.
   Otherwise, ``cls.ext == ''`` and the derived Class is stored in a folder.
