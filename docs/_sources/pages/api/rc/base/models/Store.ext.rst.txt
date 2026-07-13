:html_theme.sidebar_secondary.remove:

:html_theme.sidebar_secondary.remove:

rc.base.models.Store.ext
========================

.. py:attribute:: rc.base.models.Store.ext
   :type:  str
   :value: ''


   Class attribute specifying the file extension terminating ``self.path``.
   Override if and only if the derived Class must be stored in a file.
   Otherwise, ``cls.ext == ''`` and the derived Class is stored in a folder.
