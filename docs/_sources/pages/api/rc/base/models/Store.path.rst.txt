:html_theme.sidebar_secondary.remove:

rc.base.models.Store.path
=========================

.. py:property:: rc.base.models.Store.path
   :type: rc.base.definitions.Path


   The Path to this Store, without ``cls.ext``.
   File extension is internal, meaning ``self._path = self.path + cls.ext``.
