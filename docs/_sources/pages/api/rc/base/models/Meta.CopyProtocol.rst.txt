rc.base.models.Meta.CopyProtocol
================================

.. py:exception:: rc.base.models.Meta.CopyProtocol

   Bases: :py:obj:`rc.base.definitions.Copy`

   .. autoapi-inheritance-diagram:: rc.base.models.Meta.CopyProtocol
      :parts: 1


   ``cls.copy(src, dst)`` is selective, copying only relevant items in ``src.path`` while preserving irrelevant items in ``dst``.

