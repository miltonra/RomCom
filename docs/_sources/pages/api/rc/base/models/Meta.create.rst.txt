rc.base.models.Meta.create
==========================

.. py:method:: rc.base.models.Meta.create(path, **data)
   :classmethod:


   Create a ``Meta`` at ``path``, overwriting.

   :param path: The ``Path`` (file) to store ``self``, overwritten if existing.
                A ``.json`` extension is automatically appended.
   :param \*\*data: The ``MetaData`` to store.

   Returns: The ``Meta`` created.

