:html_theme.sidebar_secondary.remove:

rc.base.models.Meta.create
==========================

.. py:method:: rc.base.models.Meta.create(path: rc.base.definitions.PathLike, **data: rc.base.definitions.Any)
   :classmethod:


   Create a Meta at ``path``, overwriting.

   :param path: The Path (file) to store ``self``, overwritten if existing.
                A ``.json`` extension is implicitly appended.
   :param \*\*data: The ``MetaData`` to store.

   Returns: The Meta created.
   Raises: AssertionError if ``data`` is empty.

