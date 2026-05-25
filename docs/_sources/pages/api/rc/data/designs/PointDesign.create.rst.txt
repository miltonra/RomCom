rc.data.designs.PointDesign.create
==================================

.. py:method:: rc.data.designs.PointDesign.create(path, design)
   :classmethod:


   Create a ``Design`` at ``path``.

   :param path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
   :param design: The ``CoordDesign | PointDesign`` to reformat if necessary and store in ``path``.

   Returns: The ``Design`` created.

