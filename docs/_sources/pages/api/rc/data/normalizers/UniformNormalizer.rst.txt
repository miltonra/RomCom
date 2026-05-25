rc.data.normalizers.UniformNormalizer
=====================================


.. module:: rc.data.normalizers

.. toctree::
   :hidden:

   /pages/api/rc/data/normalizers/UniformNormalizer.NamedTables
   /pages/api/rc/data/normalizers/UniformNormalizer.Tables
   /pages/api/rc/data/normalizers/UniformNormalizer.create
   /pages/api/rc/data/normalizers/UniformNormalizer.defaultMeta
   /pages/api/rc/data/normalizers/UniformNormalizer.CreateP
   /pages/api/rc/data/normalizers/UniformNormalizer.IndexP
   /pages/api/rc/data/normalizers/UniformNormalizer.EqualsP
   /pages/api/rc/data/normalizers/UniformNormalizer.ReadP
   /pages/api/rc/data/normalizers/UniformNormalizer.UpdateP
   /pages/api/rc/data/normalizers/UniformNormalizer.DeleteP
   /pages/api/rc/data/normalizers/UniformNormalizer.CopyP
   /pages/api/rc/data/normalizers/UniformNormalizer.tables
   /pages/api/rc/data/normalizers/UniformNormalizer.meta
   /pages/api/rc/data/normalizers/UniformNormalizer.names
   /pages/api/rc/data/normalizers/UniformNormalizer.defaults
   /pages/api/rc/data/normalizers/UniformNormalizer.copy
   /pages/api/rc/data/normalizers/UniformNormalizer.delete
   /pages/api/rc/data/normalizers/UniformNormalizer.ext
   /pages/api/rc/data/normalizers/UniformNormalizer.NameP
   /pages/api/rc/data/normalizers/UniformNormalizer.path
   /pages/api/rc/data/normalizers/UniformNormalizer.extAppend
   /pages/api/rc/data/normalizers/UniformNormalizer.mkdir

.. py:class:: rc.data.normalizers.UniformNormalizer(path, **tables)

   Bases: :py:obj:`IndependentNormalizer`

   .. autoapi-inheritance-diagram:: rc.data.normalizers.UniformNormalizer
      :parts: 1


   Normalizer of a Design.

   Read the DataBase in ``path``.
   Reading is lazy: If ``names[i]`` occurs in ``**tables`` it's Table is not read, just updated.
   Overrides must call ``super(DataBase).__init__(path, **tables)`` as a matter of priority.

   :param path: The Path to read from.
   :param \*\*tables: Tables to update those read, in the form ``names[i]=tables[i], ...``.

   :raises FileNotFoundError: If ``path`` lacks ``self.meta`` or any member of
       ``self.Tables.names`` not mentioned in ``**tables``.

Protocols
----------

.. autoapisummary::

   rc.data.normalizers.UniformNormalizer.CreateP
   rc.data.normalizers.UniformNormalizer.IndexP
   rc.data.normalizers.UniformNormalizer.EqualsP
   rc.data.normalizers.UniformNormalizer.ReadP
   rc.data.normalizers.UniformNormalizer.UpdateP
   rc.data.normalizers.UniformNormalizer.DeleteP
   rc.data.normalizers.UniformNormalizer.CopyP
   rc.data.normalizers.UniformNormalizer.NameP


Attributes
----------

.. autoapisummary::

   rc.data.normalizers.UniformNormalizer.NamedTables
   rc.data.normalizers.UniformNormalizer.Tables
   rc.data.normalizers.UniformNormalizer.defaultMeta
   rc.data.normalizers.UniformNormalizer.ext


Properties
----------

.. autoapisummary::

   rc.data.normalizers.UniformNormalizer.tables
   rc.data.normalizers.UniformNormalizer.meta
   rc.data.normalizers.UniformNormalizer.path


Methods
-------

.. autoapisummary::

   rc.data.normalizers.UniformNormalizer.create
   rc.data.normalizers.UniformNormalizer.names
   rc.data.normalizers.UniformNormalizer.defaults
   rc.data.normalizers.UniformNormalizer.copy
   rc.data.normalizers.UniformNormalizer.delete
   rc.data.normalizers.UniformNormalizer.extAppend
   rc.data.normalizers.UniformNormalizer.mkdir


