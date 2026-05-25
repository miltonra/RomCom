rc.data.normalizers.UniformNormalizer.create
============================================

.. py:method:: rc.data.normalizers.UniformNormalizer.create(path, design, **meta)
   :classmethod:


   Normalizer of a ``PointDesign``

   :param path: The folder to store the ``Normalizer`` in. Need not exist.
   :param design: the ``Design`` to normalize, either a ``PointDesign`` or a ``CoordDesign``.
   :param coordPDF: The ``CoordPDF`` used to generate the rational PDF.
   :param \*\*meta: ``self.meta`` to update. In particular, ``isUniform = True`` infers a uniform distribution
                    for each categorical coord, discarding ``isIndependent`` as tautologically ``False``.
                    On the other hand, ``isDependent = True`` infers a ``PointPDF`` from a
                    ``PointDesign``. If both are ``False`` or absent, the ``PointPDF`` is inferred from the
                    mutually independent ``CoordPDF``s inferred from the ``CoordDesign``.

   Returns: The Normalization created.

