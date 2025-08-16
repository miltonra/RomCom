#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2025 Robert A. Milton
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

""" Normalizers for data storage. """

from __future__ import annotations

from rc.data.distributions import *


class IndependentNormalizer(DataBase):

    class NamedTables(NamedTuple):
        """ NamedTables in a Normalizer. """

        design: Design = PointDesign
        """ The Design of user data."""
        empiricalPDF: PDF = PointPDF
        """ The empirical Probability Density Function of categorical coords or points."""
        coordPDF: PDF = CoordPDF
        """ The Probability Density Function of categorical coords."""
        pointPDF: PDF = PointPDF
        """ The Probability Density Function of categorical points."""
        ratioPDF: PDF = PointPDF
        """ The ratio between ``empiricalPDF`` and ``pointPDF | coordPDF``."""
        pointStats: PointStats = PointStats
        """ The statistics of continuous coordinates by categorical point."""
        # coordStats: CoordStats = CoordStats
        # """ The statistics of continuous coordinates by categorical coord."""
        # pointStatsDiff: PointStats = PointStats
        # """ The difference between empirical and rational stats."""
        # coordStatsDiff: CoordStats = CoordStats
        # """ The difference between empirical and rational stats."""

        def __call__(self, name: str) -> Table:
            return getattr(self, name)

    Tables: NamedTables[type[Table], ...] = NamedTables()
    """ Table Types, to communicate ``Table.readOptions`` and ``Table.writeOptions``. """

    defaultMeta: MetaData = {'Tables': {name: TableType.__name__ for name, TableType in Tables._asdict().items()}}
    """ Default ``self.meta``. """

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``PointDesign``. """

    @classmethod
    def create(cls, path: PathLike, design: PointDesign, coordPDF: CoordPDF, **meta: Any) -> Self:
        """ Normalizer of a ``PointDesign``

        Args:
            path: The folder to store the ``Normalizer`` in. Need not exist.
            design: the ``Design`` to normalize, either a ``PointDesign`` or a ``CoordDesign``.
            coordPDF: The ``CoordPDF`` used to generate the rational PDF.
            **meta: ``self.meta`` to update. In particular, ``isUniform = True`` infers a uniform distribution
                for each categorical coord, discarding ``isIndependent`` as tautologically ``False``.
                On the other hand, ``isDependent = True`` infers a ``PointPDF`` from a
                ``PointDesign``. If both are ``False`` or absent, the ``PointPDF`` is inferred from the
                mutually independent ``CoordPDF``s inferred from the ``CoordDesign``.

        Returns: The Normalization created.
        """
        meta = Meta.create(cls._meta_in(path), **(cls.defaultMeta | meta))
        path = meta.path.parent
        design = PointDesign.create(path / 'design', design)
        empiricalPDF = PointPDF.create(path / 'empiricalPDF', design)
        coordPDF = CoordPDF.copy(coordPDF, path / 'coordPDF')
        pointPDF = empiricalPDF.independent(path / 'pointPDF', coordPDF)
        ratioPDF = empiricalPDF.compare(path / 'ratioPDF', pointPDF)
        pointStats = PointStats.create(path / 'pointStats', design)
        return cls(path)

class UniformNormalizer(IndependentNormalizer):
    """ Normalizer of a Design. """

    NamedTables: type[NamedTuple] = IndependentNormalizer.NamedTables

    Tables: NamedTables[type[Table], ...] = NamedTables(design=CoordDesign, empiricalPDF=CoordPDF,
                                                        coordPDF=CoordPDF, pointPDF=CoordPDF,
                                                        ratioPDF=CoordPDF, pointStats=PointStats)
    """ Table Types, to communicate ``Table.readOptions`` and ``Table.writeOptions``. """

    @classmethod
    def create(cls, path: PathLike, design: PointDesign, **meta: Any) -> Self:
        meta = Meta.create(cls._meta_in(path), **(cls.defaultMeta | meta))
        path = meta.path.parent
        coordDesign = CoordDesign.create(path / 'design', design)
        empiricalPDF = CoordPDF.create(path / 'empiricalPDF', coordDesign)
        coordPDF = CoordPDF.uniform(path / 'coordPDF', empiricalPDF)
        pointPDF = PointPDF.create(path / 'pointPDF', design).independent(path / 'pointPDF', coordPDF)
        ratioPDF = empiricalPDF.compare(path / 'ratioPDF', coordPDF)
        pointStats = PointStats.create(path / 'pointStats', design)
        return cls(path)

class Normalizer(IndependentNormalizer):

    NamedTables: type[NamedTuple] = IndependentNormalizer.NamedTables

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``design: PointDesign | CoordDesign``. """

    @classmethod
    def create(cls, path: PathLike, design: Design, **meta: Any) -> Self:
        """ Normalizer of a Design.

        Args:
            path: The folder to store the ``Normalizer`` in. Need not exist.
            design: the ``Design`` to normalize, either a ``PointDesign`` or a ``CoordDesign``.
            **meta: ``self.meta`` to update. In particular, ``isUniform = True`` infers a uniform distribution
                for each categorical coord, discarding ``isIndependent`` as tautologically ``False``.
                On the other hand, ``isDependent = True`` infers a ``PointPDF`` from a
                ``PointDesign``. If both are ``False`` or absent, the ``PointPDF`` is inferred from the
                mutually independent ``CoordPDF``s inferred from the ``CoordDesign``.

        Returns: The Normalization created.
        """
        isUniform = {'isUniform': True} if meta.pop('isUniform', False) else {}
        isDependent = {'isDependent': True} if meta.pop('isDependent', False) and not isUniform else {}
        design = PointDesign.create(path / 'design', design)
        uniform = UniformNormalizer.create(path / 'uniform', design, **meta)
        independent = IndependentNormalizer.create(path / 'independent', design, uniform['empiricalPDF'], **meta)
        meta = Meta.create(cls._meta_in(path), **(cls.defaultMeta | meta | isDependent | isUniform))
        empiricalPDF = PointPDF.copy(independent['empiricalPDF'], path / 'empiricalPDF')
        coordPDF = CoordPDF.copy(uniform['coordPDF'] if isUniform else uniform['empiricalPDF'], path / 'coordPDF')
        pointPDF = (PointPDF.copy(empiricalPDF, path / 'pointPDF') if isDependent
                    else empiricalPDF.independent(path / 'pointPDF', coordPDF))
        ratioPDF = empiricalPDF.compare(path / 'ratioPDF', pointPDF)
        pointStats = PointStats.create(path / 'pointStats', design)
        return cls(path)

