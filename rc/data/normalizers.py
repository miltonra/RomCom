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


class Normalizer(DataBase):
    """ Normalizer of a Design. """

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
        """ Creates a new instance of ``cls`` at ``path`` from ``design: PointDesign | CoordDesign``. """

    @classmethod
    def create(cls, path: PathLike, design: Design, **meta: Any) -> Self:
        """ Create a ``Normalizer`` in ``path``.

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
        tableNames = cls.names()
        design = PointDesign.create(path / tableNames[0], design)
        coord = CoordNormalizer.create(path / 'coord', design, **meta)
        point = PointNormalizer.create(path / 'point', design, coord[1], **meta)
        meta = meta(**(isDependent | isUniform))
        meta = Meta.create(cls._meta_in(path), **(cls.defaultMeta | meta))
        empiricalPDF = PointPDF.copy(point[1], path / tableNames[1])
        coordPDF = CoordPDF.copy(coord[2] if isUniform else coord[1], path / tableNames[2])
        pointPDF = (PointPDF.copy(empiricalPDF, path / tableNames[3]) if isDependent
                    else empiricalPDF.rational(path / tableNames[3], coordPDF))
        ratioPDF = empiricalPDF.compare(path / tableNames[4], pointPDF)
        pointStats = PointStats.create(path / tableNames[5], design)
        return cls(path)

class PointNormalizer(Normalizer):
    """ Normalizer of a Design. """

    NamedTables: type[NamedTuple] = Normalizer.NamedTables

    @classmethod
    def create(cls, path: PathLike, design: Design, coordPDF: CoordPDF, **meta: Any) -> Self:
        """ Create a ``Normalizer`` in ``path``.

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
        tableNames = cls.names()
        design = PointDesign.create(path / tableNames[0], design)
        empiricalPDF = PointPDF.create(path / tableNames[1], design)
        coordPDF = CoordPDF.copy(coordPDF, path / tableNames[2])
        pointPDF = empiricalPDF.rational(path / tableNames[3], coordPDF)
        ratioPDF = empiricalPDF.compare(path / tableNames[4], pointPDF)
        pointStats = PointStats.create(path / tableNames[5], design)
        return cls(path)

class CoordNormalizer(Normalizer):
    """ Normalizer of a Design. """

    NamedTables: type[NamedTuple] = Normalizer.NamedTables

    Tables: NamedTables[type[Table], ...] = NamedTables(design=CoordDesign, empiricalPDF=CoordPDF, coordPDF=CoordPDF, pointPDF=CoordPDF,
                              ratioPDF=CoordPDF, pointStats=PointStats)
    """ Table Types, to communicate ``Table.readOptions`` and ``Table.writeOptions``. """

    @classmethod
    def create(cls, path: PathLike, design: PointDesign, **meta: Any) -> Self:
        meta = Meta.create(cls._meta_in(path), **(cls.defaultMeta | meta))
        path = meta.path.parent
        tableNames = cls.names()
        coordDesign = CoordDesign.create(path / tableNames[0], design)
        empiricalPDF = CoordPDF.create(path / tableNames[1], coordDesign)
        coordPDF = CoordPDF.rational(path / tableNames[2], empiricalPDF)
        pointPDF = PointPDF.create(path / tableNames[3], design).rational(path / tableNames[3], coordPDF)
        ratioPDF = empiricalPDF.compare(path / tableNames[4], coordPDF)
        pointStats = PointStats.create(path / tableNames[5], design)
        return cls(path)

