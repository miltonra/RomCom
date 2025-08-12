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

""" Categorical input PDFs and statistics. """

from __future__ import annotations

from rc.data.designs import *

from itertools import product


class PDF(Table):
    """ The Probability Density Function(s) of categorical coords or points."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``design: PointDesign | CoordDesign``. """

    def compare(self, path: PathLike, pdf: Self) -> Self:
        """ Compare ``self`` to ``pdf``, returning the ratio.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            pdf: The ``PDF`` to compare to.

        Returns: The ratio of ``self / pdf``, stored in ``path``.
        """
        return type(self)(self.mkdir(path), (self.pd / pdf.pd).fillna(0.0))

    @classmethod
    @abstractmethod
    def rational(cls, path: PathLike, coordPDF: CoordPDF) -> Self:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create(cls, path: PathLike, design: Design) -> Self:
        """ Create a ``PDF`` Table at ``path``.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            design: The ``CoordDesign | PointDesign`` to work from.

        Returns: The ``PDF`` created.
        """


class PointPDF(PDF):
    """ The Probability Density Function of categorical points."""

    @classmethod
    def rational(cls, path: PathLike, coordPDF: CoordPDF) -> Self:
        """ Create a rational, independent ``PointPDF`` from ``coordPDF``.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            coordPDF: The ``CoordPDF`` describing rationality.

        Returns: The rational ``PointPDF`` created.
        """
        coordPDF = coordPDF.pd.reset_index(level=1)
        pdf = coordPDF.groupby(level=0).agg(list)
        axis = Design.coordSeparator.join(pdf.index.tolist())
        pdf = pdf.to_numpy().tolist()
        coord, p = zip(*pdf)
        coord = [Design.coordSeparator.join(coords) for coords in product(*coord)]
        p = [np.prod(ps) for ps in product(*p)]
        return type(cls)(cls.mkdir(path), pd.DataFrame(p, index=pd.Index(coord, name=axis), columns=['p']))

    @classmethod
    def create(cls, path: PathLike, design: PointDesign) -> Self:
        designed = design
        laxis = design.pd.columns[lCol]
        pdf = design.pd.iloc[l]
        # pdf.insert(0, 'll', pdf[laxis].apply(lambda s: s.split(Design.coordSeparator)[0]))
        pdf.insert(pdf.shape[1], 'p', 1.0 / pdf.shape[0])
        pdf.set_index(laxis, drop=True, inplace=True)
        pdf = pdf.groupby(level=0).sum()
        return cls(cls.mkdir(path), pdf)


class CoordPDF(PDF):
    """ The Probability Density Function(s) of statistically independent categorical coords."""

    readOptions: MetaData = Table.readOptions | {'index_col': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    @classmethod
    def rational(cls, path: PathLike, coordPDF: CoordPDF) -> Self:
        """ Create a uniform version of ``coordPDF``.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            coordPDF: The ``CoordPDF`` to make uniform.

        Returns: The uniform ``CoordPDF`` created.
        """
        coordPDF = coordPDF.pd
        for axis in coordPDF.index.get_level_values(0).unique():
            coordPDF.loc[[axis],['p']] = 1.0 / coordPDF.loc[[axis],['p']].shape[0]
        return CoordPDF(cls.mkdir(path), coordPDF)

    @classmethod
    def create(cls, path: PathLike, design: CoordDesign) -> Self:
        if 'i' not in design.pd.columns.get_level_values(0):
            design.pd.insert(0, ('i','i'), '0')
        design = design.pd['i']
        design.insert(design.shape[1], Design.coordSeparator, 1.0 / design.shape[0])
        design = {axis: design.loc[:, [axis, Design.coordSeparator]].set_index(axis).groupby(level=0).sum()
                        for axis in design.columns[:-1]}
        pdf = pd.concat(design.values(), axis=0).rename(columns={Design.coordSeparator: 'p'})
        pdf.index = pd.MultiIndex.from_tuples([(axis, coord) for axis in design.keys()
                                               for coord in design[axis].index], names=['axis', 'coord'])
        return cls(cls.mkdir(path), pdf)


class PointStats(Table):

    readOptions: MetaData = Table.readOptions | {'header': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    @classmethod
    def create(cls, path: PathLike, design: PointDesign) -> Self:
        """ The empirical statistics of a PointDesign per categorical points.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            design: The ``PointDesign`` to work from.

        Returns: The empirical ``PointStats`` calculated from ``design``.
        """
        laxis = design.pd.columns[lCol]
        stats = design.pd.drop(columns=['n'])
        stats.set_index(laxis, drop=True, inplace=True)
        column = stats.columns
        stat = ['min', 'max', 'mean', 'std']
        stats = stats.groupby(level=0).agg(stat)
        stats.columns = pd.MultiIndex.from_product([column, stat], names=['axis', 'stat'])
        return cls(cls.mkdir(path), stats)


class CoordStats(Table):

    readOptions: MetaData = Table.readOptions | {'header': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""


    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a ``PointPDF`` and ``PointStats``."""

    def compare(self, path: PathLike, stats: Self) -> Self:
        """ Compare ``self`` to ``stats``, returning the difference.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            stats: The ``Stats`` to compare to.

        Returns: The difference ``self - stats``, stored in ``path``.
        """
        return type(self)(self.mkdir(path), (self.pd - stats.pd).fillna(0.0))

    @classmethod
    def create(cls, path: PathLike, pointPDF: PointPDF, pointStats: PointStats) -> Self:
        """ The statistics of a ``PointPDF`` and ``PointStats`` per categorical coord.

        Args:
            path: The ``Path`` to this Table, overwritten if existing.
            pointPDF: The ``PointPDF`` to work from.
            pointStats: The ``PointStats`` to work from.

        Returns: The ``CoordStats`` describing ``pointPDF`` and ``pointStats``.
        """
        stats = pointStats.pd.drop(columns=['n'])
        stats.index = stats.index.str.partition(Design.coordSeparator)
        stats.index.droplevel(1, inplace=True)
        stats = stats.groupby(level=0).agg(
            {'p': ['sum', 'prod'], 'min': 'min', 'max': 'max', 'mean': 'mean', 'std': 'std'}
        )
        return cls(cls.mkdir(path), stats)
