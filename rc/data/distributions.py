#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2027 Robert A. Milton
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

import numpy as np
import pandas as pd

from rc.data.designs import *

from itertools import product


class PDF(Table):
    """ The Probability Density Function(s) of categorical coords or points."""

    readOptions: MetaData = Table.readOptions | {'index_col': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``design: PointDesign | CoordDesign``. """

    def compare(self, path: PathLike, pdf: Self) -> Self:
        """ Compare ``self`` to ``pdf``, returning the ratio.

        Args:
            path: The Path to this Table, overwritten if existing.
            pdf: The ``PDF`` to compare to.

        Returns: The ratio of ``self / pdf``, stored in ``path``.
        """
        return type(self)(self.mkdir(path), (self.pd / pdf.pd).fillna(0.0))

    @classmethod
    @abstractmethod
    def create(cls, path: PathLike, design: Design) -> Self:
        """ Create a ``PDF`` Table at ``path``.

        Args:
            path: The Path to this Table, overwritten if existing.
            design: The ``CoordDesign | PointDesign`` to work from.

        Returns: The ``PDF`` created.
        """


class PointPDF(PDF):
    """ The Probability Density Function of categorical points."""

    def independent(self, path: PathLike, coordPDF: CoordPDF) -> Self:
        """ Create an independent ``PointPDF`` from ``coordPDF``.

        Args:
            path: The Path to this Table, overwritten if existing.
            coordPDF: The ``CoordPDF`` describing rationality.

        Returns: The rational ``PointPDF`` created.
        """
        if coordPDF.pd.shape[0] <= 1:
            return PointPDF(self.mkdir(path), self.pd.copy(deep=True))
        coordPDF = coordPDF.pd.copy(deep=True).reset_index(level=1)
        pdf = coordPDF.groupby(level=0, sort=False).agg(list)
        axis = Design.coordSeparator.join(pdf.index.tolist())
        pdf = pdf.to_numpy().tolist()
        coord, p = zip(*pdf)
        coord = [Design.coordSeparator.join(coords) for coords in product(*coord)]
        p = [np.prod(ps) for ps in product(*p)]
        yAxis = self.pd.index.unique(level=0).tolist()
        p = np.concatenate([p,] * len(yAxis), axis=0)
        pdf = pd.DataFrame(p, index=pd.MultiIndex.from_product([yAxis, coord], names=self.pd.index.names),
                           columns=['p│'])
        return PointPDF(self.mkdir(path), pdf)

    @classmethod
    def create(cls, path: PathLike, design: PointDesign) -> Self:
        lAxis = design.pd.columns[-2]
        pdf = design.pd.iloc[:, -2:].copy(deep=True).rename(columns = {'y│': 'p│'})
        n = pdf.shape[0]
        pdf.set_index(lAxis, drop=True, inplace=True)
        pdf = pdf.groupby(level=0).count()
        lAxis = lAxis.partition(Design.coordSeparator)[2]
        pdf.index = pdf.index.str.partition(Design.coordSeparator).droplevel(1).rename([n, lAxis])
        return cls(cls.mkdir(path), pdf / pdf.groupby(level=0).sum())


class CoordPDF(PDF):
    """ The Probability Density Function(s) of statistically independent categorical coords."""

    @classmethod
    def uniform(cls, path: PathLike, coordPDF: CoordPDF) -> Self:
        """ Create a uniform version of ``coordPDF``.

        Args:
            path: The Path to this Table, overwritten if existing.
            coordPDF: The ``CoordPDF`` to make uniform.

        Returns: The uniform ``CoordPDF`` created.
        """
        coordPDF = coordPDF.pd.copy(deep=True)
        for axis in coordPDF.index.unique(level=0):
            coordPDF.loc[[axis],['p│']] = 1.0 / coordPDF.loc[[axis],['p│']].shape[0]
        return cls(cls.mkdir(path), coordPDF)

    @classmethod
    def create(cls, path: PathLike, design: CoordDesign) -> Self:
        design = design.pd.copy(deep=True)
        if 'i│' not in design.columns.get_level_values(0):
            design.insert(0, ('i│','i│'), '0')
        design = design['i│']
        design.insert(design.shape[1], 'p│', 1.0 / design.shape[0])
        design = {axis: design.loc[:, [axis, 'p│']].set_index(axis).groupby(level=0).sum()
                        for axis in design.columns[:-1]}
        pdf = pd.concat(design.values(), axis=0)
        pdf.index = pd.MultiIndex.from_tuples([(axis, coord) for axis in design.keys()
                                               for coord in design[axis].index], names=['axis', 'coord'])
        return cls(cls.mkdir(path), pdf)


class PointStats(Table):

    readOptions: MetaData = Table.readOptions | {'index_col': [0, 1], 'head': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a ``PointDesign``."""

    @classmethod
    def create(cls, path: PathLike, design: PointDesign) -> Self:
        """ The empirical statistics of a PointDesign per categorical points.

        Args:
            path: The Path to this Table, overwritten if existing.
            design: The ``PointDesign`` to work from.

        Returns: The empirical ``PointStats`` calculated from ``design``.
        """
        lAxis = design.pd.columns[-2]
        stats = design.pd.copy(deep=True).drop(columns=['n│'])
        stats.set_index(lAxis, drop=True, inplace=True)
        n = stats.shape[0]
        column = stats.columns
        stat = ['min', 'max', 'mean', lambda x: x.std(ddof=0)]
        stats = stats.groupby(level=0).agg(stat)
        stat = ['min│', 'max│', 'mean│', 'SD│']
        stats.columns = pd.MultiIndex.from_product([column, stat], names=['axis', 'stat'])
        lAxis = lAxis.partition(Design.coordSeparator)[2]
        stats.index = stats.index.str.partition(Design.coordSeparator).droplevel(1).rename([n, lAxis])
        return cls(cls.mkdir(path), stats)


class CoordStats(Table):

    readOptions: MetaData = Table.readOptions | {'head': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""


    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a ``PointPDF`` and ``PointStats``."""

    def diff(self, path: PathLike, stats: Self) -> Self:
        """ Compare ``self`` to ``stats``, returning the difference.

        Args:
            path: The Path to this Table, overwritten if existing.
            stats: The ``Stats`` to compare to.

        Returns: The difference ``self - stats``, stored in ``path``.
        """
        return type(self)(self.mkdir(path), (self.pd - stats.pd).fillna(0.0))

    @classmethod
    def create(cls, path: PathLike, pointPDF: PointPDF, pointStats: PointStats) -> Self:
        """ The statistics of a ``PointPDF`` and ``PointStats`` per categorical coord.

        Args:
            path: The Path to this Table, overwritten if existing.
            pointPDF: The ``PointPDF`` to work from.
            pointStats: The ``PointStats`` to work from.

        Returns: The ``CoordStats`` describing ``pointPDF`` and ``pointStats``.
        """
        stats = pointStats.pd.copy(deep=True)
        stats.columns = stats.columns.swaplevel()
        columns = {}
        columns['min│'] = stats.loc[:, 'min│'].groupby(level=0).agg('min')
        columns['max│'] = stats.loc[:, 'max│'].groupby(level=0).agg('max')
        columns['mean│'] = (stats.loc[:, 'mean│'].mul(pointPDF.pd.loc[:, 'p│'], axis=0)).groupby(level=0).agg('sum')
        columns['SD│'] = ((stats.loc[:, 'SD│']**2 + stats.loc['mean│']**2).mul(pointPDF.pd.loc[:, 'p│'], axis=0)
                          ).groupby(level=0).agg('sum')
        columns['SD│'] = np.sqrt(columns[:, 'SD│'] - columns['mean│']**2)
        stats = pd.DataFrame(columns.values(), columns=stats.columns)
        stats.columns = stats.columns.swaplevel()
        return cls(cls.mkdir(path), stats)
