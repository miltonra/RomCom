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

""" Models for data storage. """

from __future__ import annotations

import pandas as pd
from numpy.ma.extras import column_stack

from rc.base import *

from itertools import product



#: Column containing ``n`` (row) in a PointDesign.
ncol: int = 0

#: Column containing ``l`` (Point) in a PointDesign.
lcol: int = -2

#: Slice for ``n`` (row) in a PointDesign.
n: tuple[slice, slice] = (slice(None, None), slice(ncol, ncol + 1))

#: Slice for ``x`` (continuous inputs) in a PointDesign.
x: tuple[slice, slice] = (slice(None, None), slice(ncol + 1, lcol))

#: Slice for ``l`` (categorical Point) in a PointDesign.
l: tuple[slice, slice] = (slice(None, None), slice(lcol, lcol + 1))

#: Slice for ``y`` (output) in a PointDesign.
y: tuple[slice, slice] = (slice(None, None), slice(lcol + 1, None))


class DesignMatrix(Table):
    """ A DesignMatrix of user data, tabulating continuous inputs, categorical inputs, and outputs."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``designMatrix: PointDesign | CoordDesign``. """

    coordSeparator: str = '│'
    """ The separator used to delimit coords in a column of categorical points. """

    @classmethod
    def axisType(cls, axis: str) -> str:
        """ The axisType of the given ``axis``.

        Args:
            axis: A ``str`` axisType, such as ``'x', 'i', 'y', 'continuous', 'discrete', 'output', etc.

        Returns: ``axisType in 'x', 'i', 'y', '?'``. ``'?'`` is returned if ``axis`` is not recognized.

        """
        match axis.lower():
            case 'x' | 'input' | 'in' | 'continuous' | 'float' :
                return 'x'
            case 'i' | 'category' | 'cat' | 'discrete' | 'int' | 'str' :
                return 'i'
            case 'y' | 'output' | 'out'| 'map' | 'func' :
                return 'y'
        return '?'

    @classmethod
    @abstractmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        """ Create a ``DesignMatrix`` at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The ``CoordDesign | PointDesign`` to reformat if necessary and store in ``path``.

        Returns: The ``DesignMatrix`` created.
        """


class PDF(Table):
    """ The Probability Density Function(s) of categorical coords or points."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``designMatrix: PointDesign | CoordDesign``. """

    @classmethod
    @abstractmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        """ Create a ``PDF`` at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The ``CoordDesign | PointDesign`` to work from.

        Returns: The ``PDF`` created.
        """


class Stats(Table):
    """ The statistics of continuous coordinates by output coord or categorical point."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``designMatrix: PointDesign | CoordDesign``. """

    @classmethod
    @abstractmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        """ Create a ``Stats`` table at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The ``CoordDesign | PointDesign`` to work from.

        Returns: The ``Stats`` created.
        """


class CoordDesign(DesignMatrix):
    """ The familiar user format of a DesignMatrix which has many axes (columns), and two header rows.
    The first header row contains the axisType, the second header row contains the axis."""

    readOptions: MetaData = Table.readOptions | {'header': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        match designMatrix:
            case CoordDesign():
                # Reformat colum labels and categorical coords to strings.
                coordDesign = designMatrix.pd.rename(cls.axisType, axis='columns', level=0)
                if 'i' in coordDesign.columns.get_level_values(0):
                    coordDesign['i'] = coordDesign['i'].astype(str)
            case PointDesign():
                # Reformat the PointDesign to an CoordDesign.
                pointDesign = designMatrix.pd.set_index('n')
                pointDesign[pointDesign.columns[lcol]] = pointDesign[pointDesign.columns[lcol]].astype(str)
                # convert Points to coords.
                iDesign = pointDesign.iloc[:, lcol].apply(lambda point: point.split(cls.coordSeparator))
                iDesign = pd.DataFrame(iDesign.tolist(), index=pointDesign.index,
                                       columns=iDesign.name.split(cls.coordSeparator))
                # convert first categorical variable to output axes.
                yAxes = iDesign['l'].drop_duplicates().tolist()
                yDesign = pd.concat([iDesign['l'], pointDesign.iloc[:, lcol + 1]], axis=1)
                for yCoord in yAxes:
                    yDesign[yCoord] = (yDesign['l'] == yCoord).astype(int) * yDesign.iloc[:, 1]
                # Collect x, i, and y column groups.
                coordDesign = pd.concat([pointDesign.iloc[:, :lcol].groupby(level=0, sort=False).mean(),
                                         iDesign.iloc[:, 1:].groupby(level=0, sort=False).first(),
                                         yDesign.iloc[:, 2:].groupby(level=0, sort=False).sum(), ],
                                        axis=1)
                # Label axes correctly.
                coordDesign.columns = pd.MultiIndex.from_tuples(
                    [('x', col) for col in pointDesign.columns[:lcol]] +
                    [('i', col) for col in iDesign.columns[1:]] +
                    [('y', col) for col in yAxes])
                # Detect NaN.
                coordDesign['y'] = coordDesign['y'].replace(0.0, np.nan)
            case _:
                raise NotImplementedError(f'I do not know how to create an CoordDesign from {type(designMatrix)}')
        return cls(cls.mkdir(path), coordDesign)


class CoordPDF(PDF):

    readOptions: MetaData = Table.readOptions | {'index_col': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    def ratio(self, path: Store.Path, coordPDF: CoordPDF | None = None) -> Self:
        """ The ratio between ``self`` and ``coordPDF``, or ``self`` and a uniform distribution
        if ``coordPDF is None``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            coordPDF: The coordPDF to compare to. If ``None``, a uniform distribution is assumed.

        Returns: The ``CoordPDF`` created, containing the ratio of ``self`` to ``coordPDF``.
        """

        coordPDF = coordPDF.pd
        for axis in coordPDF.index.get_level_values(0).unique():
            coordPDF.loc[[axis, slice(None)], ['p']] = 1.0 / coordPDF.loc[[axis, slice(None)], ['p']].shape[0]
        return CoordPDF(self.mkdir(path), (self.pd / coordPDF).fillna(0.0))

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        if 'i' not in designMatrix.pd.columns.get_level_values(0):
            designMatrix.pd.insert(0, ('i','i'), '0')
        design = designMatrix.pd['i']
        design.insert(design.shape[1], DesignMatrix.coordSeparator, 1.0 / design.shape[0])
        design = {axis: design.loc[:, [axis, DesignMatrix.coordSeparator]].set_index(axis).groupby(level=0).sum()
                        for axis in design.columns[:-1]}
        pdf = pd.concat(design.values(), axis=0).rename(columns={DesignMatrix.coordSeparator: 'p'})
        pdf.index = pd.MultiIndex.from_tuples([(axis, coord) for axis in design.keys()
                                               for coord in design[axis].index], names=['axis', 'coord'])
        return cls(cls.mkdir(path), pdf)


class CoordStats(Stats):

    readOptions: MetaData = Table.readOptions | {'header': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    def diff(self, path: Store.Path, designMatrix: CoordDesign, pDF: CoordPDF) -> Self:
        """ The difference between the statistics of ``self`` and the statistics of ``design, pDF``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The coordDesign of ``self`` and ``pDF``.
            pDF: The coordPDF to compare to.

        Returns: The ``CoordStats`` created, containing ``self`` minus the stats of ``design`` under ``pDF``.
        """

    @classmethod
    def create(cls, path: Store.Path, designMatrix: CoordDesign) -> Self:
        stats = designMatrix.pd.drop(columns=['i'], level=0)
        stats = stats.agg(['min', 'max', 'mean', 'std'])
        return cls(cls.mkdir(path), stats)


class PointDesign(DesignMatrix):
    """ The internal format of ``DesignMatrix``, which is thin (has few columns), and only one header row.
    Categorical axes are concatenated into a single column of categorical points."""

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        match designMatrix:
            case PointDesign():
                # If already a PointDesign, just copy it.
                pointDesign = designMatrix.pd
            case CoordDesign():
                # Reformat the CoordDesign to a PointDesign.
                coordDesign = designMatrix.pd.rename(cls.axisType, axis='columns', level=0)
                # Strip out the ``inputAxes`` from ``designMatrix``.
                inputAxes = {'x': None, 'i': None}
                for key in inputAxes.keys():
                    inputAxes[key] = coordDesign.get(key)
                    if inputAxes[key] is not None: coordDesign.drop(columns=key, level=0, inplace=True)
                coordDesign.columns = coordDesign.columns.droplevel(0)
                # Return the categorical inputs to ``designMatrix``.
                if inputAxes['i'] is not None:
                    pointCoord = cls.coordSeparator.join(['l'] + inputAxes['i'].columns.to_list())
                    coordDesign[pointCoord] = inputAxes['i'].astype(str).agg(cls.coordSeparator.join, axis=1)
                    pointDesign = coordDesign.reset_index(names='n').melt(id_vars=['n', pointCoord],
                                                                          var_name='l',
                                                                          value_name=cls.coordSeparator,
                                                                          ignore_index=True).dropna()
                    pointDesign[pointCoord] = (pointDesign
                                              ['l'].astype(str) + cls.coordSeparator +
                                               pointDesign[pointCoord])
                    pointDesign.drop(columns=['l'], inplace=True)
                else:
                    pointCoord = 'l'
                    pointDesign = coordDesign.reset_index(names='n').melt(id_vars=['n'], var_name='l',
                                                                          value_name=cls.coordSeparator,
                                                                          ignore_index=True).dropna()
                pointDesign = pointDesign.rename(columns={cls.coordSeparator: 'y'})
                # Return the continuous inputs to ``designMatrix``.
                if inputAxes['x'] is not None:
                    columns = ['n'] + inputAxes['x'].columns.to_list() + [pointCoord, 'y']
                    pointDesign = pointDesign.join(inputAxes['x'], on='n', how='left').reindex(columns=columns)
            case _:
                raise NotImplementedError(f'I do not know how to create a PointDesign from {type(designMatrix)}')
        return cls(cls.mkdir(path), pointDesign)


class PointPDF(PDF):

    def ratio(self, path: Store.Path, coordPDF: CoordPDF) -> Self:
        """ The ratio between ``coordPDF`` and ``self``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            coordPDF: The ``CoordPDF`` to compare to.

        Returns: The ``PointPDF`` created.
        """
        ratio = self.pd / PointPDF.pd_from_coordPDF(coordPDF)
        return PointPDF(self.mkdir(path), ratio.fillna(0.0))

    @classmethod
    def create(cls, path: Store.Path, designMatrix: PointDesign) -> Self:

        laxis = designMatrix.pd.columns[lcol]
        pdf = designMatrix.pd.iloc[l]
        pdf.insert(pdf.shape[1], 'p', 1.0 / pdf.shape[0])
        pdf.set_index(laxis, drop=True, inplace=True)
        pdf = pdf.groupby(level=0).sum()
        return cls(cls.mkdir(path), pdf)

    @classmethod
    def pdFromCoordPDF(cls, coordPDF: CoordPDF) -> pd.DataFrame:
        """ Create ``PointPDF.pd`` DataFrame ``coordPDF``, assuming independence of categorical coords

        Args:
            coordPDF: The ``CoordPDF`` to create from.

        Returns: The ``PointPDF.pd`` created.
        """
        coordPDF = coordPDF.pd.reset_index(level=1, names='coord')
        pdf = coordPDF.groupby().apply(list)
        axis = DesignMatrix.coordSeparator.join(pdf.index.tolist())
        pdf = pdf.to_numpy().tolist()
        coord, p = zip(*pdf)
        coord = [DesignMatrix.coordSeparator.join(coords) for coords in product(*coord)]
        p = [np.prod(ps) for ps in product(*p)]
        return pd.DataFrame(p, index=pd.Index(coord, name=axis), columns=['p'])

class PointStats(Stats):

    readOptions: MetaData = Table.readOptions | {'header': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    def diff(self, path: Store.Path, designMatrix: PointDesign, pDF: PointPDF) -> Self:
        """ The difference between the statistics of ``self`` and the statistics of ``design, pDF``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The coordDesign of ``self`` and ``pDF``.
            pDF: The coordPDF to compare to.

        Returns: The ``CoordStats`` created, containing ``self`` minus the stats of ``design`` under ``pDF``.
        """
        y2PDF = self.pdY2PDF(designMatrix, pDF)

    @classmethod
    def pdY2PDF(cls, designMatrix: PointDesign, pDF: PointPDF) -> pd.DataFrame:
        """ The probability weighted ``y, y**2`` calculated from ``design, pDF``.

        Args:
            designMatrix: The PointDesign containing ``y``.
            pDF: The PointPDF of ``y``.

        Returns: The probability weighted ``design`` of ``y, y**2``.
        """
        design = designMatrix.pd
        design.loc[:, DesignMatrix.coordSeparator] = design.pd.iloc[y]
        design.pd.iloc[y] *= pDF.pd['p']
        design.loc[:, DesignMatrix.coordSeparator] *= design.pd.iloc[y]
        return design.fillna(0.0).groupby(level=0).mean()

    @classmethod
    def create(cls, path: Store.Path, designMatrix: PointDesign) -> Self:
        laxis = designMatrix.pd.columns[lcol]
        stats = designMatrix.pd.drop(columns=['n'])
        stats.set_index(laxis, drop=True, inplace=True)
        column = stats.columns
        stat = ['min', 'max', 'mean', 'std']
        stats = stats.groupby(level=0).agg(stat)
        index = pd.MultiIndex.from_product([column, stat], names=['axis', 'stat'])
        return cls(cls.mkdir(path), stats)


class Normalizer(DataBase):
    """ Normalizer of a DesignMatrix. """

    class NamedTables(NamedTuple):
        """ NamedTables in a Normalizer. """

        design: DesignMatrix | Matrix = PointDesign
        """The DesignMatrix of user data."""
        pDF: PDF | Matrix = PointPDF
        """The Probability Density Function of categorical coords or points."""
        stats: Table | Matrix = PointStats
        """The statistics of continuous coordinates by categorical coord or point."""
        pDFRatio: PDF | Matrix = PointPDF
        """The Probability Density Function ratio between rational and empirical PDFs."""
        statsDiff: Table | Matrix = PointStats
        """The difference between rational and empirical stats."""

        def __call__(self, name: str) -> Table | Matrix:
            return getattr(self, name)

    Tables: NamedTables[type[Table], ...] = NamedTables()
    """ Table Types, to communicate ``Table.readOptions`` and ``Table.writeOptions``. """

    defaultMeta: MetaData = {'Tables': {name: TableType.__name__ for name, TableType in Tables._asdict().items()}}
    """ Default ``self.meta``. """

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``designMatrix: PointDesign | CoordDesign``. """

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix, **meta: Any) -> Self:
        """ Create a ``Normalizer`` in ``path``.

        Args:
            path: The folder to store the ``Normalizer`` in. Need not exist.
            designMatrix: the ``DesignMatrix`` to normalize, either a ``PointDesign`` or a ``CoordDesign``.
            **meta: ``self.meta`` to update. In particular, ``isUniform = True`` infers a uniform distribution
                for each categorical coord, discarding ``isIndependent`` as tautologically ``False``.
                On the other hand, ``isDependent = True`` infers a ``PointPDF`` from a
                ``PointDesign``. If both are ``False`` or absent, the ``PointPDF`` is inferred from the
                mutually independent ``CoordPDF``s inferred from the ``CoordDesign``.

        Returns: The Normalization created.
        """
        isUniform = {'isUniform': True} if meta.pop('isUniform', False) else {}
        isDependent = {'isDependent': True} if meta.pop('isDependent', False) and not isUniform else {}
        meta = Meta.create(cls._meta_in(path), **(cls.defaultMeta | meta | isUniform))
        path = meta.path.parent
        if cls is Normalizer:
            coord = CoordNormalizer.create(path / 'coord', designMatrix, **meta)
            point = PointNormalizer.create(path / 'point', designMatrix, **meta)
            meta = meta(**isDependent)
            tables = {name: TableType.copy(point[name], path / name)
                            for name, TableType in cls.Tables._asdict().items()}
            # tables = {name: PointPDF.createFromCoordPDF(path / name, coord[name])
            # if TableType is PDF and not isDependent
            # else TableType.copy(point[name], path / name)
            #           for name, TableType in cls.Tables._asdict().items()}
        else:
            # tables = {name: CoordPDF.create(path / name, designMatrix, **isUniform) if TableType is CoordPDF
            #                 else TableType.create(path / name, designMatrix)
            #           for name, TableType in cls.Tables._asdict().items()}
            tables = cls.Tables._asdict()
            design = next(iter(tables))
            Design = tables.pop(design)
            designMatrix = Design.create(path/ design, designMatrix)
            tables = {name: TableType.create(path / name, designMatrix)
                      for name, TableType in tables.items()}
        return cls(path)

class CoordNormalizer(Normalizer):
    """ Normalizer of a DesignMatrix. """

    NamedTables: type[NamedTuple] = Normalizer.NamedTables

    Tables: NamedTables[type[Table], ...] = NamedTables(design=CoordDesign, pDF=CoordPDF, stats=CoordStats)
    """ Table Types, to communicate ``Table.readOptions`` and ``Table.writeOptions``. """


class PointNormalizer(Normalizer):
    """ Normalizer of a DesignMatrix. """

    NamedTables: type[NamedTuple] = Normalizer.NamedTables

    Tables: NamedTables[type[Table], ...] = NamedTables(design=PointDesign, pDF=PointPDF, stats=PointStats)
    """ Table Types, to communicate ``Table.readOptions`` and ``Table.writeOptions``. """


# class Normalizer(DataBase):
#     """ Normalizer of a Repo. """
#     class NamedTables(NamedTuple):
#
#         data: DesignMatrix | MetaData = pd.DataFrame(columns=('x', 'l', 'y'))
#
#         def __call__(self, name: str) -> Table | Matrix | MetaData:
#             """ Returns the Table named ``name``."""
#             return getattr(self, name)
#
#     options: NamedTables[MetaData] = NamedTables(data = DesignMatrix.defaultOptions)
#
#     defaultMetaData: MetaData = {'category delimiter' : '│'}
#
#     def __call__(self, **meta: Any) -> Self:
#         """ Optimize and update ``self``.
#
#         Args:
#             **meta: Optimization ``MetaData``.
#
#         Returns: ``self``
#         """
#         self._tables(**meta)
#         return self
#
#     def __init__(self, path: Store.Path, **tables: Table | Pd.DataFrame):
#         super().__init__(path, **tables)
#
#
#     @classmethod
#     def create(cls, path: Store.Path, designMatrix: DesignMatrix, **meta: Any) -> Self:
#         """ Create a ``Normalizer`` in ``path``.
#
#         Args:
#             path: The folder to store the ``Normalizer`` in. Need not exist,
#                 any existing ``Tables`` will be overwritten if it does.
#             designMatrix: The ``DesignMatrix`` to normalize.
#             **meta: Optimization ``MetaData``.
#
#         Returns: The ``Normalizer`` created.
#         """
#         Meta.create(cls._meta_in(path), **(cls.defaultMetaData | meta))
#         return cls(path, data = designMatrix)
#

# class Repo(DataBase):
#     """ A Repository of data and models. Informally a dataset and all the things we'd like to do to it. """
#
#     class NamedTables(NamedTuple):
#
#         data: Table | Matrix | MetaData = pd.DataFrame(columns=('x', 'l', 'y'))
#
#         def __call__(self, name: str) -> Table | Matrix | MetaData:
#             """ Returns the Table named ``name``."""
#             return getattr(self, name)
#
#     options: NamedTables[MetaData] = NamedTables(data = Table.Options.defaults())
#
#     defaultMetaData: MetaData = {'K': 0}
#
#     @property
#     def fold(self):
#         """ The current fold. """
#         return self._fold
#
#     @fold.setter
#     def fold(self, value: int):
#         """ The current fold. A negative value refers test data in the fold numbered ``abs(value)``.
#             In case ``abs(value)`` is 0 or greater ``len(self)-1`` the current fold is ``self``. """
#         self._fold = value
#
#     def __len__(self) -> int:
#         """ 1 + K proper folds in ``self``. """
#         return self._meta['K'] + 1
#
#     def __getitem__(self, fold: int | slice) -> Path | tuple[Path, ...]:
#         """ Indexer returns the ``Path`` (s) to the Folds indexed or sliced by ``fold``. """
#         if isinstance(fold, int):
#             return self.path  if fold == 0 else self.path / f'{abs(fold)}'
#         else:
#             return tuple((self[i] for i in range(len(self))))[fold]
#
#     def __setitem__(self, fold: int | slice , tables: Table | Matrix | tuple[Table | Matrix, ...]):
#         """ Indexer creates the ``Fold`` (s) named or sliced by ``name``."""
#         self[fold] = tables
#
#     def __call__(self, **meta: Any) -> Self:
#         """ Optimize and update ``self``.
#
#         Args:
#             **meta: Optimization ``MetaData``.
#
#         Returns: ``self``
#         """
#         self(**meta)
#         return self
#

