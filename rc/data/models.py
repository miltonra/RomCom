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

from rc.base import *

from copy import deepcopy
import itertools
import random
import shutil
import scipy.stats
from enum import IntEnum



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

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``designMatrix: PointDesign | CoordDesign``. """

    axisSeparator: str = '│'
    """ The separator used to delimit axes in a column of categorical Points. """

    @classmethod
    def axisType(cls, axis: str) -> str:
        
        match axis.lower():
            case 'x' | 'input' | 'continuous' | 'float' :
                return 'x'
            case 'i' | 'category' | 'discrete' | 'int' :
                return 'i'
            case 'y' | 'output' | 'map' | 'func' :
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

    @classmethod
    def copy(cls, src: Self, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source ``DesignMatrix``.
            dst: The destination ``Path``, overwritten if existing.
                A ``.csv`` extension is automatically appended.

        Returns: The ``DesignMatrix`` now stored in ``dst``.
        """
        return cls.create(dst, src)


class CoordPDF(Table):
    """ The Probability Density Functions of categorical coordinates in a CoordDesign."""

    readOptions: MetaData = Table.readOptions | {'index': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    writeOptions: MetaData = Table.writeOptions | {'myOption': 'myValue'}
    """ File write options passed directly to
    `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`__."""


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
                iDesign = pointDesign.iloc[:, lcol].apply(lambda point: point.split(cls.axisSeparator))
                iDesign = pd.DataFrame(iDesign.tolist(), index=pointDesign.index,
                                       columns=iDesign.name.split(cls.axisSeparator))
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


class PointPDF(Table):
    """ The Probability Density Function of categorical points in a PointDesign."""

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
                    pointCoord = cls.axisSeparator.join(['l'] + inputAxes['i'].columns.to_list())
                    coordDesign[pointCoord] = inputAxes['i'].astype(str).agg(cls.axisSeparator.join, axis=1)
                    pointDesign = coordDesign.reset_index(names='n').melt(id_vars=['n', pointCoord],
                                                                          var_name='l', value_name='y',
                                                                          ignore_index=True).dropna()
                    pointDesign[pointCoord] = (pointDesign
                                              ['l'].astype(str) + cls.axisSeparator +
                                               pointDesign[pointCoord])
                    pointDesign.drop(columns=['l'], inplace=True)
                else:
                    pointCoord = 'l'
                    pointDesign = coordDesign.reset_index(names='n').melt(id_vars=['n'],
                                                                        var_name='l', value_name='y',
                                                                        ignore_index=True).dropna()
                # Return the continuous inputs to ``designMatrix``.
                if inputAxes['x'] is not None:
                    columns = ['n'] + inputAxes['x'].columns.to_list() + [pointCoord, 'y']
                    pointDesign = pointDesign.join(inputAxes['x'], on='n', how='left').reindex(columns=columns)
            case _:
                raise NotImplementedError(f'I do not know how to create a PointDesign from {type(designMatrix)}')
        return cls(cls.mkdir(path), pointDesign)


class Normalization(DataBase):

    class NamedTables(NamedTuple):

        coordDesign: Table | Matrix | type[Table] = None
        pointDesign: Table | Matrix | type[Table] = None

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)


    Tables: NamedTables[type[Table], ...] = NamedTables(coordDesign=CoordDesign, pointDesign=PointDesign)
    """ Class attribute of the form ``NamedTables(**{names[i]: Tables[i], ...})``. """

    defaultMetaData: MetaData = {'Tables': Tables._asdict()}
    """ Class attribute. Should be overridden."""

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix, **meta: Any) -> Self:
        """ Create a ``Normalization`` in ``path``. """

# class Normalization(DataBase):
#     """ Normalization of a Repo. """
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
#         """ Create a ``Normalization`` in ``path``.
#
#         Args:
#             path: The folder to store the ``Normalization`` in. Need not exist,
#                 any existing ``Tables`` will be overwritten if it does.
#             designMatrix: The ``DesignMatrix`` to normalize.
#             **meta: Optimization ``MetaData``.
#
#         Returns: The ``Normalization`` created.
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

