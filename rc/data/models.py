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



#: Column containing ``n`` (row) in a TableOfStates.
ncol: int = 0

#: Column containing ``l`` (state) in a TableOfStates.
lcol: int = -2

#: Slice for ``n`` (row) in a TableOfStates.
n: tuple[slice, slice] = (slice(None, None), slice(ncol, ncol + 1))

#: Slice for ``x`` (continuous inputs) in a TableOfStates.
x: tuple[slice, slice] = (slice(None, None), slice(ncol + 1, lcol))

#: Slice for ``l`` (categorical state) in a TableOfStates.
l: tuple[slice, slice] = (slice(None, None), slice(lcol, lcol + 1))

#: Slice for ``y`` (output) in a TableOfStates.
y: tuple[slice, slice] = (slice(None, None), slice(lcol + 1, None))


class DesignMatrix(VariableTable):
    """ The familiar user format of a DesignMatrix which is fat (has many columns), and two header rows.
    The first header row contains the column kinds, the second header row contains the column names."""

    @classmethod
    def kinds(cls, label: str) -> str:
        match label.lower():
            case 'x' | 'input' | 'continuous' | 'float' :
                return 'x'
            case 'i' | 'category' | 'discrete' | 'int' :
                return 'i'
            case 'y' | 'output' | 'map' | 'func' :
                return 'y'
        return '?'

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        """ Create a ``DesignMatrix`` at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The ``DesignMatrix`` to reformat if necessary and store in ``path``.

        Returns: The ``DesignMatrix`` created.
        """
        if not isinstance(designMatrix, StateMatrix):
            designMatrix = designMatrix.pd.rename(cls.kinds, axis='columns', level=0)
            if 'i' in designMatrix.columns.get_level_values(0):
                designMatrix['i'] = designMatrix['i'].astype(str)
            return cls(cls.mkdir(path), designMatrix)
        # Reformat the stateMatrix to a DesignMatrix.
        stateMatrix = designMatrix.pd.set_index('n')
        stateMatrix[stateMatrix.columns[lcol]] = stateMatrix[stateMatrix.columns[lcol]].astype(str)
        # convert states to variables.
        result = {'i': stateMatrix.iloc[:,lcol].apply(lambda state: state.split(StateMatrix.variableDelimiter))}
        result['i'] = pd.DataFrame(result['i'].tolist(), index = stateMatrix.index,
                                   columns = result['i'].name.split(StateMatrix.variableDelimiter))
        # convert first categorical variable to output dimensions.
        ycols = result['i']['l'].drop_duplicates().tolist()
        result['y'] = pd.concat([result['i']['l'], stateMatrix.iloc[:, lcol + 1]], axis = 1)
        for ycol in ycols:
            result['y'][ycol] = (result['y']['l'] == ycol).astype(int) * result['y'].iloc[:, 1]
        # Collect x, i, and y column groups.
        result['all'] = pd.concat([stateMatrix.iloc[:, :lcol].groupby(level = 0, sort=False).mean() ,
                                   result['i'].iloc[:, 1:].groupby(level = 0, sort=False).first(),
                                   result['y'].iloc[:, 2:].groupby(level = 0, sort=False).sum(),],
                                  axis = 1)
        # Label column groups.
        result['all'].columns = pd.MultiIndex.from_tuples([('x', col) for col in stateMatrix.columns[:lcol]] +
                                                          [('i', col) for col in result['i'].columns[1:]] +
                                                          [('y', col) for col in ycols])
        # Detect NaN.
        result['all']['y'] = result['all']['y'].replace(0.0, np.nan)
        return cls(cls.mkdir(path), result['all'])

    @classmethod
    def copy(cls, src: DesignMatrix, dst: Store.Path) -> Self:
        """ Copy ``src`` to ``dst``, overwriting.

        Args:
            src: The source ``Table``.
            dst: The destination ``Path``, overwritten if existing.
                A ``.csv`` extension is automatically appended.

        Returns: The ``Table`` now stored at ``dst``.
        """
        return cls.create(dst, src)


class StateMatrix(DesignMatrix):
    """ The internal format of ``DesignMatrix``, which is thin (has few columns), and only one header row."""

    readOptions: MetaData = StateTable.readOptions
    """ File read options passed directly to `pd.read_csv <https: //pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    writeOptions: MetaData = StateTable.writeOptions
    """ File write options passed directly to `pd.DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`__."""

    class CreateP(CreateP):
        """ Creates a new ``DesignMatrixByState`` at ``path`` from a ``DesignMatrix``. """

    variableDelimiter: str = '│'
    """ The seperator used to delimit variables in the column of categorical states. """

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        """ Create a ``DesignMatrixByState`` at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The ``DesignMatrix`` to reformat if necessary and store in ``path``.

        Returns: The ``DesignMatrixByState`` created.
        """
        if isinstance(designMatrix, StateMatrix):
            return cls(cls.mkdir(path), designMatrix) # If already a StateMatrix, just copy it.

        # Reformat the DesignMatrix to a StateMatrix.
        designMatrix = designMatrix.pd.rename(cls.kinds, axis = 'columns', level = 0)
        # Strip out the inputs from ``designMatrix`` and keep them in ``inputs``.
        inputs = {'x': None, 'i': None}
        for key in inputs.keys():
            inputs[key] = designMatrix.get(key)
            if inputs[key] is not None: designMatrix.drop(columns = key, level = 0, inplace = True)
        designMatrix.columns = designMatrix.columns.droplevel(0)
        # Return the categorical inputs to ``designMatrix``.
        if inputs['i'] is not None:
            state = cls.variableDelimiter.join(['l'] + inputs['i'].columns.to_list())
            designMatrix[state] = inputs['i'].astype(str).agg(cls.variableDelimiter.join, axis = 1)
            designMatrix = designMatrix.reset_index(names='n').melt(id_vars = ['n', state],
                                                var_name = 'l', value_name = 'y', ignore_index = True).dropna()
            designMatrix[state] = designMatrix['l'].astype(str) + cls.variableDelimiter + designMatrix[state]
            designMatrix.drop(columns = ['l'], inplace = True)
        else:
            state = 'l'
            designMatrix = designMatrix.reset_index(names='n').melt(id_vars=['n'],
                                                var_name='l', value_name='y', ignore_index=True).dropna()
        # Return the continuous inputs to ``designMatrix``.
        if inputs['x'] is not None:
            columns = ['n'] + inputs['x'].columns.to_list() + [state, 'y']
            designMatrix = designMatrix.join(inputs['x'], on = 'n', how = 'left').reindex(columns = columns)
        return cls(cls.mkdir(path), designMatrix)


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
