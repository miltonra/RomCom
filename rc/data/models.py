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


#: Slice for ``n`` (row) in a Repo Table.
n: tuple[slice, slice] = (slice(None, None, None), slice(None, 1, None))

#: Slice for ``x`` (continuous inputs) in a Repo Table.
x: tuple[slice, slice] = (slice(None, None, None), slice(1, -2, None))

#: Slice for ``l`` (categorical state) in a Repo Table.
l: tuple[slice, slice] = (slice(None, None, None), slice(-2, -1, None))

#: Slice for ``y`` (output) in a Repo Table.
y: tuple[slice, slice] = (slice(None, None, None), slice(-1, None, None))


class DesignMatrix(Table):
    """ The familiar user format of ``DesignMatrix`` which is fat (has many columns). """

    class CreateP(CreateP):
        """ Creates a new ``DesignMatrix`` at ``path`` from a ``DesignMatrix``. """

    Label = Path | str
    """ Class attribute aliasing acceptable Types for column (or index) labels. """

    class Options(NamedTuple):

        read: MetaData =  {'index_col': 0, 'header': [0, 1]}  # Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   # Write options passed to ``pd.DataFrame.to_csv``.

    defaultOptions: MetaData = Options().read | Options().write
    """ Default file handling ``DesignMatrix.Options()``."""

    @classmethod
    def headers(cls, label: str) -> str:
        match label.lower():
            case 'x' | 'input' | 'continuous' | 'float' :
                return 'x'
            case 'i' | 'category' | 'discrete' | 'int' :
                return 'i'
            case 'l' | 'column' | 'label' | 'str' :
                return 'l'
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
        if not isinstance(designMatrix, DesignMatrix00):
            return cls(cls.mkdir(path), designMatrix)
        pd = designMatrix.pd
        inputs = {'x': pd.iloc[:, -2], 'i': None}
        pd.take([0,-2, -1], axis = 1, inplace = True)

        return cls(cls.mkdir(path), pd)


class DesignMatrix00(DesignMatrix):
    """ The internal format of ``DesignMatrix``, which is thin (has few columns), and only one header row."""

    class CreateP(CreateP):
        """ Creates a new ``DesignMatrix00`` at ``path`` from a ``DesignMatrix``. """

    class Options(NamedTuple):

        read: MetaData =  {'index_col': 0, 'header': 0}  # Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   # Write options passed to ``pd.DataFrame.to_csv``.

    categoryDelimiter: str = '│'
    """ The delimiter used to separate categories in a categorical column. """

    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix) -> Self:
        """ Create a ``DesignMatrix00`` at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            designMatrix: The ``DesignMatrix`` to reformat if necessary and store in ``path``.

        Returns: The ``DesignMatrix00`` created.
        """
        if isinstance(designMatrix, DesignMatrix00):
            return cls(cls.mkdir(path), designMatrix) # If already a DesignMatrix00, just copy it.

        # Reformat the DesignMatrix to a DesignMatrix00.
        pd = designMatrix.pd.rename(cls.headers, axis = 'columns', level = 0)
        # Strip out the inputs from ``pd`` and keep them in ``inputs``.
        inputs = {'x': None, 'i': None}
        for key in inputs.keys():
            inputs[key] = pd.get(key)
            if inputs[key] is not None: pd.drop(columns = key, level = 0, inplace = True)
        pd.columns = pd.columns.droplevel(0)
        # Return the categorical inputs to ``pd``.
        if inputs['i'] is not None:
            state = cls.categoryDelimiter.join(['l'] + inputs['i'].columns.to_list())
            pd[state] = inputs['i'].astype(str).agg(cls.categoryDelimiter.join, axis = 1)
            pd = pd.reset_index(names='n').melt(id_vars = ['n', state],
                                                var_name = 'l', value_name = 'y', ignore_index = True).dropna()
            pd[state] = pd['l'].astype(str) + cls.categoryDelimiter + pd[state]
            pd.drop(columns = ['l'], inplace = True)
        else:
            state = 'l'
            pd = pd.reset_index(names='n').melt(id_vars=['n'],
                                                var_name='l', value_name='y', ignore_index=True).dropna()
        # Return the continuous inputs to ``pd``.
        if inputs['x'] is not None:
            columns = ['n'] + inputs['x'].columns.to_list() + [state, 'y']
            pd = pd.join(inputs['x'], on = 'n', how = 'left').reindex(columns = columns)
        return cls(cls.mkdir(path), pd)


class Normalization(DataBase):
    """ Normalization of a Repo. """
    class NamedTables(NamedTuple):

        data: DesignMatrix | MetaData = pd.DataFrame(columns=('x', 'l', 'y'))

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)

    options: NamedTables[MetaData] = NamedTables(data = DesignMatrix.defaultOptions)

    defaultMetaData: MetaData = {'category delimiter' : '│'}

    def __call__(self, **meta: Any) -> Self:
        """ Optimize and update ``self``.

        Args:
            **meta: Optimization ``MetaData``.

        Returns: ``self``
        """
        self._tables(**meta)
        return self

    def __init__(self, path: Store.Path, **tables: Table | Pd.DataFrame):
        super().__init__(path, **tables)


    @classmethod
    def create(cls, path: Store.Path, designMatrix: DesignMatrix, **meta: Any) -> Self:
        """ Create a ``Normalization`` in ``path``.

        Args:
            path: The folder to store the ``Normalization`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            designMatrix: The ``DesignMatrix`` to normalize.
            **meta: Optimization ``MetaData``.

        Returns: The ``Normalization`` created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMetaData | meta))
        return cls(path, data = designMatrix)


class Repo(DataBase):
    """ A Repository of data and models. Informally a dataset and all the things we'd like to do to it. """

    class NamedTables(NamedTuple):

        data: Table | Matrix | MetaData = pd.DataFrame(columns=('x', 'l', 'y'))

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)

    options: NamedTables[MetaData] = NamedTables(data = Table.Options.defaults())

    defaultMetaData: MetaData = {'K': 0}

    @property
    def fold(self):
        """ The current fold. """
        return self._fold

    @fold.setter
    def fold(self, value: int):
        """ The current fold. A negative value refers test data in the fold numbered ``abs(value)``.
            In case ``abs(value)`` is 0 or greater ``len(self)-1`` the current fold is ``self``. """
        self._fold = value

    def __len__(self) -> int:
        """ 1 + K proper folds in ``self``. """
        return self._meta['K'] + 1

    def __getitem__(self, fold: int | slice) -> Path | tuple[Path, ...]:
        """ Indexer returns the ``Path`` (s) to the Folds indexed or sliced by ``fold``. """
        if isinstance(fold, int):
            return self.path  if fold == 0 else self.path / f'{abs(fold)}'
        else:
            return tuple((self[i] for i in range(len(self))))[fold]

    def __setitem__(self, fold: int | slice , tables: Table | Matrix | tuple[Table | Matrix, ...]):
        """ Indexer creates the ``Fold`` (s) named or sliced by ``name``."""
        self[fold] = tables

    def __call__(self, **meta: Any) -> Self:
        """ Optimize and update ``self``.

        Args:
            **meta: Optimization ``MetaData``.

        Returns: ``self``
        """
        self(**meta)
        return self

