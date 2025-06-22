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
n: Tuple[slice, slice] = (slice(None, None, None), slice(None, 1, None))

#: Slice for ``x`` (inputs) in a Repo Table.
x: Tuple[slice, slice] = (slice(None, None, None), slice(1, -2, None))

#: Slice for ``l`` (categorical state) in a Repo Table.
l: Tuple[slice, slice] = (slice(None, None, None), slice(-2, -1, None))

#: Slice for ``y`` (output) in a Repo Table.
y: Tuple[slice, slice] = (slice(None, None, None), slice(-1, None, None))


class DesignMatrix(Table):
    """ The familiar user format of ``DesignMatrix`` which is fat (has many columns). """

    Label = Path | str
    """ Class attribute aliasing acceptable Types for column (or index) labels. """

    class Options(NamedTuple):

        read: MetaData =  {'index_col': 0, 'header': [0, 1]}  # Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   # Write options passed to ``pd.DataFrame.to_csv``.

    skeleton: Pd.DataFrame = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
                                    (('Input', 'float'),('Category', 'int'),
                                     ('Column', 'str'), ('Output', 'func'))))
    """ DataFrame of the minimal, skeleton ``DesignMatrix``."""

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
        return '~'

    @classmethod
    def create(cls, path: Store.Path, src: NormalDesignMatrix, columns_in_l: Label = '') -> Self:
        """ Reformat the ``NormalDesignMatrix`` in ``src`` as a ``Self(DesignMatrix)``.

        Args:
            path: The ``Path`` to store the ``DesignMatrix`` created, overwritten if existing.
            src: The ``NormalDesignMatrix`` to reformat.

        Returns: The ``NormalDesignMatrix`` created at ``dst``.
        """

    @classmethod
    def copy(cls, src: Self, dst: Store.Path = '') -> NormalDesignMatrix:
        """ Reformat this ``DesignMatrix`` to a ``NormalDesignMatrix``.

        Args:
            src: The ``DesignMatrix`` to reformat.
            dst: Optional ``Path`` to the ``NormalDesignMatrix``.
                Defaults to ``''``, which overwrites ``src``.

        Returns: The ``NormalDesignMatrix`` created at ``dst``.
        """


class NormalDesignMatrix(DesignMatrix):
    """ The internal format of ``DesignMatrix``, which is thin (has few columns). """

    class Options(NamedTuple):

        read: MetaData =  {'index_col': 0, 'header': 0}  # Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   # Write options passed to ``pd.DataFrame.to_csv``.

    def create(cls, path: Store.Path, src: NormalDesignMatrix) -> Self:
        return cls(path, update= src)

    def copy(cls, src: Self, dst: Store.Path = '') -> NormalDesignMatrix:
        return cls(dst, update= src) if dst else src


class Normalization(DataBase):
    """ Normalization of a Repo. """
    class NamedTables(NamedTuple):

        data: DesignMatrix | MetaData = DesignMatrix.skeleton

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

    def __init__(self, path: Store.Path, **tables: Table | PD.DataFrame):
        super().__init__(path, **tables)


    @classmethod
    def create(cls, path: Store.Path, data: DesignMatrix, **meta: Any) -> Self:
        """ Create a ``Normalization`` in ``path``.

        Args:
            path: The folder to store the ``Normalization`` in. Need not exist,
                any existing ``Tables`` will be overwritten if it does.
            **meta: Optimization ``MetaData``.

        Returns: The ``Normalization`` created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMetaData | meta))
        return cls(path, data = data)


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

    def __getitem__(self, fold: int | slice) -> Path | Tuple[Path, ...]:
        """ Indexer returns the ``Path`` (s) to the Folds indexed or sliced by ``fold``. """
        if isinstance(fold, int):
            return self.path  if fold == 0 else self.path / f'{abs(fold)}'
        else:
            return tuple((self[i] for i in range(len(self))))[fold]

    def __setitem__(self, fold: int | slice , tables: Table | Matrix | Tuple[Table | Matrix, ...]):
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

