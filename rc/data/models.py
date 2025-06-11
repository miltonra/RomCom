#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2025 Robert A. Milton. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#  following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#  following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
#  promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Models for data storage. """

from __future__ import annotations

from torchvision.transforms.v2 import Normalize

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

    skeleton: PD.DataFrame = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
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
        return cls(path, data = src)

    def copy(cls, src: Self, dst: Store.Path = '') -> NormalDesignMatrix:
        return cls(dst, data = src) if dst else src


class Normalization(DataBase):
    """ Normalization of a Repo. """
    class Tables(Tables):

        class NT(NamedTuple):

            data: DesignMatrix | MetaData = DesignMatrix.skeleton

            def __call__(self, name: str) -> Table | Matrix | MetaData:
                """ Returns the Table named ``name``."""
                return getattr(self, name)

        options: NT[MetaData] = NT(data = DesignMatrix.defaultOptions)

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
    class Tables(Tables):

        class NT(NamedTuple):

            data: Table | Matrix | MetaData = pd.DataFrame(columns=('x', 'l', 'y'))

            def __call__(self, name: str) -> Table | Matrix | MetaData:
                """ Returns the Table named ``name``."""
                return getattr(self, name)

        options: NT[MetaData] = NT(data = {option: value for default in Table.Options._field_defaults.values()
                                           for option, value in default.items()})

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
        self._tables[name] = tables

    def __call__(self, **meta: Any) -> Self:
        """ Optimize and update ``self``.

        Args:
            **meta: Optimization ``MetaData``.

        Returns: ``self``
        """
        self._tables(**meta)
        return self

