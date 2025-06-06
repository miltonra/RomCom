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
    """ Template for a design matrix. """

    class Options(NamedTuple):

        read: MetaData =  {'index_col': 0, 'header': [0, 1]}  # Read options passed to ``pd.read_csv``.
        write: MetaData =  {}   # Write options passed to ``pd.DataFrame.to_csv``.

        def __call__(self, name: str) -> Table | Matrix | MetaData:
            """ Returns the Table named ``name``."""
            return getattr(self, name)

    skeleton: PD.DataFrame = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
                                    (('Input', 'float'),('Category', 'int'), ('Output', 'float'))))
    """ DataFrame of the minimal, skeleton ``DesignMatrix``."""

    defaultOptions: MetaData = Options().read | Options().write
    """ Default file handling ``DesignMatrix.Options()``."""


class Fold(DataBase):
    """ A fold of training and test data. """

    class Tables(Tables):

        class NT(NamedTuple):

            train: DesignMatrix | MetaData = DesignMatrix.skeleton
            test: DesignMatrix | MetaData = DesignMatrix.skeleton

            def __call__(self, name: str) -> Table | Matrix | MetaData:
                """ Returns the Table named ``name``."""
                return getattr(self, name)

        options: NT[MetaData] = NT(train=DesignMatrix.defaultOptions, test=DesignMatrix.defaultOptions)

    defaultMetaData: MetaData = {'M': 1}

    @classmethod
    def create(cls, path: Store.Path, train: DesignMatrix, test: DesignMatrix = DesignMatrix.skeleton,
               **metadata: MetaData) -> Self:
        """

        Args:
            path: The folder to store the ``Fold`` in. Need not exist, any existing ``train.csv`` or ``test.csv``
                will be overwritten if it does.
            train:
            test:
            **metadata:

        Returns: ``self``.

        """





        """ Create a ``DataBase`` in ``path``.

        Args:
            **tables_and_meta: Data to update ``cls.Tables.table_defaults``, in the form ``names[i]=tables[i]``,
                and optional ``MetaData`` to update ``cls.defaultMetaData`` in the form ``meta=MetaData``.

        Returns: The ``DataBase`` created.
        """
        return super().create(path, train = train, test = test, meta = cls.defaultMetaData | metadata)


class Repo(Fold):
    """ A Repository housing a ``Normalization``. and ``K`` ``Fold`` (s). """

    defaultMetaData: MetaData = {'options': Tables.options._asdict(), 'Normalization': None, 'K': 0}

    def __len__(self) -> int:
        """ Counts the ``Fold`` s in ``self``. """
        return self._meta['K']

    def __getitem__(self, fold: int | slice) -> Fold | Tuple[int, ...]:
        """ Indexer returns the ``Fold`` (s) named or sliced by ``name``. """
        if isinstance(fold, int):
            return Fold(self.path / f'{fold}')
        else:
            return tuple(range(self._meta['K']))[fold]

    def __setitem__(self, name: int | slice , tables: Table | Matrix | Tuple[Table | Matrix, ...]):
        """ Indexer creates the ``Fold`` (s) named or sliced by ``name``."""
        self._tables[name] = tables

    def __call__(self, **metadata: Any) -> Self:
        """ Optimize and update ``self``.

        Args:
            **metadata: Optimization ``MetaData``.

        Returns: ``self``
        """
        self._tables(**metadata)
        return self



"""
    defaultMetaData: MetaData = {'options': Tables.options._asdict(),
                                 'split':'|', 'Method': 'Mean and SD', 'Bounds': (0.0, 0.0)}

    class Method(IntEnum):
        Explicit = 0
        Range = 1
        Moments = 2
"""