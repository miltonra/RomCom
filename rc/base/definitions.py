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

""" Basic Protocols, Types and constants.

All modules of RomCom ``import *`` from ``rc.definitions``, so all types and constants in this module are referenced
without adornment throughout RomCom. The ``rc.definitions`` namespace includes::

    from typing import *
    from collections.abc import *
    from abc import ABC, abstractmethod
    import polars as pl
    import numpy as np
    import torch as tc
    import unittest as ut
    from inspect import stack
    from pathlib import Path
"""

from typing import *
from collections.abc import *
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
import polars as pl
import numpy as np
import torch as tc
import unittest as ut
from inspect import stack
from pathlib import Path
from warnings import warn

import pandas as pd


zero: float = 1.0E-64
"""Tolerance when testing floats for equality."""


Int: TypeAlias = pl.Int32
"""Alias ``pl.Int32``. The Type of every ``int``."""


Ints: tuple[Type, ...] = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128,)
"""Every Type of ``int``."""


Float: TypeAlias = pl.Float32
"""Alias ``pl.Float32``. The Type of every ``float``."""


Floats: tuple[Type, ...] = (pl.Float16, pl.Float32, pl.Float64,)
"""Every Type of ``float``."""


String: TypeAlias = pl.String
"""Alias ``pl.String``."""


PathLike: TypeAlias = Path | str
"""Alias ``Path | str | ``. ArgumentType of the ``path`` to a Store."""


IndexLike: TypeAlias = str | int | Iterable[str | int] | slice
"""Alias ``str | int | Iterable[str | int] | slice``. ArgumentType of the *IndexP Protocol*."""


DataFrame: TypeAlias = pl.DataFrame
"""Alias ``pl.DataFrame``. """


Category: TypeAlias = pl.Categorical
"""Alias `pl.Categorical <https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/#data-type-categorical>`__."""


MetaData: TypeAlias = Mapping[str, Any]
"""Alias ``Mapping[str, Any]``. ArgumentType and ReturnType of Meta. """


class Protocol(BaseException):
    """ The Protocol from which all Protocols derive. Any ``cls`` defines SubClasses of Protocol to document its API, especially dunder methods."""
    def __init__(self, *args, **kwargs):
        """

        :meta private:
        """
        raise NotImplementedError('Abstract Class, do not instantiate.')


class NameP(Protocol):
    """``str(self) = str(self.path.name)`` and ``repr(self) = str(self.path)``. """


class IndexP(Protocol):
    """ ``self[key: IndexLike]`` is not implemented. Override ``__getitem__(self, key)``, ``__setitem__(self, key, value)`` and ``__len__(self)``. """

    Index: TypeAlias = tuple[int, ...]
    """Alias ``tuple[int, ...]``. Return Type of the IndexP *Protocol*."""


class EqualsP(Protocol):
    """ Not implemented. """


class CreateP(Protocol):
    """``cls.create(path)`` is selective, preserving irrelevant items in ``path``. """


class ReadP(Protocol):
    """``cls(path)`` reads from ``path``. """


class UpdateP(Protocol):
    """``self(**updates)`` updates ``self`` then writes to ``self.path``. """


class DeleteP(Protocol):
    """``cls.delete(path)`` is selective, preserving irrelevant items in ``path``. """


class CopyP(Protocol):
    """``cls.copy(src, dst)`` is selective, copying only relevant items in ``src`` while preserving irrelevant items in ``dst``."""


class Np:
    """ Extended NumPy types and constants. This Class should never be instantiated or SubClassed.

    Attributes:
        DType = np.dtype
        Array = np.ndarray
        Tensor = Array
        Vector = Tensor[i,1]
        CoVector = Tensor[1,j]
        Matrix = Tensor[i,j]
    """
    DType = np.dtype    #: :meta private:
    Array = np.ndarray  #: :meta private:
    Tensor = Array      #: :meta private:
    Vector = Tensor     #: :meta private:
    CoVector = Tensor   #: :meta private:
    Matrix = Tensor     #: :meta private:

    def __init__(self):
        raise NotImplementedError('Abstract Class, do not instantiate.')


class Tc:
    """ Extended PyTorch types and constants. This Class should never be instantiated or SubClassed.

    Attributes:
        DType = tc.dtype
        Tensor = tc.Tensor
        Vector = Tensor[i,1]
        CoVector = Tensor[1,j]
        Matrix = Tensor[i,j]
        BatchVector = Tensor[...,i,1]
        BatchCoVector = Tensor[...,1,j]
        BatchMatrix = Tensor[...,i,j]
    """
    DType = tc.dtype    #: :meta private:
    Tensor = tc.Tensor  #: :meta private:
    Vector = Tensor     #: :meta private:
    CoVector = Tensor   #: :meta private:
    Matrix = Tensor     #: :meta private:
    BatchVector = Tensor     #: :meta private:
    BatchCoVector = Tensor   #: :meta private:
    BatchMatrix = Tensor     #: :meta private:
    Int: DType = tc.int32
    Float: DType = tc.float32
    zero: Tensor = tc.tensor(zero, dtype = Float)  #: :meta private:

    def __init__(self):
        raise NotImplementedError('This Class is not intended to be instantiated or SubClassed.')


TableData: TypeAlias = DataFrame | Np.Matrix | Tc.Matrix
"""Alias ``DataFrame | Np.Matrix | Tc.Matrix``. ArgumentType of a Table."""


class Test:

    root = Path('../tst')

    @classmethod
    def folder(cls) -> Path:
        frame = stack()[1]
        package, caller = Path(frame[1]).parent.name, frame[3]
        if caller.startswith("test_"): caller = caller[5:]
        return cls.root / package / caller

