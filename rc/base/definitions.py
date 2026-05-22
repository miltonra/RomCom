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
    from copy import copy, deepcopy
    from inspect import stack
    from pathlib import Path
    import polars as pl
    import numpy as np
    import torch as tc
    import unittest as ut
"""

from typing import *
from collections.abc import *
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from inspect import stack
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
import torch as tc
import unittest as ut


zero: float = 1.0E-64
"""Tolerance when testing floats for equality."""


PathLike: TypeAlias = Path | str
""" = ``Path | str``. Class attribute aliasing valid Types for specifying the ``path`` to a Store."""


class Protocol(BaseException):
    """ The Protocol from which all Protocols derive. Any ``cls`` defines SubClasses of Protocol to document its API, especially dunder methods."""
    def __init__(self, *args, **kwargs):
        """

        :meta private:
        """
        raise NotImplementedError('Abstract Class, do not instantiate.')

class IndexP(Protocol):
    """ ``self[key]`` is not implemented. Override ``__getitem__(self, key)``, ``__setitem__(self, key, value)`` and ``__len__(self)``. """
    Index: TypeAlias = str | int | Iterable[str | int] | slice
    """ = ``str | int | Iterable[str | int] | slice``. Types of index supported by ``IndexP``. """

class EqualityP(Protocol):
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

class StrReprP(Protocol):
    """``str(self) = str(self.path.name)`` and ``repr(self) = str(self.path)``. """



class Pl:
    """ Extended Polars types and constants. This Class should never be instantiated or SubClassed.

    Attributes:
        DataFrame = pl.DataFrame
    """
    DataFrame = pl.DataFrame    #: :meta private:

    def __init__(self):
        raise NotImplementedError('Abstract Class, do not instantiate.')


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
    Float: DType = tc.float64
    zero: Tensor = tc.tensor(zero, dtype = Float)  #: :meta private:

    def __init__(self):
        raise NotImplementedError('This Class is not intended to be instantiated or SubClassed.')

class Test:

    root = Path('../tst')

    @classmethod
    def folder(cls) -> Path:
        frame = stack()[1]
        package, caller = Path(frame[1]).parent.name, frame[3]
        if caller.startswith("test_"): caller = caller[5:]
        return cls.root / package / caller

