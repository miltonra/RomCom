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

""" Basic Protocols, Types and constants.

All modules of RomCom ``import *`` from ``rc.definitions``, so all types and constants in this module are referenced
without adornment throughout RomCom. The ``rc.definitions`` namespace includes::

from typing import *
from abc import ABC, abstractmethod
from pathlib import Path
from copy import copy, deepcopy

"""

from __future__ import annotations

from typing import *
from abc import ABC, abstractmethod
from pathlib import Path
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import torch as tc


zero: float = 1.0E-64
"""Tolerance when testing floats for equality."""


class Protocol(Exception):
    """ The Protocol from which all Protocols derive. Any ``cls`` defines sub classes of Protocol to document its API, especially dunder methods."""

class Indexed(Protocol):
    """ ``self[key]`` is not implemented. Override ``__getitem__(self, key)`` and ``__setitem__(self, key, value)``. """

class Len(Protocol):
    """``len(self)`` is not implemented. Override ``__len__(self)``. """

class Create(Protocol):
    """``cls.create(path)`` is selective, preserving irrelevant items in ``path``. """

class Read(Protocol):
    """``cls(path)`` reads from ``path``. """

class Update(Protocol):
    """``self(**updates)`` updates ``self`` then writes to ``self.path``. """

class Delete(Protocol):
    """``cls.delete(path)`` is selective, preserving irrelevant items in ``path``. """

class Copy(Protocol):
    """``cls.copy(src, dst)`` is selective, copying only relevant items in ``src.path`` while preserving irrelevant items in ``dst``."""

class StrRepr(Protocol):
    """``str(self) = str(self.path.name)`` and ``repr(self) = str(self.path)``. """



class Pd:
    """ Extended Pandas types and constants.

    Attributes:
        DataFrame: pd.DataFrame.
        Index: pd.Index.
        MultiIndex: pd.MultiIndex.
    """
    DataFrame = pd.DataFrame    #: :meta private:
    Index = pd.Index            #: :meta private:
    MultiIndex = pd.MultiIndex  #: :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


class Np:
    """ Extended NumPy types and constants. This class should never be instantiated or subclassed.

    Attributes:
        DType: ``np.dtype``.
        Array: ``np.ndarray``.
        Tensor: ``Array``.
        Vector: Column vector, first order Tensor ``.shape = (i,1)``.
        CoVector = Tensor: Row vector, first order Tensor ``.shape = (1,j)``.
        Matrix = Tensor: Second order Tensor ``.shape = (i,j)``.
    """
    DType = np.dtype    #: :meta private:
    Array = np.ndarray  #: :meta private:
    Tensor = Array      #: :meta private:
    Vector = Tensor     #: :meta private:
    CoVector = Tensor   #: :meta private:
    Matrix = Tensor     #: :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


class Tc:
    """ Extended PyTorch types and constants. This class should never be instantiated or subclassed.

    Attributes:
        DType: ``tc.dtype``.
        Tensor: ``tc.Tensor``.
        Vector: Column vector, first order Tensor ``.shape = (i,1)``.
        CoVector = Tensor: Row vector, first order Tensor ``.shape = (1,j)``.
        Matrix = Tensor: Second order Tensor ``.shape = (i,j)``.
        BatchVector = Tensor: Vector ``.shape = (...,i,1)``.
        BatchCoVector = Tensor: CoVector ``.shape = (...,1,j)``.
        BatchMatrix = Tensor: Matrix ``.shape = (...,i,j)``.
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
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')
