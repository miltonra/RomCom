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

""" Type and constant definitions.

All modules of RomCom ``import *`` from this module, so all types and constants in this module are referenced 
without adornment throughout RomCom."""

from typing import *
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch as tc


zero: float = 1.0E-64
"""Tolerance when testing floats for equality."""


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
