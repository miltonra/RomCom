#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2024 Robert A. Milton. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
# 
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Type and constant definitions.

All modules of RomCom ``import *`` from this module, so all types and constants in this module are referenced without adornment throughout RomCom."""

LOGGING_LEVEL: dict[str,str] = {'NOTHING LOGGED': '3', 'ERROR': '2', 'ERROR+WARN': '1', 'ERROR+WARN+INFO': '0'}    #: Admissible logging verbosity levels.

TF_CPP_MIN_LOG_LEVEL: str = LOGGING_LEVEL['ERROR']   #: ``LOGGING_LEVEL`` for TensorFlow.

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = str(TF_CPP_MIN_LOG_LEVEL)

from typing import *
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import gpflow as gf
import rc.gpf as mf
import pandas as pd

Options = dict[str, Any] #: Type for passing options as ``**kwargs``.

ZERO: float = 1.0E-64  #: Tolerance when testing floats for equality.


def Int() -> Type:
    """ The ``dtype`` of ``int`` in :ref:`rc.run.context.Environment`. """
    return gf.config.default_int()


def Float() -> Type:
    """ The ``dtype`` of ``float`` in :ref:`rc.run.context.Environment`. """
    return gf.config.default_float()


class PD:
    """ Extended Pandas types and constants.

    Attributes:
        DataFrame: pd.DataFrame.
        Index: pd.Index.
        MultiIndex: pd.MultiIndex.
    """
    DataFrame = pd.DataFrame    #: :meta private:
    Index = pd.Index    #: :meta private:
    MultiIndex = pd.MultiIndex  #:  :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


class NP:
    """ Extended TensorFlow types and constants. This class should never be instantiated or subclassed.

    Attributes:
        DType: ``np.dtype``.
        Array: ``np.ndarray``.
        Tensor: ``Array``.
        Vector: Column vector, first order ``Tensor`` shaped (i,1).
        Covector = Tensor: Row vector, first order ``Tensor`` shaped (1,j).
        Matrix = Tensor: Second order ``Tensor`` shaped (i,j).
        VectorLike: ``int | float | Sequence[int | float] | Array``.
        MatrixLike: ``VectorLike | Sequence[VectorLike]``.
        ArrayLike: ``MatrixLike | Sequence[MatrixLike] | Sequence[Sequence[MatrixLike]]``.
        TensorLike: ``ArrayLike``.
    """
    DType = np.dtype    #:  :meta private:
    Array = np.ndarray  #:  :meta private:
    Tensor = Array      #:  :meta private:
    Vector = Tensor    #:  :meta private:
    Covector = Tensor  #:  :meta private:
    Matrix = Tensor    #:  :meta private:
    VectorLike = int | float | Sequence[int | float] | Array   #:  :meta private:
    MatrixLike = VectorLike | Sequence[VectorLike]   #:  :meta private:
    ArrayLike = TensorLike = MatrixLike | Sequence[MatrixLike] | Sequence[Sequence[MatrixLike]]   #:  :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


class TF:
    """ Extended TensorFlow types and constants. This class should never be instantiated or subclassed.

    Attributes:
        DType: ``np.dtype``.
        Tensor: ``tf.Tensor``.
        Vector: Column vector, first order ``Tensor`` shaped (i,1).
        Covector = Tensor: Row vector, first order ``Tensor`` shaped (1,j).
        Matrix = Tensor: Second order ``Tensor`` shaped (i,j).
        VectorLike: ``int | float | Sequence[int | float] | Array``.
        MatrixLike: ``VectorLike | Sequence[VectorLike]``.
        TensorLike: ``MatrixLike | Sequence[MatrixLike] | Sequence[Sequence[MatrixLike]]``.
        Slice = Tensor: A pair of ``int`` s used for slicing a Tensor rank.
        NaN: ``tf.constant(np.NaN, dtype=Float())`` representing Not a Number.
    """
    DType = np.dtype    #:  :meta private:
    Tensor = tf.Tensor  #:  :meta private:
    Vector = Tensor     #:  :meta private:
    Covector = Tensor   #:  :meta private:
    Matrix = Tensor     #:  :meta private:
    VectorLike = int | float | Sequence[int | float] | Tensor  #:  :meta private:
    MatrixLike = VectorLike | Sequence[VectorLike]   #:  :meta private:
    TensorLike = MatrixLike | Sequence[MatrixLike] | Sequence[Sequence[MatrixLike]]  #:  :meta private:
    Slice = tf.Tensor  #:  :meta private:
    NaN: Tensor = tf.constant(np.NaN, dtype=Float())  #:  :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')
