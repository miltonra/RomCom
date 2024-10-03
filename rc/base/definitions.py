#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2024 Robert A. Milton. All rights reserved.
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

""" Type and constant definitions.

All modules of RomCom ``import *`` from this module, so all types and constants in this module are referenced 
without adornment throughout RomCom."""


from typing import *
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch as tc
# import tensorflow as tf
# import gpflow as gf
# import rc.gpf as mf


#: Admissible logging verbosity levels.
LOGGING_LEVEL: dict[str,str] = {'NOTHING LOGGED': '3', 'ERROR': '2', 'ERROR+WARN': '1', 'ERROR+WARN+INFO': '0'}


#: ``LOGGING_LEVEL`` for TensorFlow.
TF_CPP_MIN_LOG_LEVEL: str = LOGGING_LEVEL['ERROR']


from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = str(TF_CPP_MIN_LOG_LEVEL)


Options = dict[str, Any] #: Type for passing options as ``**kwargs``.


Zero: float = 1.0E-64  #: Tolerance when testing floats for equality.


# def Int() -> Type:
#     """ The ``dtype`` of ``int`` in :ref:`rc.run.context.Environment`. """
#     return gf.config.default_int()
#
#
# def Float() -> Type:
#     """ The ``dtype`` of ``float`` in :ref:`rc.run.context.Environment`. """
#     return gf.config.default_float()


class PD:
    """ Extended Pandas types and constants.

    Attributes:
        DataFrame: pd.DataFrame.
        Index: pd.Index.
        MultiIndex: pd.MultiIndex.
    """
    DataFrame = pd.DataFrame    #: :meta private:
    Index = pd.Index    #: :meta private:
    MultiIndex = pd.MultiIndex  #: :meta private:

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
        Vector: Column vector, first order Tensor ``.shape = (i,1)``.
        CoVector = Tensor: Row vector, first order Tensor ``.shape = (1,j)``.
        Matrix = Tensor: Second order Tensor ``.shape = (i,j)``.
    """
    DType = np.dtype    #: :meta private:
    Array = np.ndarray  #: :meta private:
    Tensor = Array      #: :meta private:
    Vector = Tensor    #: :meta private:
    CoVector = Tensor  #: :meta private:
    Matrix = Tensor    #: :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


class TC:
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
        Slice = Tensor: A pair of ``int`` s used for slicing a Tensor rank.
        NaN: ``tc.constant(np.NaN, dtype=Float())`` representing Not a Number.
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
    Slice = list[int, int]  #: :meta private:
    NaN: Tensor = tc.nan  #: :meta private:
    Zero: Tensor = tc.tensor(Zero, dtype = Float)  #: :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


# class TF:
#     """ Extended TensorFlow types and constants. This class should never be instantiated or subclassed.
#
#     Attributes:
#         DType: ``np.dtype``.
#         Tensor: ``tf.Tensor``.
#         Vector: Column vector, first order Tensor ``.shape = (i,1)``.
#         Covector = Tensor: Row vector, first order Tensor ``.shape = (1,j)``.
#         Matrix = Tensor: Second order Tensor ``.shape = (i,j)``.
#         Slice = Tensor: A pair of ``int`` s used for slicing a Tensor rank.
#         NaN: ``tf.constant(np.NaN, dtype=Float())`` representing Not a Number.
#     """
#     DType = np.dtype    #: :meta private:
#     Tensor = tf.Tensor  #: :meta private:
#     Vector = Tensor     #: :meta private:
#     Covector = Tensor   #: :meta private:
#     Matrix = Tensor     #: :meta private:
#     Slice = tf.Tensor  #: :meta private:
#     NaN: Tensor = tf.constant(np.NaN, dtype=Float())  #: :meta private:

    def __init__(self):
        """

        :meta private:
        """
        raise NotImplementedError('This class is not intended to be instantiated or subclassed.')


