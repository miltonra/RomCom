#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2021 Robert A. Milton. All rights reserved.
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

""" Run this module first thing, to test_data your installation of romcomma. """

from __future__ import annotations

from romcomma.typing_ import *
from romcomma import run
from romcomma.data import Fold, Store, Frame
from romcomma.test import functions
import numpy as np
from pathlib import Path
import shutil
import scipy.stats

BASE_PATH = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\2.2')


def fold_and_rotate_with_tests(store: Store, K: int, rotation: NP.Matrix):
    store._data.df = store._data.df * 5
    store._data.write()
    fold_and_rotate(store, K, rotation)
    shutil.copytree(store.fold_folder(store.K), store.folder / f'fold.{store.K + 1}')
    fold = Fold(store, store.K + 1)
    fold.X_rotation = np.transpose(rotation)
    Frame(fold._test_csv, fold.normalization.undo_from(fold._test_data.df))
    fold = Fold(store, store.K)
    Frame(store.folder / 'undone.csv', fold.normalization.undo_from(fold.test_data.df))


def fold_and_rotate(store: Store, K: int, rotation: NP.Matrix):
    store.into_K_folds(K)
    for k in range(store.K + 1):
        fold = Fold(store, k)
        fold.X_rotation = rotation


# noinspection PyShadowingNames
def run_gps(name, function_names: Sequence[str], N: int, noise_variance: [float], random: bool, M: int = 5, K: int = 2):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    store_folder = '.'.join(function_names) + f'.{M:d}.{sum(noise_variance)/len(noise_variance):.3f}.{N:d}'
    if random:
        rotation = scipy.stats.ortho_group.rvs(M)
        store_folder += '.random'
    else:
        rotation = np.eye(M)
        store_folder += '.rom'
    store_folder = BASE_PATH / store_folder
    store = functions.sample(f, N, M, noise_variance, store_folder)
    # fold_and_rotate_with_tests(store, K, rotation)
    fold_and_rotate(store, K, rotation)
    run.gps(name=name, store=store, is_read=False, is_isotropic=False, is_independent=True, kernel_parameters=None, parameters=None,
            optimize=True, test=True)


# noinspection PyShadowingNames
def compare_gps(name, function_names: Sequence[str], N: int, noise_variance: [float], random: bool, M: int = 5):
    if isinstance(function_names, str):
        function_names = [function_names]
    store_folder = '.'.join(function_names) + f'.{M:d}.{sum(noise_variance)/len(noise_variance):.3f}.{N:d}'
    if random:
        store_folder += '.random'
    else:
        store_folder += '.rom'
    store_folder = BASE_PATH / store_folder
    store = Store(store_folder)
    run.gps(name=name, store=store, is_read=None, is_isotropic=False, is_independent=False, kernel_parameters=None, parameters=None,
            optimize=True, test=True)


def noise_variance(L: int, scale: float, diagonal: bool = False, random: bool = False):
    if diagonal:
        result = np.eye(L)
    elif random:
        result = np.random.random_sample((2, 2))
        result = np.matmul(result, result.transpose())
    else:
        result = np.ones(shape=(L,L))/2 + np.eye(L)/2
    return scale * result


if __name__ == '__main__':
    with run.Context('Test'):
        for N in (800,):
            for noise_variance in (0.001,):
                for random in (False, True):
                    for M in (5,):
                        run_gps('initial', ['sin.1', 'sin.1'], N, [noise_variance] * 2, random, M)
                        compare_gps('initial', ['sin.1', 'sin.1'], N, [noise_variance] * 2, random, M)
