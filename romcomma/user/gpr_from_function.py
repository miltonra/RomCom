#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2023 Robert A. Milton. All rights reserved.
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

""" Create a Repository by sampling a test function, adding noise, and optionally rotating the input basis """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.run import context, function, sample, summarised, results


ROOT: Path = Path('C:/Users/fc1ram/Documents/Research/rom-comma/dat/SoftwareTest/2.3')     #: The root folder to house all data repositories.
READ: bool = False    #: Whether to read an existing Repository, or create a new one overwriting it.
IGNORE_EXCEPTIONS: bool = False    #: Whether to ignore exceptions, normally due to failed GPR optimisation.
K: int = -5   #: The number of Folds in a new repository.
Ms: Tuple[int] = (6, )   #: The number of inputs.
Ns: Tuple[int] = (200, )   #: The number of samples (datapoints).
DOE: sample.DOE.Method = sample.DOE.latin_hypercube    #: The Design Of Experiments to generate the sample inputs.
FUNCTION_VECTOR: function.Vector = function.ISHIGAMI  #: The function vector to apply to the inputs generated by the DOE.
NOISE_MAGNITUDES: Tuple[float] = (0.1,)   #: The noise-to-signal ratio, which is equal to the StdDev of the noise added to the normalised function output.
IS_NOISE_COVARIANT: Tuple[bool] = (False,)   #: Whether the Gaussian noise applied to the outputs is statistically independent between outputs.
IS_NOISE_VARIANCE_RANDOM: Tuple[bool] = (False,)    #: Whether the noise variance is stochastic or fixed.
ROTATIONS = (None,)     #: Rotation applied to the input basis after the function vector has been sampled.
IS_READ: bool | None = None    #: Whether to read the GPR model from file.
IS_COVARIANT: bool | None = False    #: Whether the GPR likelihood is covariant.
IS_ISOTROPIC: bool | None = False    #: Whether the GPR kernel is isotropic.

if __name__ == '__main__':
    with context.Environment('Test', device='CPU'):
        for M in Ms:
            for N in Ns:
                for noise_magnitude in NOISE_MAGNITUDES:
                    for is_noise_covariant in IS_NOISE_COVARIANT:
                        for is_noise_variance_random in IS_NOISE_VARIANCE_RANDOM:
                            noise_variance = sample.GaussianNoise.Variance(len(FUNCTION_VECTOR), noise_magnitude, is_noise_covariant, is_noise_variance_random)
                            ext = 0
                            for rotation in ROTATIONS:
                                with context.Timer(f'M={M}, N={N}, noise={noise_magnitude}, is_noise_covariant={is_noise_covariant}, ' +
                                                   f'is_noise_variance_random={is_noise_variance_random}, ext={ext}', is_inline=False):
                                    if READ:
                                        repo = sample.Function(ROOT, DOE, FUNCTION_VECTOR, N, M, noise_variance, str(ext), False).repo
                                    else:
                                        repo = (sample.Function(ROOT, DOE, FUNCTION_VECTOR, N, M, noise_variance, str(ext), True)
                                                .into_K_folds(K).rotate_folds(rotation).repo)
                                models = summarised.gpr(name='gpr', repo=repo, is_read=IS_READ, is_covariant=IS_COVARIANT, is_isotropic=IS_ISOTROPIC,
                                                        ignore_exceptions=IGNORE_EXCEPTIONS)
                                results.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}},
                                            {repo.folder / model: {'model': model} for model in models}, IGNORE_EXCEPTIONS).from_folders(repo.folder / 'gpr',
                                                                                                                                         True)
                                results.Collect({'variance': {}, 'log_marginal': {}}, {f'{repo.folder / model}/likelihood': {'model': model} for model in models},
                                            IGNORE_EXCEPTIONS).from_folders((repo.folder / 'gpr') / 'likelihood', True)
                                results.Collect({'variance': {}, 'lengthscales': {}}, {f'{repo.folder / model}/kernel': {'model': model} for model in models},
                                            IGNORE_EXCEPTIONS).from_folders((repo.folder / 'gpr') / 'kernel', True)
                                ext += 1
