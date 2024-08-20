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


""" Benchmark GPR and GSA for known test functions. """


from __future__ import annotations
import argparse
import tarfile
import os

from rc.base.definitions import *
from rc import user
from rc.base.models import ToyDataBase, ToyModel

#: Parameters to generate data from test functions.
K: int = 2  #: The number of Folds in a new repository.
Ms: Tuple[int, ...] = (7, )  #: The number of inputs.
Ns: Tuple[int, ...] = (300, )   #: The number of samples (datapoints).
DOE: user.sample.DOE.Method = user.sample.DOE.latin_hypercube   #: The Design Of Experiments to generate the sample inputs.
FUNCTION_VECTOR: user.functions.Vector = user.functions.OAKLEY2004  #: The function vector to apply to the inputs generated by the DOE.
NOISE_MAGNITUDES: Tuple[float] = (0.04, )    #: The noise-to-signal ratio, which is equal to the StdDev of the noise added to the normalised function output.
IS_NOISE_COVARIANT: bool = False  #: Whether the Gaussian noise applied to the outputs is statistically independent between outputs.
IS_NOISE_VARIANCE_DETERMINED: bool = True  #: Whether the noise variance is fixed or random.
ROTATIONS = {'': None}  #: Dict of rotations applied to the input basis after the function vector has been sampled.
#: Parameters to run Gaussian Process Regression.
IS_GPR_READ: bool | None = False  #: Whether to read the GPR model from file.
IS_GPR_COVARIANT: bool | None = False  #: Whether the GPR likelihood is covariant.
IS_GPR_ISOTROPIC: bool | None = None  #: Whether the GPR kernel is isotropic.
#: Parameters to run Global Sensitivity Analysis.
GSA_KINDS: List[user.run.GSA.Kind] = user.run.GSA.ALL_KINDS  #: A list of the kinds of GSA to do.
IS_GSA_ERROR_CALCULATED: bool = True  #: Whether to calculate the GSA standard error
IS_GSA_ERROR_PARTIAL: bool = False  #: Whether the calculated the GSA standard error is partial


def run(root: str | Path) -> Path:
    """ Run benchmark data generation and/or Gaussian Process Regression and/or Global Sensitivity Analysis, and collect the results.

    Args:
        root: The root folder.
    Returns: The root path written to.
    """
    with user.contexts.Environment('Test', device='/CPU'):
        KIND_NAMES = [kind.name.lower() for kind in GSA_KINDS]
        for noise_magnitude in NOISE_MAGNITUDES:
            for M in Ms:
                for N in Ns:
                    noise_variance = user.sample.GaussianNoise.Variance(len(FUNCTION_VECTOR), noise_magnitude, IS_NOISE_COVARIANT, IS_NOISE_VARIANCE_DETERMINED)
                    for rotation_name, rotation in ROTATIONS.items():
                        with user.contexts.Timer(f'M={M}, N={N}, noise={noise_magnitude}', is_inline=False):
                            repo = user.sample.Function(root, DOE, FUNCTION_VECTOR, N, M, noise_variance, None,
                                                        True).repo.into_K_folds(K).rotate_folds(rotation)

                            # Run GPR, or collect stored GPR models.
                            # models = user.run.gpr(name='gpr', repo=repo, is_read=IS_GPR_READ, is_covariant=IS_GPR_COVARIANT,
                            #                       is_isotropic=IS_GPR_ISOTROPIC, ignore_exceptions=False)

                            # Collect GPR results from GPR models.
                            # user.results.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}},
                            #                      {repo.folder / model: {'model': model} for model in models},
                            #                      False).from_folders(repo.folder / 'gpr', True)
                            # user.results.Collect({'variance': {}, 'log_marginal': {}},
                            #                      {f'{repo.folder / model}/likelihood': {'model': model} for model in models},
                            #                      False).from_folders((repo.folder / 'gpr') / 'likelihood', True)
                            # user.results.Collect({'variance': {}, 'lengthscales': {}},
                            #                      {f'{repo.folder / model}/kernel': {'model': model} for model in models},
                            #                      False).from_folders((repo.folder / 'gpr') / 'kernel', True)

                            # Run GSA and collect results, or just collect results.
                            # user.run.gsa('gpr', repo, is_covariant=IS_GPR_COVARIANT, is_isotropic=False, kinds=GSA_KINDS,
                            #              is_error_calculated=IS_GSA_ERROR_CALCULATED, ignore_exceptions=False, is_T_partial=IS_GSA_ERROR_PARTIAL)
                            # user.results.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if IS_GSA_ERROR_CALCULATED else {}),
                            #                      {f'{repo.folder / model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                            #                       for kind_name in KIND_NAMES for model in models},
                            #                      True).from_folders((repo.folder / 'gsa'), True)
    return root


if __name__ == '__main__':
    # Run the code.
    root = Path('test')
    # print(f'Root path is {run(root)}')
    toydatabase = ToyDataBase(root)
    toymodel = ToyModel.create(root / 'model')
    toymodel(data=(1,2,3))
