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

from romcomma.base.definitions import *
from romcomma import user

#: Parameters to generate data from test functions.
K: int = -2  #: The number of Folds in a new repository.
Ms: Tuple[int, ...] = (7, 9, 11, 13, 15, 17, 19)  #: The number of inputs.
Ns: Tuple[int, ...] = (60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 520, 580, 640, 720, 800, 880, 960, 1050, 1150, 1260, 1380, 1510, 1650, 1800, 1960,
                       2130, 2210, 2400, 2600, 2820, 3060, 3320, 3600, 3900, 4220, 4560, 4920, 5420, 5860, 6340, 6860, 7420, 8000, 8600, 9200,
                       9840,)   #: The number of samples (datapoints).
DOE: user.sample.DOE.Method = user.sample.DOE.latin_hypercube  #: The Design Of Experiments to generate the sample inputs.
FUNCTION_VECTOR: user.functions.Vector = user.functions.ALL  #: The function vector to apply to the inputs generated by the DOE.
NOISE_MAGNITUDES: Tuple[float] = (0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0,
                                  5.0,)   #: The noise-to-signal ratio, which is equal to the StdDev of the noise added to the normalised function output.
IS_NOISE_VARIANCE_DETERMINED: bool = True  #: Whether the noise variance is fixed or random.
ROTATIONS: Dict[str, NP.Matrix] = {'': None}  #: Dict of rotations applied to the input basis after the function vector has been sampled.
#: Parameters to run Gaussian Process Regression.
IS_GPR_READ: bool | None = None  #: Whether to read the GPR model from file.
IS_GPR_ISOTROPIC: bool | None = False  #: Whether the GPR kernel is isotropic.
#: Parameters to run Global Sensitivity Analysis.
GSA_KINDS: List[user.run.GSA.Kind] = user.run.GSA.ALL_KINDS  #: A list of the kinds of GSA to do.
IS_GSA_ERROR_CALCULATED: bool = True  #: Whether to calculate the GSA standard error


def run(args: argparse.Namespace, root: str | Path) -> Path:
    """ Run benchmark data generation and/or Gaussian Process Regression and/or Global Sensitivity Analysis, and collect the results.

    Args:
        args: The command line arguments passed to this module.
        root: The root folder.
    Returns: The root path written to.
    """
    with user.contexts.Environment('Test', device='GPU' if args.GPU else 'CPU'):
        KIND_NAMES = [kind.name.lower() for kind in GSA_KINDS]
        gprs, gsas = {}, {}
        for noise_magnitude in NOISE_MAGNITUDES:
            for M in Ms:
                for N in Ns:
                    noise_variance = user.sample.GaussianNoise.Variance(len(FUNCTION_VECTOR), noise_magnitude, args.is_noise_covariant, IS_NOISE_VARIANCE_DETERMINED)
                    for rotation_name, rotation in ROTATIONS.items():
                        ext = rotation_name + f'.{args.ext}' if args.ext else ''
                        ext = ext if ext else None
                        with user.contexts.Timer(f'M={M}, N={N}, noise={noise_magnitude}, ext={ext}', is_inline=False):

                            # Get data sample, either from function or file.
                            if args.function:
                                repo = user.sample.Function(root, DOE, FUNCTION_VECTOR, N, M, noise_variance, ext,
                                                            True).repo.into_K_folds(K).rotate_folds(rotation)
                            else:
                                repo = user.sample.Function(root, DOE, FUNCTION_VECTOR, N, M, noise_variance, ext, False).repo

                            # Run GPR, or collect stored GPR models.
                            if args.gpr:
                                models = user.run.gpr(name='gpr', repo=repo, is_read=IS_GPR_READ, is_covariant=args.is_gpr_covariant,
                                                      is_isotropic=IS_GPR_ISOTROPIC, ignore_exceptions=args.ignore,
                                                      likelihood_variance=args.likelihood_variance)
                            else:
                                models = [path.name for path in repo.folder.glob('gpr.*')]

                            # Collect GPR results from GPR models.
                            user.results.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1]}},
                                                 {repo.folder / model: {'model': model} for model in models},
                                                 args.ignore).from_folders(repo.folder / 'gpr', True)
                            user.results.Collect({'variance': {}, 'log_marginal': {}},
                                                 {f'{repo.folder / model}/likelihood': {'model': model} for model in models},
                                                 args.ignore).from_folders((repo.folder / 'gpr') / 'likelihood', True)
                            user.results.Collect({'variance': {}, 'lengthscales': {}},
                                                 {f'{repo.folder / model}/kernel': {'model': model} for model in models},
                                                 args.ignore).from_folders((repo.folder / 'gpr') / 'kernel', True)
                            gprs |= {f'{repo.folder}/gpr': {'M': M, 'noise magnitude': noise_magnitude, 'IS_NOISE_COVARIANT': args.is_noise_covariant,
                                                            'IS_NOISE_VARIANCE_DETERMINED': IS_NOISE_VARIANCE_DETERMINED, 'ext': ext}}

                            # Run GSA and collect results, or just collect results.
                            if args.gsa:
                                user.run.gsa('gpr', repo, is_covariant=args.is_gpr_covariant, is_isotropic=False, kinds=GSA_KINDS,
                                             is_error_calculated=IS_GSA_ERROR_CALCULATED, ignore_exceptions=args.ignore,
                                             is_T_partial=args.is_T_partial)
                                user.results.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if IS_GSA_ERROR_CALCULATED else {}),
                                                     {f'{repo.folder / model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                                                      for kind_name in KIND_NAMES for model in models},
                                                     args.ignore).from_folders((repo.folder / 'gsa'), True)
                            else:
                                user.results.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if IS_GSA_ERROR_CALCULATED else {}),
                                                     {f'{repo.folder / model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                                                      for kind_name in KIND_NAMES for model in models},
                                                     True).from_folders((repo.folder / 'gsa'), True)
                            gsas |= {f'{repo.folder}/gsa': {'M': M, 'noise magnitude': noise_magnitude, 'IS_NOISE_COVARIANT': args.is_noise_covariant,
                                                            'IS_NOISE_VARIANCE_DETERMINED': IS_NOISE_VARIANCE_DETERMINED, 'ext': ext}}
    user.results.Collect({'test_summary': {'header': [0, 1]}}, gprs, True).from_folders(root / 'gpr', True)
    user.results.Collect({'variance': {}, 'log_marginal': {}}, gprs, True).from_folders((root / 'gpr') / 'likelihood', True)
    user.results.Collect({'variance': {}, 'lengthscales': {}}, gprs, True).from_folders((root / 'gpr') / 'kernel', True)
    user.results.Collect({'S': {}, 'V': {}, 'T': {}, 'W': {}}, gsas, True).from_folders((root / 'gsa'), True)
    if args.copy:
        dst = Path(args.copy)
        user.results.copy(root / 'gpr', dst / 'gpr')
        user.results.copy(root / 'gsa', dst / 'gsa')
    return root


if __name__ == '__main__':

    # Get the command line arguments.
    parser = argparse.ArgumentParser(description='A program to benchmark GPR and GSA against a (vector) test function.')
    # Control flow arguments.
    parser.add_argument('-f', '--function', action='store_true', help='Flag to sample the test function to generate test data.')
    parser.add_argument('-r', '--gpr', action='store_true', help='Flag to run Gaussian process regression.')
    parser.add_argument('-s', '--gsa', action='store_true', help='Flag to run global sensitivity analysis.')
    parser.add_argument('-i', '--ignore', action='store_true', help='Flag to ignore exceptions.')
    parser.add_argument('-G', '--GPU', action='store_true', help='Flag to run on a GPU instead of CPU.')
    # Optional parameter setters
    parser.add_argument('-K', '--folds', help='The number of k-folds to use (negative to omit improper fold). Defaults to 2.', type=int)
    parser.add_argument('-M', '--input_dim', help='The input dimension M. Defaults to [7, 10, 12, 15, 18].', type=int)
    parser.add_argument('-c', '--is_noise_covariant', action='store_true', help='Whether noise (uncertainty) is covariant across outputs.')
    parser.add_argument('-C', '--is_gpr_covariant', action='store_true', help='Whether GPR (likelihood) is covariant across outputs.')
    parser.add_argument("-l", "--likelihood_variance", help="Initial guess for likelihood variance to be calibrated.", type=float)
    parser.add_argument('-p', '--is_T_partial', action='store_true', help='Whether GSA error T is partial.')
    # File locations
    parser.add_argument('-e', '--ext', help='The extension appended to each Store name.', type=str)
    parser.add_argument('-t', '--tar', help='Outputs a .tar.gz file to path.', type=str)
    parser.add_argument('-y', '--copy', help='Copies collectd results to path.', type=str)
    parser.add_argument('root', help='The path of the root folder to house all data repositories.', type=str)
    args = parser.parse_args()  # Convert arguments to argparse.Namespace.
    # Run the code.
    K = args.folds if args.folds else K
    Ms = (args.input_dim,) if args.input_dim else Ms
    root = Path(args.root)
    print(f'Root path is {run(args, root)}')
    # Tar outputs
    if args.tar:
        tar_path = Path(args.tar)
        tar_path.parents[0].mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, 'w:gz') as tar:
            for item in os.listdir(args.root):
                tar.add(Path(args.root, item), arcname=item)
