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

""" Contains tests of the gpf api."""


# from rc.gpf import base, kernels, likelihoods, models
import numpy as np
import gpflow as gf
from gpflow.ci_utils import reduce_in_tests
# import tensorflow as tf
from rc.user import contexts

use_gpu = True

if __name__ == '__main__':
    with contexts.Environment('Test', device='/GPU' if use_gpu else '/CPU'):
        X1 = np.random.rand(100, 1)  # Observed locations for first output
        X2 = np.random.rand(50, 1) * 0.5  # Observed locations for second output

        Y1 = np.sin(6 * X1) + np.random.randn(*X1.shape) * 0.03
        Y2 = np.sin(6 * X2 + 0.7) + np.random.randn(*X2.shape) * 0.1

        # Augment the input with ones or zeros to indicate the required output dimension
        X_augmented = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))

        # Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
        Y_augmented = np.vstack((np.hstack((Y1, np.zeros_like(Y1))), np.hstack((Y2, np.ones_like(Y2)))))

        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W

        # Base kernel
        k = gf.kernels.RBF(active_dims = [0])

        # Coregion kernel
        coreg = gf.kernels.Coregion(output_dim = output_dim, rank = rank, active_dims = [1])

        kern = k * coreg

        # This likelihood switches between Gaussian noise with different variances for each f_i:
        lik = gf.likelihoods.SwitchedLikelihood(
            [gf.likelihoods.Gaussian(), gf.likelihoods.Gaussian()]
        )

        gf.utilities.print_summary(k)
        gf.utilities.print_summary(coreg)
        gf.utilities.print_summary(kern)
        # now build the GP model as normal
        m = gf.models.VGP((X_augmented, Y_augmented), kernel = kern, likelihood = lik)

        # fit the covariance function parameters
        gf.optimizers.Scipy().minimize(
            m.training_loss,
            m.trainable_variables,
            options = dict(maxiter = 1E4),
            method = "L-BFGS-B",)

        B = coreg.output_covariance().numpy()
        print("B =", B)
