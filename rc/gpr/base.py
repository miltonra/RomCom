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

""" Basic facilities for GPR Models."""


from __future__ import annotations

from rc.base.models import *


class Variance(tc.nn.Module):
    """ A (Co)Variance Matrix, efficiently represented."""

    #: Class attribute holding the SDTransform.
    SDTransform: tc.distributions.Transform = tc.distributions.transforms.SoftplusTransform()

    #: Class attribute flooring the CorrTransform.
    CorrTransform: tc.distributions.Transform = tc.distributions.transforms.CorrCholeskyTransform()

    #: Instance attribute holding the lower bound the SD.
    SDfloor: float

    #: Instance attribute holding ``Variance.SDTransform.inv(SD)``.
    sd: TC.BatchVector
    
    #: Instance attribute holding ``Variance.CorrTransform.inv(tc.cholesky(correlation)``.
    corr: TC.BatchVector | None

    def forward(self, is_cho: bool = True) -> TC.BatchMatrix | TC.BatchVector:
        """ Aliases ``self.__call__()``. Do not call.

        Args:
            is_cho: False to return the variance, True to return its Cholesky lower triangle.

        Returns: ``tc.cholesky(variance) if is_cho else variance``.
            In either case the result is shaped (...,L,L), or (...,L,1) if diagonal (not covariant).
        """
        cho = self.SDTransform(self.sd) + self.SDfloor
        if self.corr is not None:
            cho = tc.einsum('...Ln, ...Ll -> ...Ll', cho, self.CorrTransform(self.corr))
        return cho if is_cho else tc.einsum('...Ln, ...ln -> ...Ll', cho, cho)

    def __init__(self, sd: TC.BatchVector, SDfloor: float = 0, corr: TC.BatchVector | None = None):
        """ Construct a diagonal (...,L,1) or square (...,L,L) Variance object.

        Args:
            sd: The Standard Deviation ``sd``, passed as ``Variance.SDTransform.inv(SD)``.
            SDfloor: The lower bound applicable to the Standard Deviation.
            corr: The Cholesky lower triangle of ``correlation``,
                passed as ``Variance.CorrTransform.inv(tc.cholesky(correlation))``.
        """
        super().__init__()
        self.sd = tc.nn.Parameter(sd, requires_grad = True)
        self.corr = None if corr is None else tc.nn.Parameter(corr, requires_grad = True)
        self.SDfloor = SDfloor

    @classmethod
    def create(cls, variance: TC.BatchMatrix | TC.BatchCoVector, SDfloor: float = 0) -> Self:
        """ Create a Variance object from a diagonal (...,L,1) CoVector or square (...,L,L) Matrix.

        Args:
            variance: The (co)variance matrix to be stored. If a BatchedCovector is supplied, this
                represents a  diagonal variance matrix of dimension ``variance.shape[-2]``.
            SDfloor: The lower bound applicable to the Standard Deviation.

        Returns: The Variance object efficiently representing ``variance``.
        """
        variance = tc.tensor(variance, TC.Float)
        corr = None
        if variance.shape[-1] == 1:
            sd = tc.sqrt(variance[..., :, 0])
        else:
            assert variance.shape[-2] == variance.shape[-1], (f'Variance must be shaped (...,L,1) '
                                                              f'or (...,L,L), not {variance.shape}.')
            sd = tc.sqrt(variance.diagonal(0, -2, -1))
            corr = tc.cholesky(variance / (sd.unsqueeze(-1) * sd.unsqueeze(-2)))
        return cls(cls.SDTransform.inv(tc.maximum(sd.unsqueeze(-1) - SDfloor, TC.Zero)), SDfloor,
                   cls.CorrTransform.inv(corr))
