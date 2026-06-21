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

""" Test functions, taken from `SALib test functions <https://salib.readthedocs.io/en/latest/api/SALib.test_functions.html>`_."""

from rc.base import *

import SALib.test_functions.Ishigami as SALibIshigami
import SALib.test_functions.Sobol_G as SALibSobolG
import SALib.test_functions.oakley2004 as SALibOakley2004


class Scalar:
    """A scalar function ``scalar`` such that ``scalar(x, params)`` calls
        ``self.salib(self.loc + self.scale * x[:, :self.m], **(self.params | params)``."""

    @property
    def salib(self) -> Callable[Np.Matrix, float]:
        return self._salib

    @property
    def loc(self) -> Np.Vector:
        return self._loc

    @property
    def scale(self) -> Np.Vector:
        return self._scale

    @property
    def m(self) -> int:
        return self._m

    @property
    def params(self) -> dict[str, Np.Array]:
        return self._params

    def __call__(self, x: Np.Matrix, **params: Np.Matrix) -> Np.Matrix:
        return np.reshape(self._salib(self._loc + self._scale * x[:, :self._m], **(self._params | params)),
                          (x.shape[0], 1))

    def __init__(self, salib: Callable[Np.Matrix, float], loc: Np.Vector, scale: Np.Vector, m: int,
                 **params: float | Np.Array | Sequence[float | Np.Array]):
        """ A scalar function, which calls ``call(loc + scale * x[:, :m], **params)``.

        Args:
            salib: The SALib function called.
            loc: Input offset.
            scale: Input scale.
            m: The number of input dimensions.
            **params: Function data applied to call.
        """
        self._salib = salib
        self._loc = loc
        self._scale = scale
        self._m = m
        self._params = params


class Vector(dict):
    """ A vector functon, which is little more than a named dictionary of Scalar functions,
        such that ``vector(x, **params)`` concatenates ``scalar(x, **params)``
        for each dictionary item ``key: Scalar``. """

    @classmethod
    def concat(cls, name: str, vectors: Sequence[Vector]) -> Vector:
        """ Concatenate vectors.

        Args:
            name: The name of the returned ``Vector``.
            vectors: A sequence of ``Vector`` functions to concatenate.

        Returns: The concatenation of ``vectors``, named ``name``.
        """
        result = cls(name)
        for vector in vectors:
            result.update({f'{vector.name}.{key}': scalar for key, scalar in vector.items()})
        return result

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> dict:
        """ Meta data for providing to ``data.storage``."""
        return {'name': self.name, 'salib': {l: function for l, function in enumerate(self.keys())}}

    def subVector(self, name: str, scalars: Sequence[str]) -> Vector:
        """ Create a subVector of ``self``.

        Args:
            name: The name of the ``subVector``.
            scalars: The keys of the items of ``self`` to be included in subVector.
        Returns: A new instance of ``Vector`` named ``name`` containing the ``Scalars`` keyed ``scalars``.
            Effectively the pseudo-slice ``self[scalars]``.
        """
        return Vector(name, **{scalar: self[scalar] for scalar in scalars})

    def __call__(self, x: Np.Matrix, **params) -> Np.Matrix:
        return np.concatenate([scalar(x, **params) for scalar in self.values()], axis = 1)

    def __init__(self, name: str, **scalars: Scalar):
        """ Construct a vector function.

        Args:
            name: The name of this Vector.
            **scalars: The dict of Scalars comprising this Vector.
        """
        super().__init__(**scalars)
        self._name = name


""" The ishigami function without data. """
_ishigami = {'salib': SALibIshigami.evaluate, 'loc': -np.pi, 'scale': 2 * np.pi}


""" Modified Sobol G-function without data."""
_sobolG = {'salib': SALibSobolG.evaluate, 'loc': 0, 'scale': 1}


""" Modified oakley & O'Hagan (2004) function without data."""
_oakley2004 = {'salib': SALibOakley2004.evaluate, 'loc': -1, 'scale': 2}


def linspace(start: float, stop: float, shape: Sequence[int]) -> Np.Matrix:
    """ A multi-dimensional version of ``np.linspace``, distributing values throughout ``shape``.

    Args:
        start: Start value, which will be returned in ``linspace(...)[0,...,0]``.
        stop: Stop value, which will be returned in ``linspace(...)[-1,...,-1]``.
        shape: The ``linspace.shape`` to return.
    Returns: ``np.reshape(np.linspace(start, stop, int(np.prod(shape)), endpoint=True), shape)``.
    """
    return np.reshape(np.linspace(start, stop, int(np.prod(shape)), endpoint = True), shape)


ishigami = Vector(name = 'ishigami',
                  standard = Scalar(**_ishigami, m = 3, A = 7.0, B = 0.1),
                  balanced = Scalar(**_ishigami, m = 3, A = 20.0, B = 1.0),
                  sin = Scalar(**_ishigami, m = 3, A = 0.0, B = 0.0),
                  )
""" Three example ishigami functions, taking (at least) 3 continuous inputs."""


sobolG: Vector = Vector(name = 'sobolG',
                 weak5_2 = Scalar(**_sobolG, m = 5, a = np.array([3, 6, 9, 18, 27]),
                                  alpha = np.ones((5,)) * 2.0),
                 strong5_2 = Scalar(**_sobolG, m = 5, a = np.array([1 / 2, 1, 2, 4, 8]),
                                    alpha = np.ones((5,)) * 2.0),
                 strong5_4 = Scalar(**_sobolG, m = 5, a = np.array([1 / 2, 1, 2, 4, 8]),
                                    alpha = np.ones((5,)) * 4.0),
                 )
""" Three example modified Sobol G-functions, taking (at least) 5 continuous inputs."""


oakley2004_5: Vector = Vector(name = 'oakley2004',
                      lin = Scalar(**_oakley2004, m = 5,
                                   A = [linspace(start = 5.0, stop = 5.0 / 2, shape = [5, ]), ] + [
                                       np.zeros([5])] * 2,
                                   M = np.zeros([5, 5])),
                      quad = Scalar(**_oakley2004, m = 5,
                                    A = [linspace(start = 5.0, stop = 5.0 / 2, shape = [5, ]), ] + [
                                        np.zeros([5])] * 2,
                                    M = linspace(start = 5.0, stop = 1.0, shape = [5, 5])),
                      rev = Scalar(**_oakley2004, m = 5,
                                   A = [-linspace(start = 5.0, stop = 5.0 / 2, shape = [5, ]), ] + [
                                       np.zeros([5])] * 2,
                                   M = linspace(start = 1.0, stop = 5.0, shape = [5, 5])),
                      )
""" Three example modified oakley & O'Hagan (2004) functions, taking (at least) 5 continuous inputs."""


oakley2004: Vector = Vector(name = 'oakley2004',
                    lin = Scalar(**_oakley2004, m = 7,
                                 A = [linspace(start = 7.0, stop = 7.0 / 2, shape = [7, ]), ] + [
                                     np.zeros([7])] * 2,
                                 M = np.zeros([7, 7])),
                    quad = Scalar(**_oakley2004, m = 7,
                                  A = [linspace(start = 7.0, stop = 7.0 / 2, shape = [7, ]), ] + [
                                      np.zeros([7])] * 2,
                                  M = linspace(start = 7.0, stop = 1.0, shape = [7, 7])),
                    rev = Scalar(**_oakley2004, m = 7,
                                 A = [-linspace(start = 7.0, stop = 7.0 / 2, shape = [7, ]), ] + [
                                     np.zeros([7])] * 2,
                                 M = linspace(start = 1.0, stop = 7.0, shape = [7, 7])),
                    )
""" Three example modified oakley & O'Hagan (2004) functions, taking (at least) 7 continuous inputs."""


combo: Vector = Vector.concat(name = 'combo', vectors = (ishigami, sobolG, oakley2004))
"""The concatenation of ``ishigami, sobolG, oakley2004``. """


"""The concatenation of ishigami, sobolG, oakley2004. """
_categorized: dict[str, Vector] = {'ish': ishigami, 'sob': sobolG, 'oak': oakley2004}


def categorized(x, f: str, p: int) -> float:
    """ A single-output test function, categorized by ``f`` and ``p``.

    Args:
        x: ``(n, m)`` design matrix of continuous inputs.
        f: The function category: 0 for ``ishigami``, 1 for ``sobolG``, 2 for ``oakley2004``.
        p: The parameter category: 0 for ``standard/weak5_2/lin``, 1 for ``balanced/strong5_2/quad``,
            2 for ``sin/strong5_4/rev``.

    Returns: The categorized function value evaluated at ``x,f,p``.
    """
    return _categorized[f].values[p](x)
