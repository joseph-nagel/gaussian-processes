'''Gaussian process utilities.'''

from .kernels import (
    IsotropicKernel,
    SquaredExponential,
    AbsoluteExponential
)

from .modules import ExactInferenceGP

