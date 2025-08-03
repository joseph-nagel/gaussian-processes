'''
Kernel functions.

Summary
-------
Some elementary kernel functions are provided in this module.
They allow for constructing covariance matrices.
Note that the implementations serve didactic purposes only.

'''

from abc import ABCMeta, abstractmethod

import torch


class IsotropicKernel(metaclass=ABCMeta):
    '''Isotropic kernel base class.'''

    def distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        p: int = 2
    ) -> torch.Tensor:
        '''Compute pairwise distances.'''

        x1 = torch.as_tensor(x1)

        dtype = x1.dtype
        device = x1.device

        if x2 is None:
            x2 = torch.tensor(0.0, dtype=dtype, device=device)
        else:
            x2 = torch.as_tensor(x2)

        if x1.ndim <= 1:
            x1 = x1.view(-1, 1)

        if x2.ndim <= 1:
            x2 = x2.view(-1, 1)

        dist = torch.cdist(x1, x2, p=p).squeeze()

        return dist

    @abstractmethod
    def kernel(self, dist: torch.Tensor) -> torch.Tensor:
        '''Evaluate isotropic kernel.'''
        raise NotImplementedError

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''Compute covariance matrix.'''

        dist = self.distance(x1, x2)
        cov = self.kernel(dist)

        return cov


class SquaredExponential(IsotropicKernel):
    '''Squared exponential cov. function.'''

    def __init__(
        self,
        sigma: float = 1.0,
        length: float = 1.0
    ) -> None:

        self.sigma = abs(sigma)
        self.length = abs(length)

    def kernel(self, dist: torch.Tensor) -> torch.Tensor:
        '''Evaluate squared exp. kernel.'''
        return self.sigma**2 * torch.exp(-0.5 * dist**2 / self.length**2)


class AbsoluteExponential(IsotropicKernel):
    '''Absolute exponential cov. function.'''

    def __init__(
        self,
        sigma: float = 1.0,
        length: float = 1.0
    ) -> None:

        self.sigma = abs(sigma)
        self.length = abs(length)

    def kernel(self, dist: torch.Tensor) -> torch.Tensor:
        '''Evaluate absolute exp. kernel.'''
        return self.sigma**2 * torch.exp(-torch.abs(dist) / self.length)

