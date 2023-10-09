'''
Kernel functions.

Summary
-------
Some elementary kernel functions are implemented.
They allow for constructing covariance matrices.
Note that the kernels are simply provided for convenience,
but they do not establish custom kernels for gpytorch.

'''

import torch


class IsotropicKernel():
    '''Isotropic kernel base class.'''

    def distance(self, x1, x2=None, p=2):
        '''Compute pairwise distances.'''

        x1 = torch.as_tensor(x1)

        dtype = x1.dtype
        device = x1.device

        if x2 is None:
            x2 = torch.tensor(0., dtype=dtype, device=device)
        else:
            x2 = torch.as_tensor(x2)

        if x1.ndim <= 1:
            x1 = x1.view(-1, 1)

        if x2.ndim <= 1:
            x2 = x2.view(-1, 1)

        dist = torch.cdist(x1, x2, p=p).squeeze()

        return dist


class SquaredExponential(IsotropicKernel):
    '''Squared exponential cov. function.'''

    def __init__(self, sigma=1, tau=1):
        self.sigma = abs(sigma)
        self.tau = abs(tau)

    def __call__(self, x1, x2=None):
        '''Compute covariances.'''
        dist = self.distance(x1, x2, p=2)
        cov = self.sigma**2 * torch.exp(-0.5 * dist**2 / self.tau**2)
        return cov


class AbsoluteExponential(IsotropicKernel):
    '''Absolute exponential cov. function.'''

    def __init__(self, sigma=1, tau=1):
        self.sigma = abs(sigma)
        self.tau = abs(tau)

    def __call__(self, x1, x2=None):
        '''Compute covariances.'''
        dist = self.distance(x1, x2, p=1)
        cov = self.sigma**2 * torch.exp(-dist / self.tau)
        return cov

