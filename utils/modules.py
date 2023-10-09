'''
Modules.

Summary
-------
A simple gpytorch module for exact inference is implemented.
It merely equips the correspoonding base class with two modules
computing the mean value and covariance matrix of the GP.

'''

import gpytorch


class ExactInferenceGP(gpytorch.models.ExactGP):
    '''GP with zero/const. mean and squared exp. covariance.'''

    def __init__(self,
                 x_train,
                 y_train,
                 likelihood,
                 mean='zero'):

        # call base class init
        super().__init__(x_train, y_train, likelihood) # Why is the training data needed for initialization?

        # construct mean module
        if mean == 'zero':
            self.mean_module = gpytorch.means.ZeroMean()
        elif mean == 'const':
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            raise ValueError('Unknown mean mode: {}'.format(mean))

        # construct covariance module
        self.cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        '''Return GP prior distribution.'''
        mean = self.mean_module(x)
        cov = self.cov_module(x)

        mvn = gpytorch.distributions.MultivariateNormal(mean, cov)
        return mvn

