'''
Modules.

Summary
-------
A simple gpytorch module for exact inference is implemented.
It merely equips the corresponding base class with two modules
computing the mean value and covariance matrix of the GP.

'''

import gpytorch


class ExactInferenceGP(gpytorch.models.ExactGP):
    '''GP with zero mean and squared exp. covariance.'''

    def __init__(self,
                 x_train=None,
                 y_train=None,
                 prior_length=None,
                 prior_var=None,
                 noise_var=None):

        # initialize likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # call base class init
        super().__init__(x_train, y_train, likelihood)

        # construct mean module
        self.mean_module = gpytorch.means.ZeroMean()

        # construct covariance module
        self.cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        # set initial hyperparam values
        if prior_length is not None:
            self.cov_module.base_kernel.lengthscale = prior_length

        if prior_var is not None:
            self.cov_module.outputscale = prior_var

        if noise_var is not None:
            self.likelihood.noise = noise_var

    @property
    def prior_length(self):
        return self.cov_module.base_kernel.lengthscale.detach()

    @property
    def prior_var(self):
        return self.cov_module.outputscale.detach()

    @property
    def noise_var(self):
        return self.likelihood.noise.detach()

    def forward(self, x):
        '''Return GP prior distribution.'''
        mean = self.mean_module(x)
        cov = self.cov_module(x)

        mvn = gpytorch.distributions.MultivariateNormal(mean, cov)
        return mvn

