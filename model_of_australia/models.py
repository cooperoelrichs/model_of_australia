import os

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy import stats

import theano
import theano.tensor as tt
theano.config.gcc.cxxflags = "-fbracket-depth=1024" # default is 256

from model_of_australia.printing_tools import PrintingTools
from model_of_australia.data_loader import DataLoader

# ITERS = 1e4
# TUNE_ITERS = 2e3


class ModelSummariser():
    def traceplots(trace, output_dir):
        plt.figure(figsize=(7, 7))
        pm.traceplot(trace, combined=True)
        plt.tight_layout()
        DataLoader.maybe_make_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'traceplots.png'))

    def autocorrelation_plots(trace, burn=None):
        if burn is not None:
            burn = int(burn)
        pm.plots.autocorrplot(
            trace, max_lag=100, burn=burn,
            plot_transformed=False, symmetric_plot=False,
            ax=None, figsize=None
        )
        plt.show()

    def resid_plots(y, y_hat):
        e = y - y_hat
        plt.plot(e, 'kx')
        plt.show()

        plt.plot(y_hat, y, 'kx')
        plt.show()

    def summary(trace):
        print(pm.summary(trace))

    def mini_summary(trace, names):
        for dims, a, _ in names:
            if dims==0:
                ts = [(a, trace[a])]
            elif dims==1:
                T = trace[a]
                T_ndim = T.ndim
                if T_ndim==2:
                    ts = [
                        ('%s %i' % (a, i), T[:, i])
                        for i in range(T.shape[1])
                    ]
                elif T_ndim==3:
                    ts = [
                        ('%s %i' % (a, i), T[:, :, i])
                        for i in range(T.shape[2])
                    ]
                else:
                    raise ValueError(
                        'trace ndim not suported: ndim=%i, shape=%s' %
                        (T_ndim, str(T.shape))
                    )
            else:
                raise ValueError('too many dimensions: %s' % str(dims))

            for a, t in ts:
                print(
                    '%s: mean = %.4f, sd = %.6f, shape=%s' %
                    (a, t.mean(), t.std(), str(t.shape))
                )

    def gelman_rubin(trace, names, mean_only):
        print('gelman_rubin:')
        gr = pm.gelman_rubin(trace)
        for _, a, _ in names:
            if mean_only:
                print(' - %s: %.3f' % (a, gr[a].mean()))
            else:
                print(' - %s: %s' % (a, str(gr[a])))

    def dic(model, trace):
        print('dic = %.3f' % (pm.stats.dic(model=model, trace=trace)))

    def summarise(
        r, names, output_dir, burn=None, dic=True,
        gelman_rubin_mean_only=False,
    ):
        if burn is not None:
            burn = int(burn)

        m, t = r
        t_ = t[burn:]

        # for _, a, _ in names:
        #     print(a, t_[a].shape, t_[a][:5])

        ModelSummariser.mini_summary(t_, names)
        if t_.nchains > 1:
            ModelSummariser.gelman_rubin(t_, names, gelman_rubin_mean_only)
        if dic:
            dic(m, t_)
        ModelSummariser.traceplots(t_, output_dir)


class ModelPostProcessor():
    def print_params(params):
        print('Parameters:')
        for a, param in params:
            print(' - %s: %s' % (a, ', '.join(['%.5f' % x for x in param])))

    def extract_beta(t, dist):
        if dist == stats.norm:
            return dist.fit(t)
        elif dist == stats.halfcauchy:
            raise ValueError('Half Cauchy not supported')
            # return (dist.fit(t, floc=0)[1],)
        elif dist == stats.invgamma:
            f = dist.fit(t, floc=0)
            return (f[0], f[2])
        elif dist == stats.expon:
            return (1 / dist.fit(t, floc=0)[1],)
        else:
            raise ValueError('Dist not supported: %s' % str(dist))

    def extract_betas(trace, names):
        betas = []
        for dims, a, dist in names:
            if dims==0:
                betas += [(a, extract_beta(trace[a], dist))]
            elif dims==1:
                T = trace[a]
                for i in range(T.shape[1]):
                    betas += [(a, extract_beta(T[:, i], dist))]
        return dict(betas)


    def dist_fitter(t, dist):
        if dist == stats.norm:
            return dist.fit(t)
        elif dist == stats.halfcauchy:
            raise ValueError('Half Cauchy not supported')
            # return dist.fit(t, floc=0)
        elif dist == stats.invgamma:
            x = dist.fit(t, floc=0)
            return x
        else:
            raise ValueError('Dist not supported: %s' % str(dist))
        return fitted

    def simple_fit(a, t, i, dist, verbose):
        fitted = ModelPostProcessor.dist_fitter(t, dist)
        if verbose:
            print(' -', a, i, fitted)
        return [(i, fitted, dist)]

    def beta_extracter_version2(trace, names, verbose=False):
        betas = {}
        for dims, a, dist in names:
            T = trace[a]
            if dims == 0:
                betas[a] = ModelPostProcessor.simple_fit(
                    a, T, 0, dist, verbose
                )
            elif dims == 1:
                A = []
                if T.ndim==2:
                    for i in range(T.shape[1]):
                        A += ModelPostProcessor.simple_fit(
                            a, T[:, i], i, dist, verbose
                        )
                elif T.ndim==3:
                    for i in range(T.shape[2]):
                        A += ModelPostProcessor.simple_fit(
                            a, T[:, :, i], i, dist, verbose
                        )
                else:
                    raise ValueError()
                betas[a] = A
            else:
                raise RuntimeError('Dims not supported: %i' % dims)
        return betas

def base_gdp_model_fn(y, iters, tune_iters):
    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 1e6)
        sd = pm.InverseGamma('sd', 1, 1)
        y_hat = pm.Normal('y', mu=mu, sd=sd, observed=y)

        trace = pm.sample(
            int(iters),
            tune=int(tune_iters),
            step=pm.Metropolis(),
            njobs=4,
        )

    return model, trace

def simple_australian_model_fn(y, iters, tune_iters):
    with pm.Model() as model:
        d = pm.Normal('d', mu=0, sd=1e6, observed=y)
        trace = pm.backends.ndarray.NDArray(name='.', model=model, vars=[d])
        trace.samples = {}
        trace.samples['d'] = y
        mt = pm.backends.base.MultiTrace([trace])

    # with pm.Model() as model:
    #     d = pm.Normal('d', mu=0, sd=1e6)
    #     y_hat = pm.Normal('y', mu=d, sd=1e6, observed=y)
    #
    #     trace = pm.sample(
    #         int(iters),
    #         tune=int(tune_iters),
    #         step=pm.Metropolis(),
    #         njobs=4,
    #     )

    return model, mt


def simple_international_model_fn(Y, iters, tune_iters):
    # Hierarchical National GDP Model:
    #  - common distribution for mean and variation of national gdp growth; and
    #  - distribution for each nations gdp growth.

    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 1e6)
        sd = pm.InverseGamma('sd', alpha=1, beta=1)

        Y_hat = [
            pm.Normal('y%i' % i, mu, sd, observed=Y[:, i][np.isfinite(Y[:, i])])
            for i in range(Y.shape[1])
        ]

        step = pm.Metropolis()
        trace = pm.sample(
            int(iters),
            tune=int(tune_iters),
            step=step,
            njobs=4,
        )

    return model, trace


def international_shared_variance_model_fn(
    Y, iters, tune_iters
):
    with pm.Model() as model:
        MU = pm.Normal('mu', 0, 1e6, shape=Y.shape[1])
        sd = pm.InverseGamma('sd', alpha=1, beta=1)

        Y_hat = [
            pm.Normal('y%i' % i, MU[i], sd, observed=Y[:, i][np.isfinite(Y[:, i])])
            for i in range(Y.shape[1])
        ]

        step = pm.Metropolis()
        trace = pm.sample(
            int(iters),
            tune=int(tune_iters),
            step=step,
            njobs=4,
        )

    return model, trace


def internationally_influenced_australian_model_fn(
    y, priors, iters, tune_iters
):
    with pm.Model() as model:
        mu = pm.Normal(
            'mu', mu=priors['mu'][0][1][0], sd=priors['mu'][0][1][1]
        )
        sd = pm.InverseGamma(
            'sd', alpha=priors['sd'][0][1][0], beta=priors['sd'][0][1][1]
        )
        y_hat = pm.Normal('y', mu=mu, sd=sd, observed=y)

        step = pm.Metropolis()
        trace = pm.sample(
            int(iters),
            tune=int(tune_iters),
            step=step,
            njobs=4,
        )

    return model, trace


def correlated_sectors_model_fn(X, iters, tune_iters):
    with pm.Model() as model:
        SD_gva = pm.InverseGamma('SD_gva', 1, 1, shape=X.shape[1])
        mu_eco = pm.Normal(
            'mu_eco', mu=0, sd=1e6, observed=np.mean(X, axis=1).reshape(-1, 1)
        )
        D_gva = pm.Normal(
            'D_gva', mu=mu_eco, sd=SD_gva, observed=X, shape=X.shape[1]
        )

        step = pm.Metropolis()
        trace = pm.sample(
            int(iters),
            trace=[mu_eco, SD_gva],
            tune=int(tune_iters),
            step=step,
            njobs=4,
        )

    return model, trace
