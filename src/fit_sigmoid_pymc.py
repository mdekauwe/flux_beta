#!/usr/bin/env python3.4
"""
Fit a sigmoid to some data

That's all folks.
"""
__author__ = "Martin De Kauwe, Rhys Whitley"
__version__ = "1.0 (16.02.2017)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
import pymc3 as pm
import pandas as pd
from theano import tensor as tt

from scipy import optimize
from scipy.stats.mstats import mquantiles
import sys

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def fitMe(df, site, x_range, to_screen=False):

    with pm.Model() as sig_model:
        # Hyperpriors
        a = pm.Normal('a', mu=0, sd=1e2)
        b = pm.Normal('b', mu=0, sd=1e2)

        # model error
        sigma = pm.Uniform('sigma', lower=0, upper=1000)
        model = tt_sigmoid(df.sw.values, a, b)
        like = pm.Normal('like', mu=model, sd=sigma, observed=df.beta.values)

    with sig_model:
        try:
            # find an optimal start position
            start = pm.find_MAP(model=sig_model.model,
                                fmin=optimize.fmin_powell)
        except ValueError:
            return

        step = pm.Metropolis()

        try:
            mcmc_traces = pm.sample(5e4, step=step, start=start, njobs=-1)
        except ValueError:
            return

    make_plot(df, site, mcmc_traces, x_range, to_screen)

def tt_sigmoid(x, a, b):
    """
    Sigmoid function using tensor
    """
    return 1.0 / (1.0 + tt.exp(-a * (x - b)))

def np_sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

def make_plot(df, site, mcmc_traces, x_range, to_screen):

    #pm.traceplot(mcmc_traces)

    # posteriors for the parameters
    a_post = mcmc_traces["a"][:, None]
    b_post = mcmc_traces["b"][:, None]

    # mean prediction
    beta_pred = np_sigmoid(x_range.T, a_post, b_post)
    mean_pred = beta_pred.mean(axis=0)

    # vectorized bottom and top 2.5% quantiles for "confidence interval"
    qs = mquantiles(beta_pred, [0.025, 0.975], axis=0)

    plt.figure(figsize=(10, 6))
    plt.fill_between(x_range[:, 0], *qs, alpha=0.7, color="salmon")
    plt.plot(x_range, mean_pred, lw=2, ls="-", color="crimson")
    plt.scatter(df.sw.values, df.beta.values, color="k", s=50, alpha=0.5)
    plt.xlim(x_range.min(), x_range.max())
    plt.ylim(-0.02, 1.02)
    plt.xlabel("SW")
    plt.ylabel("Beta")
    #plt.legend(loc="upper left")
    if to_screen:
        plt.show()
    else:
        plt.savefig("plots/%s.png" % (site), dpi=80)


if __name__ == "__main__":

    site = "blah"
    x = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
    y = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])
    df = pd.DataFrame({'sw':x, 'beta':y})
    # dimension to plot along
    x_range = np.linspace(df.sw.min()-5, df.sw.max()+5, 100)[:, None]

    fitMe(df, site, x_range, to_screen=True)
