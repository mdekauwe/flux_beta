#!/usr/bin/env python3.4
"""
Fit a sigmoid to some data

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (15.02.2017)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
import pymc3 as pm3
import pandas as pd
from theano import tensor as tt

from scipy import optimize
from scipy.stats.mstats import mquantiles
import sys

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def fitMe(df):

    with pm3.Model() as sig_model:
        a = pm3.Normal('a', mu=0, sd=1e2)
        b = pm3.Normal('b', mu=0, sd=1e2)
        sigma = pm3.Uniform('sigma', lower=0, upper=1000)
        model = tt_sigmoid(df.sw.values, a, b)
        error = pm3.Normal('error', mu=model, sd=sigma,
                           observed=df.beta.values)

    with sig_model:
        # find an optimal start position
        start = pm3.find_MAP(model=sig_model.model, fmin=optimize.fmin_powell)
        step = pm3.Metropolis()
        mcmc_traces = pm3.sample(5e4, step=step, start=start, njobs=-1)

    make_plot(mcmc_traces)

def tt_sigmoid(x, a, b):
    """
    Sigmoid function using tensor
    """
    return 1.0 / (1.0 + tt.exp(-a * (x - b)))

def np_sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

def make_plot(mcmc_traces):

    pm3.traceplot(mcmc_traces)

    # posteriors for the parameters
    a_post = mcmc_traces["a"][:, None]
    b_post = mcmc_traces["b"][:, None]

    # dimension to plot along
    #swx = np.linspace(400, 650, 1000)[:, None]
    swx = np.linspace(0, 16, 16)[:, None]

    # mean prediction
    beta_pred = np_sigmoid(swx.T, a_post, b_post)
    mean_pred = beta_pred.mean(axis=0)

    # vectorized bottom and top 2.5% quantiles for "confidence interval"
    qs = mquantiles(beta_pred, [0.025, 0.975], axis=0)

    plt.figure(figsize=(10, 6))
    plt.fill_between(swx[:, 0], *qs, alpha=0.7, color="salmon")
    plt.plot(swx, mean_pred, lw=2, ls="-", color="crimson")
    plt.scatter(df.sw.values, df.beta.values, color="k", s=50, alpha=0.5)
    plt.xlim(swx.min(), swx.max())
    plt.ylim(-0.02, 1.02)
    plt.xlabel("SW")
    plt.ylabel("Beta")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":

    x = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
    y = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])
    df = pd.DataFrame({'sw':x, 'beta':y})

    fitMe(df)

    """
    fname = "/Users/mdekauwe/Desktop/crap.csv"
    df = pd.read_csv(fname)
    df.columns = ["sw", "beta"]

    # Have points where the soil is full but no or not much ET, screen these
    screen = np.max(df.sw) * 0.9
    #df.beta = np.where(df.sw>screen, np.max(df.beta), df.beta)
    df.beta = np.where(df.sw>screen, 1.0, df.beta)

    swx = np.linspace(400, 650, 1000)[:, None]
    """
