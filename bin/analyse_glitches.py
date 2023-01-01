import sys
from glob import glob
import pickle

import numpy as np
import scipy.signal as sig

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist, infer
from cplxdist import CplxNormal
import arviz as az

numpyro.set_host_device_count(4)

datadir = 'data'

from glitchmodel import extract_glitch, rfft, irfft

# Frequency-domain signal model
@jax.jit
def fsignal(freqs, f0, gbw):
    return jnp.exp(-0.5*gbw*(jnp.log(freqs) - jnp.log(f0))**2)

# Bayesian model
def glitch_model(freqs, invasd, data=None):
    amp_r = numpyro.sample("amp_r", dist.Normal(0, 200))
    amp_i = numpyro.sample("amp_i", dist.Normal(0, 50))
    t = numpyro.sample("time", dist.Normal(0, 20))
    f0 = numpyro.sample('f0', dist.Uniform(0.0025, 0.3))
    gbw = numpyro.sample('gbw', dist.Uniform(0.25, 8.))

    with numpyro.plate("data", len(data)):
        numpyro.sample("y", CplxNormal((amp_r+1.j*amp_i)*jnp.exp(-1.j*t*freqs)*invasd*fsignal(freqs, f0, gbw), 0.5), obs=data)

sampler = infer.MCMC(
    infer.NUTS(glitch_model),
    num_warmup=4000,
    num_samples=2000,
    num_chains=4,
    progress_bar=True,
)

freqs = np.linspace(0, np.pi, 513)

ifo = sys.argv[1]
key = sys.argv[2]
files = sorted(glob(f"{datadir}/{ifo}-{key}-*.npz"))

result = {}
for ii, ff in enumerate(files):
    npz = np.load(ff)
    invasd, whts = extract_glitch(npz)
    fglitch = rfft(np.roll(whts, -len(whts)//2))
    sampler.run(jax.random.PRNGKey(0),
                    freqs[1:], invasd[1:],
                    data=fglitch[1:])

    itrace = az.from_numpyro(sampler)
    summ = az.summary(itrace, kind='stats')
    result[(ifo, key, ii)] = summ
    print('Calculated', ifo, key, ii)

with open(f"{ifo}-{key}-results.pkl", 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)