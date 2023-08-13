import sys
from glob import glob
import pickle
from joblib import Parallel, delayed

import numpy as np

import jax
import numpyro
from numpyro import infer
import arviz as az

numpyro.set_host_device_count(4)

from antiglitch import freqs, to_fd, extract_glitch, rfft

# Frequency-domain signal model
from antiglitch import fsignal

# Bayesian model
from antiglitch import glitch_model, new_model

sampler = infer.MCMC(
    infer.NUTS(new_model, init_strategy=infer.init_to_sample),
    num_warmup=3000,
    num_samples=2000,
    num_chains=4,
    progress_bar=False,
    jit_model_args=True,
)

# Read all glitches of a certain type from the .npz files
ifo = sys.argv[1]
datadir = sys.argv[2]

def process(ifo, key, ii, ff):
    npz = np.load(ff)
    invasd, whts, _ = extract_glitch(npz)
    fglitch = to_fd(whts)
    sampler.run(jax.random.PRNGKey(0),
                    freqs[1:], invasd[1:],
                    data=fglitch[1:])

    itrace = az.from_numpyro(sampler)
    summ = az.summary(itrace, kind='stats')
    return ((ifo, key, ii), summ)

simlst = []
for key in ['tomte','blip','koi','low_blip']:
    files = sorted(glob(f"{datadir}/{ifo}-{key}-*.npz"))
    if not files:
        continue
    for ii, ff in enumerate(files):
        simlst.append((ifo, key, ii, ff))

result = dict([process(*arg) for arg in simlst])

with open(f"Aug2023-{ifo}-physical-results.pkl", 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
