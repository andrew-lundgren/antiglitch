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
from antiglitch import glitch_model2

sampler = infer.MCMC(
    infer.NUTS(glitch_model2, init_strategy=infer.init_to_median, dense_mass=True),
    num_warmup=2000,
    num_samples=1000,
    num_chains=4,
    progress_bar=False,
    jit_model_args=True,
)

# Read all glitches of a certain type from the .npz files
datadir = sys.argv[1]
ifo = sys.argv[2]
key = sys.argv[3]

def process(ifo, key, ii, ff):
    npz = np.load(ff)
    invasd, whts, _ = extract_glitch(npz)
    fglitch = to_fd(whts)
    sampler.run(jax.random.PRNGKey(0),
                    freqs[1:], invasd[1:],
                    data=fglitch[1:])

    itrace = az.from_numpyro(sampler)
    summ = az.summary(itrace, kind='all', round_to=8)
    return ((ifo, key, ii), summ)

simlst = []
files = sorted(glob(f"{datadir}/{ifo}-{key}-*.npz"))
if not files:
    exit()
for ii, ff in enumerate(files):
    simlst.append((ifo, key, ii, ff))

result = dict([process(*arg) for arg in simlst])

with open(f"Aug2023v2-{ifo}-{key}-physical-results.pkl", 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
