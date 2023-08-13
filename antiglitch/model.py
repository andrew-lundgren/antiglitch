import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from .cplxdist import CplxNormal

#from .utils import center, fglitch_from_sample

from functools import partial
rfft = partial(rfft, norm='ortho')
irfft = partial(irfft, norm='ortho')

freqs = jnp.linspace(0, 4096, 513)

@jax.jit
def fsignal(freqs, f0, gbw):
    return jnp.exp(-0.5*gbw*(jnp.log(freqs) - jnp.log(f0))**2)

def fglitch_from_sample(amp_r, amp_i, f0, gbw, time, **kwargs):
    ftmp = (amp_r+1.j*amp_i)*fsignal(freqs, f0, gbw)*jnp.exp(-1.j*(time*freqs))
    return ftmp

# Bayesian model
def glitch_model(freqs, invasd, data=None):
    """Reparamaterised physical model"""
    amp_r = numpyro.sample("amp_r", dist.Normal(0, 200))
    amp_i = numpyro.sample("amp_i", dist.Normal(0, 200))
    t = numpyro.sample("time", dist.Normal(0, 0.02))
    f0 = numpyro.sample('f0', dist.Uniform(10., 400.))
    gbw = numpyro.sample('gbw', dist.Uniform(0.25, 8.))

    with numpyro.plate("data", len(data)):
        numpyro.sample("y", CplxNormal((amp_r+1.j*amp_i)*jnp.exp(-2.j*jnp.pi*t*freqs)*invasd*fsignal(freqs, f0, gbw), 0.5), obs=data)
        
def new_model(freqs, invasd, data=None):
    """Reparamaterised physical model"""
    amp_r = numpyro.sample("amp_r", dist.Normal(0, 200))
    amp_i = numpyro.sample("amp_i", dist.Normal(0, 200))
    tx = numpyro.sample("t_", dist.Beta(2, 2))
    t = numpyro.deterministic("time", 0.05*(2.*tx-1.))
    f0 = numpyro.sample('f0', dist.LogUniform(10., 600.))
    gbw = numpyro.sample('gbw', dist.LogUniform(0.25, 8.))

    with numpyro.plate("data", len(data)):
        numpyro.sample("y", CplxNormal((amp_r+1.j*amp_i)*jnp.exp(-2.j*jnp.pi*t*freqs)*invasd*fsignal(freqs, f0, gbw), 0.5), obs=data)
