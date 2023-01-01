import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig
import jax
import jax.numpy as jnp

#from .utils import center, fglitch_from_sample

from functools import partial
rfft = partial(rfft, norm='ortho')
irfft = partial(irfft, norm='ortho')

freqs = jnp.linspace(0, np.pi, 513)

@jax.jit
def fsignal(freqs, f0, gbw):
    return jnp.exp(-0.5*gbw*(jnp.log(freqs) - jnp.log(f0))**2)

def fglitch_from_sample(amp_r, amp_i, f0, gbw, time, **kwargs):
    ftmp = (amp_r+1.j*amp_i)*fsignal(freqs, f0, gbw)*jnp.exp(-1.j*(time*freqs))
    return ftmp

# Bayesian model
def glitch_model(freqs, invasd, data=None):
    amp_r = numpyro.sample("amp_r", dist.Normal(0, 200))
    amp_i = numpyro.sample("amp_i", dist.Normal(0, 50))
    t = numpyro.sample("time", dist.Normal(0, 20))
    f0 = numpyro.sample('f0', dist.Uniform(0.0025, 0.3))
    gbw = numpyro.sample('gbw', dist.Uniform(0.25, 8.))

    with numpyro.plate("data", len(data)):
        numpyro.sample("y", CplxNormal((amp_r+1.j*amp_i)*jnp.exp(-1.j*t*freqs)*invasd*fsignal(freqs, f0, gbw), 0.5), obs=data)
