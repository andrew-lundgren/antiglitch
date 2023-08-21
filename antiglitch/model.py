import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO # , autoguide
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
    ftmp = (amp_r+1.j*amp_i)*fsignal(freqs, f0, gbw)*jnp.exp(-2.j*jnp.pi*(time*freqs))
    return ftmp

def fglitch_normed(invasd, amp_r, amp_i, f0, gbw, time, **kwargs):
    raw = invasd*fsignal(freqs, f0, gbw)
    norm = 1./jnp.sqrt(jnp.sum(jnp.real(raw*raw.conjugate())))
    ftmp = norm*(amp_r+1.j*amp_i)*raw*jnp.exp(-2.j*jnp.pi*(time*freqs))
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
        
def glitch_model2(freqs, invasd, data=None):
    """Narrow beta dist for time, true Gaussian in r"""
    amp_r = numpyro.sample("amp_r", dist.Normal(0, 400))
    amp_i = numpyro.sample("amp_i", dist.Normal(0, 400))
    amp = numpyro.deterministic("amp", jnp.sqrt(amp_r*amp_r+amp_i*amp_i))
    numpyro.factor('r', -jnp.log(amp))

    tx = numpyro.sample("t_", dist.Beta(2, 2))
    t = numpyro.deterministic("time", 0.01*(2.*tx-1.))
    f0 = numpyro.sample('f0', dist.Uniform(10., 600.))
    gbw = numpyro.sample('gbw', dist.Uniform(0.25, 8.))
    
    raw = invasd*fsignal(freqs, f0, gbw)
    #norm = 1./jnp.sqrt(jnp.sum(jnp.real(raw*raw.conjugate())))

    with numpyro.plate("data", len(data)):
        numpyro.sample("y", CplxNormal((amp_r+1.j*amp_i)*jnp.exp(-2.j*jnp.pi*t*freqs)*raw, 0.5), obs=data)

def create_mle_model(freqs, maxamp=10000., progress_bar=False):
    """MLE model"""
    def mle_model(freqs, invasd, data=None):
        amp_r = numpyro.param("amp_r", 0., constraint=constraints.interval(-maxamp, maxamp))
        amp_i = numpyro.param("amp_i", 0., constraint=constraints.interval(-maxamp, maxamp))
        amp = numpyro.deterministic("amp", jnp.sqrt(amp_r*amp_r+amp_i*amp_i))

        t = numpyro.param("time", 0., constraint=constraints.interval(-0.01, 0.01))
        f0 = numpyro.param('f0', 100., constraint=constraints.interval(10., 600.))
        gbw = numpyro.param('gbw', 1., constraint=constraints.interval(0.25, 8.))

        raw = invasd*fsignal(freqs, f0, gbw)
        norm = 1./jnp.sqrt(jnp.sum(jnp.real(raw*raw.conjugate()))) 

        with numpyro.plate("data", len(data)):
            numpyro.sample("y", CplxNormal(norm*(amp_r+1.j*amp_i)*jnp.exp(-2.j*jnp.pi*t*freqs)*raw, 0.5), obs=data)

    def mle_guide(freqs, invasd, data=None):
        pass
    
    def train(data, seed = 101, num_steps = 10000):
        optimizer = numpyro.optim.Adam(step_size=1e-3)
        svi = SVI(mle_model, mle_guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(jax.random.PRNGKey(seed), num_steps,
                                freqs, jnp.array(data['invasd']), jnp.array(data['fdata']),
                                progress_bar = progress_bar)
        return svi_result
    
    return train
