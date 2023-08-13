import numpy as np
import jax.numpy as jnp
from jax import lax
from numpyro.distributions import Distribution, Normal
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample

# FIXME: Maybe we should check the type, or check that abs(x) is real
class _whatever(constraints._SingletonConstraint):
    """Not sure how to check for complex number, so just be happy with anything"""
    def __call__(self, x):
        return True

    def feasible_like(self, prototype):
        return jax.numpy.zeros_like(prototype)
        
class CplxNormal(Distribution):
    arg_constraints = {"loc": _whatever(), "scale": constraints.positive}
    support = _whatever()
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(CplxNormal, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps_r = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        eps_i = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + eps_r * self.scale + 1.j* eps_i * self.scale

    @validate_sample
    def log_prob(self, value):
        normalize_term = 2*jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * jnp.real(value_scaled.conjugate()*value_scaled) - normalize_term

    # FIXME: Do CDF and inverse CDF make sense for a complex variable?
    def cdf(self, value):
        return NotImplementedError
        scaled = (value - self.loc) / self.scale
        return ndtr(scaled)

    def icdf(self, q):
        return NotImplementedError
        return self.loc + self.scale * ndtri(q)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale**2, self.batch_shape)
    
# FIXME: Add a distribution which has an inverse PSD weighting