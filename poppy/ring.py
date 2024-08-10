import jax
import functools

# BEGIN RING

@functools.partial(jax.jit, static_argnums = 0)
def Z2(q): # The finite ring Z/q x Z/q.
    Zq = jax.numpy.arange(q)
    Z2q = jax.numpy.array([jax.numpy.tile(Zq,q), jax.numpy.repeat(Zq,q)]).T
    return Z2q

@functools.partial(jax.jit, static_argnums = 0)
def M2(q): # The finite ring M_2( Z/q ).
    Z2q = Z2(q)
    M2q = jax.numpy.array([jax.numpy.tile(Z2q.T,q*q).T, jax.numpy.repeat(Z2q.T,q*q).reshape(2,-1).T]).swapaxes(0,1).reshape(-1,2,2)
    return M2q

# END RING