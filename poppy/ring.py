import jax
import functools

@functools.partial(jax.jit, static_argnums = (0,1,2))
def ZZ(m,n, dtype): # The finite ring Z/m x Z/n.
    Zm = jax.numpy.arange(m, dtype = dtype)
    Zn = jax.numpy.arange(n, dtype = dtype)
    ZmZn = jax.numpy.array([jax.numpy.tile(Zm,n), jax.numpy.repeat(Zn,m)]).T
    return ZmZn

@functools.partial(jax.jit, static_argnums = (0,1))
def Z2(q, dtype): # The finite ring Z/q x Z/q.
    return ZZ(q,q, dtype)

@functools.partial(jax.jit, static_argnums = (0,1))
def M2(q, dtype): # The finite ring M_2( Z/q ).
    Z2q = Z2(q, dtype)
    M2q = jax.numpy.array([jax.numpy.tile(Z2q.T,q*q).T, jax.numpy.repeat(Z2q.T,q*q).reshape(2,-1).T]).swapaxes(0,1).reshape(-1,2,2)
    return M2q
