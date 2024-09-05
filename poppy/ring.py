import jax
import functools

@functools.partial(jax.jit, static_argnums = (0,1))
def ZZ(m,n): # The finite ring Z/m x Z/n.
    Zm = jax.numpy.arange(m)
    Zn = jax.numpy.arange(n)
    ZmZn = jax.numpy.array([jax.numpy.tile(Zm,n), jax.numpy.repeat(Zn,m)]).T
    return ZmZn

@functools.partial(jax.jit, static_argnums = 0)
def Z2(q): # The finite ring Z/q x Z/q.
    return ZZ(q,q)

@functools.partial(jax.jit, static_argnums = 0)
def M2(q): # The finite ring M_2( Z/q ).
    Z2q = Z2(q)
    M2q = jax.numpy.array([jax.numpy.tile(Z2q.T,q*q).T, jax.numpy.repeat(Z2q.T,q*q).reshape(2,-1).T]).swapaxes(0,1).reshape(-1,2,2)
    return M2q