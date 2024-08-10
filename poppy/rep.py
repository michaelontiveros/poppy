import jax
import functools
from poppy.linear import DTYPE

# BEGIN RESHAPE

@functools.partial(jax.jit, static_argnums = 1)
def block(a,f): 
    s = a.shape
    n = f.n 
    return a.reshape(s[:-2]+(s[-2]//n,n, s[-1]//n,n)).swapaxes(-2,-3)
@functools.partial(jax.jit, static_argnums = 1)
def unblock(a,f):
    s = a.shape
    n = f.n
    return a.swapaxes(-2,-3).reshape(s[:-4]+(s[-4]*n, s[-3]*n))

# END RESHAPE
# BEGIN LIFT/PROJECT

@functools.partial(jax.jit, static_argnums = 1)
def int2vec(i,f):
    return jax.numpy.floor_divide(jax.numpy.expand_dims(i,3)*jax.numpy.ones(f.n, dtype = DTYPE).reshape((1,1,1,f.n)), jax.numpy.power(f.p*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.arange(f.n, dtype = DTYPE)).reshape((1,1,1,f.n)))%f.p

@functools.partial(jax.jit, static_argnums = 1)
def vec2int(v,f):
    return jax.numpy.sum(v*jax.numpy.power(f.p*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.arange(f.n, dtype = DTYPE)).reshape((1,1,1,f.n)), axis = -1, dtype = DTYPE)

@functools.partial(jax.jit, static_argnums = 1)
def vec2mat(v,f):
    return jax.numpy.tensordot(v,f.BASIS, axes = ([-1],[0]))%f.p

@jax.jit
def mat2vec(m):
    return m.swapaxes(-1,-2).T[0].T

@functools.partial(jax.jit, static_argnums = 1)
def int2mat(i,f):
    return vec2mat(int2vec(i,f),f)

@functools.partial(jax.jit, static_argnums = 1)
def mat2int(m,f):
    return vec2int(mat2vec(m),f)

# END LIFT/PROJECT