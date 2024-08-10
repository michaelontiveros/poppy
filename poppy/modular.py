import jax
import functools

# BEGIN MODULAR ARITHMETIC

@functools.partial(jax.jit, static_argnums = 1)
def negmod(a,p):
    return (-a)%p

@functools.partial(jax.jit, static_argnums = 2)
def addmod(a,b,p):
    return (a+b)%p

@functools.partial(jax.jit, static_argnums = 2)
def submod(a,b,p):
    return (a-b)%p

@functools.partial(jax.jit, static_argnums = 2)
def mulmod(a,b,p):
    return (a*b)%p

@functools.partial(jax.jit, static_argnums = 2)
def matmulmod(a,b,p):
    return (a@b)%p

# END MODULAR ARITHMETIC