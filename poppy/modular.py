import jax

@jax.jit
def mod(a,p):
    return jax.numpy.round(a)%p
