import jax
from poppy.array import zeros, ones

@jax.jit
def entropy(x):
# https://arxiv.org/abs/math/0008089
  @jax.jit
  def sum(sy,i):
    s,y = sy
    y = x*y
    s = s+i*y
    return (s,y),i
  b = x.shape[0]
  f = x.field
  return jax.lax.scan(sum,(zeros(b,f),ones(b,f)),f.INV[1:])[0][0]