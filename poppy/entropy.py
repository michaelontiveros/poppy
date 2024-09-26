import jax
from poppy.array import zeros, ones

# Maxim Kontsevich: The 1.5-logarithm. 1995
# Jean-Louis Cathelineau: Infinitesimal dilogarithms, extensions and cohomology. 2011
def entropy(x):
  @jax.jit
  def sum(sy,i):
    s,y = sy
    y = x*y
    s = s+i*y
    return (s,y),i
  b = x.shape[0]
  f = x.field
  z = jax.lax.scan(sum,(zeros(b,f),ones(b,f)),f.INV[1:])[0][0]
  for i in range(1,x.field.n):
    z = z.frb()
  return z