import jax
import functools
from poppy.ring import Z2

def factorial(n):
  return jax.numpy.arange(2,n+1).prod().item()

@functools.partial(jax.jit, static_argnums = 1)
def int2perm_jit(m,n): # The mth permutation on n elements.
  elem = jax.numpy.arange(n, dtype = jax.numpy.int8)
  shift = jax.numpy.arange(1,n+1, dtype = jax.numpy.int8)
  perm = jax.numpy.zeros(n, dtype = jax.numpy.int8)
  coeff = jax.numpy.zeros(n-1, dtype = jax.numpy.int8)
  for i in range(2,n+1): # Compute the factorial expansion of m.
    c = jax.numpy.astype(m%i, jax.numpy.int8)
    coeff = coeff.at[n-i].set(c)
    m //= i
  for i in range(n-1): # Translate the expansion to a permutation.
    j = jax.numpy.where(coeff[i] <= (n-i-1), coeff[i], n-i-1)
    perm = perm.at[i].set(elem[j])
    elem = jax.numpy.where(elem >= perm[i], elem[shift], elem)
  perm = perm.at[n-1].set(elem[0])
  return perm

def int2perm(m,n): # The mth permutations on n elements.
  # m is an array of nonnegative integers less than n!.
  return jax.vmap(int2perm_jit, in_axes = (0,None))(m,n)

@functools.partial(jax.jit, static_argnums = 1)
def perm2int_jit(perm,n): # The index of a permutation on n elements.
  elem = jax.numpy.arange(n, dtype = jax.numpy.int8) 
  coeff = jax.numpy.zeros(n-1, dtype = jax.numpy.int8)
  pos = jax.numpy.arange(1,n+1)
  for i in range(n-1): # Compute the factorial expansion of perm.
    c = elem[perm[i]]
    coeff = coeff.at[i].set(c)
    elem = jax.numpy.where(pos > perm[i], elem-1, elem)
  coeff = jax.numpy.flip(coeff)
  basis = jax.lax.associative_scan(jax.numpy.multiply, pos[:n-1]) # The factorial basis.
  m = (basis*coeff).sum()
  return m

def perm2int(perm,n): # The indices of permutations on n elements.
  # perm is an array of permutations.
  return jax.vmap(perm2int_jit, in_axes = (0,None))(perm,n)

@functools.partial(jax.jit, static_argnums = (0,1))
def sym(n,N): # The symmetric group on n elements.
  return int2perm(jax.numpy.arange(N),n) # N = n!.

@functools.partial(jax.jit, static_argnums = (0,1))
def symtable(n,N): # The multiplication table of the symmetric group.
  def mul(ij):
    ij = int2perm(ij,n)
    k = perm2int_jit(ij[0][ij[1]],n)
    return k
  indices = Z2(N)
  table = jax.vmap(mul)(indices).reshape((N,N))
  return table