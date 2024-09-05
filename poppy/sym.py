import jax
import functools
from poppy.ring import Z2, ZZ
from poppy.array import array, zeros

def factorial(n):
  return jax.numpy.arange(2,n+1).prod().item()

@functools.partial(jax.jit, static_argnums = 1)
def int2prm_jit(m,n): # The mth permutation on n elements.
  elem = jax.numpy.arange(n, dtype = jax.numpy.int8)
  shift = jax.numpy.arange(1,n+1, dtype = jax.numpy.int8)
  prm = jax.numpy.zeros(n, dtype = jax.numpy.int8)
  coeff = jax.numpy.zeros(n-1, dtype = jax.numpy.int8)
  for i in range(2,n+1): # Compute the factorial expansion of m.
    c = jax.numpy.astype(m%i, jax.numpy.int8)
    coeff = coeff.at[n-i].set(c)
    m //= i
  for i in range(n-1): # Translate the expansion to a permutation.
    j = jax.numpy.where(coeff[i] <= (n-i-1), coeff[i], n-i-1)
    prm = prm.at[i].set(elem[j])
    elem = jax.numpy.where(elem >= prm[i], elem[shift], elem)
  prm = prm.at[n-1].set(elem[0])
  return prm

def int2prm(m,n): # The mth permutations on n elements.
  # m is an array of nonnegative integers less than n!.
  return jax.vmap(int2prm_jit, in_axes = (0,None))(m,n)

@functools.partial(jax.jit, static_argnums = 1)
def prm2int_jit(prm,n): # The index of a permutation on n elements.
  elem = jax.numpy.arange(n, dtype = jax.numpy.int8) 
  coeff = jax.numpy.zeros(n-1, dtype = jax.numpy.int8)
  pos = jax.numpy.arange(1,n+1)
  for i in range(n-1): # Compute the factorial expansion of perm.
    c = elem[prm[i]]
    coeff = coeff.at[i].set(c)
    elem = jax.numpy.where(pos > prm[i], elem-1, elem)
  coeff = jax.numpy.flip(coeff)
  basis = jax.lax.associative_scan(jax.numpy.multiply, pos[:n-1]) # The factorial basis.
  m = (basis*coeff).sum()
  return m

def prm2int(prm,n): # The indices of permutations on n elements.
  # prm is an array of permutations.
  return jax.vmap(prm2int_jit, in_axes = (0,None))(prm,n)

@functools.partial(jax.jit, static_argnums = (0,1))
def sym(n,N): # The symmetric group on n elements.
  return int2prm(jax.numpy.arange(N),n) # N = n!.

@functools.partial(jax.jit, static_argnums = (0,1))
def symtbl(n,N): # The multiplication table of the symmetric group.
  def mul(ij):
    ij = int2prm(ij,n)
    k = prm2int_jit(ij[0][ij[1]],n)
    return k
  indices = Z2(N)
  table = jax.vmap(mul)(indices).reshape((N,N))
  return table

@jax.jit
def iterate(fx,i):
    f,x = fx
    return (f[x],x),f[x]

def orbit(prm):
  n = len(prm)
  i = jax.numpy.arange(n)
  return jax.lax.scan(iterate, (prm,prm), i)[1]

def cosets(prm):
  return jax.numpy.unique(orbit(prm), axis = 0)

def order(cyc,n): # The order of a cycle in the nth symmetric group.
  return jax.numpy.count_nonzero(1+jax.numpy.unique(cyc, size = n, fill_value = -1))

def prm2prt(prm): # The partition of a permutation.
  c = jax.numpy.unique(jax.numpy.sort(cosets(prm), axis = 0), axis = 1)
  n = len(prm)
  return jax.vmap(order, in_axes = (1,None))(c,n)

def prm2cyc(prm): # Factor a permutation into disjoint cycles.
  n = len(prm)
  c = orbit(prm)
  for j in range(n):
    i = jax.numpy.unique(c[:,j], size = n, return_index = True)[1]
    i = i.at[1:].set(jax.numpy.where(i[1:]==i[0], n,i[1:]))
    i = jax.numpy.sort(i)
    c = c.at[:,j].set(jax.numpy.where(i<n, c[:,j][i], n))
  idx = jax.numpy.unique(jax.numpy.sort(c, axis = 0), axis = 1, return_index = True)[1]
  return c[:,idx]

def prm2trp(prm): # Factor a permutation into transpositions.
  n = len(prm)
  c = prm2cyc(prm)
  t = jax.numpy.zeros((0,2), dtype = jax.numpy.int8)
  for j in range(c.shape[1]):
    i = jax.numpy.repeat(c[:,j],2)[1:-1]
    N = jax.numpy.argmax(i)
    N = N-(N%2) if i[N] == n else 2*n
    t = jax.numpy.vstack([t,i[:N].reshape((-1,2))])
  return t

def sgn(prm): # The sign of a permutation.
  return (-1)**len(prm2trp(prm))

def transpose(prm,t):
  tt = jax.numpy.flip(t)
  prm = prm.at[t].set(prm[tt])
  return prm,t

def trp2prm(trp,n): # Multiply transpositions in the nth symmetric group.
  prm = jax.numpy.arange(n, dtype = jax.numpy.int8)
  return jax.lax.scan(transpose, prm, trp)[0]

def prt2int(prt):
  return prt.sum().item()

def prt2dgm(prt): # Extract the diagram from a partition.
  n = prt2int(prt)
  m = prt.max()
  l = len(prt)
  indices = jax.numpy.arange(n, dtype = jax.numpy.int8)
  s = jax.lax.associative_scan(jax.numpy.add, prt)
  d = -jax.numpy.ones((l,m), dtype = jax.numpy.int8)
  d = d.at[0].set(indices[:m])
  for i in range(1,l):
    d = d.at[i,:prt[i]].set(indices[s[i-1]:s[i]])
  return d

def hook(dgm,ij):
  i = ij[0]
  j = ij[1]
  I = jax.numpy.arange(dgm.shape[0])
  J = jax.numpy.arange(dgm.shape[1])
  r = jax.numpy.count_nonzero(jax.numpy.where(J>=j,dgm[i,:],0))
  c = jax.numpy.count_nonzero(jax.numpy.where(I>i,dgm[:,j],0))
  return jax.numpy.where(r+c>0,r+c,1)

def dgm2dim(dgm): # Calculate the dimension of a representation indexed by a diagram.
  r = dgm.shape[0]
  c = dgm.shape[1]
  ij = ZZ(r,c)
  N = jax.numpy.max(dgm+1)
  return factorial(N)//jax.vmap(hook, in_axes = (None,0))(dgm+1,ij).prod()

def prt2dim(prt): # Calculate the dimension of a representation index by a partition.
  return dgm2dim(prt2dgm(prt))

def concat(A,B,ij): # The direct sum of permutations.
  i = ij[0]
  j = ij[1]
  return jax.numpy.concatenate([A[i],A.shape[1]+B[j]])

def prm2inv(prm): # Invert a permutation.
  return prm.at[prm].set(jax.numpy.arange(len(prm), dtype = jax.numpy.int8))

def prt2rng(prt,dgm): # Map a partition to the group ring of the symmetric group.
  p = prt[0].item()
  R = sym(p,factorial(p))
  for i in range(1,len(prt)):
    p = prt[i].item()
    S = sym(p,factorial(p))
    Z = ZZ(len(R),factorial(p))
    R = jax.vmap(concat, in_axes = (None,None,0))(R,S,Z)
  f = dgm.ravel()[jax.numpy.where(dgm.ravel() >= 0)]
  fi = prm2inv(f)
  return f[R[:,fi]]

def mul1(A,B,ij): # The ijth term of the product AB in the group ring.
  i = ij[0]
  j = ij[1]
  return A[i][B[j]]

def mul(A,B): # Symmetric roup ring multiplication.
  ij = ZZ(len(A),len(B))
  return jax.vmap(mul1, in_axes = (None,None,0))(A,B,ij)

def prt2dual(prt): # The dual partition.
  dgm = prt2dgm(prt)
  return jax.numpy.count_nonzero(dgm+1, axis = 0)

def prt2prj(prt): # The projector.
  n = prt2int(prt)
  dgm = prt2dgm(prt)
  dprt = prt2dual(prt)
  g1 = prt2rng(prt,dgm)
  g2 = prt2rng(dprt,dgm.T)
  c2 = jax.numpy.zeros(len(g2), dtype = jax.numpy.int8)
  for i in range(len(g2)):
    c2 = c2.at[i].set(sgn(g2[i]))
  g = mul(g1,g2)
  c = c2.repeat(len(g1))
  return g,c

def symbas(n): # The transpositions.
  ij = Z2(n)
  basis = jax.numpy.zeros((n*n,n), dtype = jax.numpy.int8)
  id = jax.numpy.arange(n, dtype = jax.numpy.int8)
  for i in range(n*n):
    basis = basis.at[i].set(transpose(id,ij[i])[0])
  return basis

def rngorb(g): # The symmetric group orbit in the group ring.
  n = g.shape[-1]
  N = factorial(n)
  b = sym(n,N).reshape((N,1,n))
  return jax.vmap(mul, in_axes = (0,None))(b,g)

def trpmul(g): # The transposition orbit in the group ring.
  n = g.shape[-1]
  trp = symbas(n).reshape((n*n,1,n))
  return jax.vmap(mul, in_axes = (0,None))(trp,g)

def int2vec(i,N): # The standard ith basis elements in N dimensions.
  id = jax.numpy.arange(N)
  return jax.numpy.where(i[None,:] == id[:,None],1,0)

def prt2bas(prt,f): # The basis of the Specht module of the partition over the field.
  prt = jax.numpy.sort(prt, descending = True)
  n = prt2int(prt)
  N = factorial(n)
  dim = prt2dim(prt)
  prj,c = prt2prj(prt)
  prjorb = rngorb(prj)
  intorb = jax.vmap(prm2int, in_axes = (0,None))(prjorb,n)
  vecorb = jax.vmap(int2vec, in_axes = (0,None))(intorb,N)
  cpos = jax.numpy.where(c>0,1,0).reshape((1,1,-1))
  cneg = jax.numpy.where(c<0,1,0).reshape((1,1,-1))
  posorb = array(cpos*vecorb,f)
  negorb = array(cneg*vecorb,f)
  orb = (posorb-negorb).separate().sum()
  _,prm = orb.imprm() 
  bas = prjorb[prm[0,:dim]]
  return bas,c

def bas2rep(bas,c,f): # The symmetric group representation over the Specht module.
  dim,N,n = bas.shape
  basorb = trpmul(bas)
  intorb = jax.vmap(jax.vmap(prm2int, in_axes = (0,None)), in_axes = (0,None))(basorb,n)
  vecorb = jax.vmap(jax.vmap(int2vec, in_axes = (0,None)), in_axes = (0,None))(intorb,N)
  cpos = jax.numpy.where(c>0,1,0).reshape((1,1,1,-1))
  cneg = jax.numpy.where(c<0,1,0).reshape((1,1,1,-1))
  posorb = (cpos*vecorb).swapaxes(1,-1)
  negorb = (cneg*vecorb).swapaxes(1,-1)
  g = (array(posorb[0],f)-array(negorb[0],f)).sum()
  rep = zeros((0,dim,dim),f)
  for i in range(n*n):
    gi = (array(posorb[i],f)-array(negorb[i],f)).sum()
    rep = rep.stack(g.hstack(-gi).ker()[:,:dim,dim:])
  return rep

def symrep(prt,f): # The symmetric group representation.
  bas,c = prt2bas(prt,f)
  return bas2rep(bas,c,f)