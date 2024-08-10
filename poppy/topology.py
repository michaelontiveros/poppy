import jax
import functools
from poppy.linear import DTYPE
from poppy.array import array, zeros, ones

# BEGIN TOPOLOGY

@functools.partial(jax.jit, static_argnums = 0)
def polygon(n,field): # The boundary operator of a polygon.
    V,E,F = n,n,1
    S = array(jax.numpy.eye(V,E, k = 0, dtype = DTYPE), field)                 # Source vertices.
    T = array(jax.numpy.eye(V,E, k = 1, dtype = DTYPE).at[-1,0].set(1), field) # Target vertices.
    d0 = zeros((1,V),field)
    d1 = S-T
    d2 = ones((E,F),field)
    d3 = zeros((F,1),field)
    return d0,d1,d2,d3

@jax.jit
def iterate(ap,i):
    a,perm = ap
    return (a[perm],perm),a[perm]

@functools.partial(jax.jit, static_argnums = 0)
def involution(n,perm): # Construct an involution from a permutation.
    return jax.numpy.arange(n).at[perm[:n//2]].set(perm[n//2:]).at[perm[n//2:]].set(perm[:n//2])

@functools.partial(jax.jit, static_argnums = 0)
def rotation(n,perm): # Rotate edges of the polygon around vertices of the identification space.
    shift = jax.numpy.arange(1,n+1)%n
    return shift[involution(n,perm)]

@functools.partial(jax.jit, static_argnums = 1)
def unique_jit(a,n):
    return jax.numpy.unique(a, size = n, fill_value = n)
unique = jax.vmap(unique_jit, in_axes = (1,None))

def surface(perm,field): # The boundary operator of a closed orientable surface made out of a polygon.
    F = 1            # Number of faces.
    E = len(perm)//2 # Number of edges.
    RE = jax.numpy.arange(2*E)             # Edge representatives.
    BE = jax.numpy.eye(2*E, dtype = DTYPE) # Edge basis.
    L = array(BE[:,perm[:E]],field)        # Left edges.
    R = array(BE[:,perm[E:]],field)        # Right edges.
    RV = jax.numpy.unique(unique(jax.lax.scan(iterate,(RE,rotation(2*E,perm)),RE)[1], 2*E), axis = 0) # Vertex representatives.
    V = len(RV)      # Number of vertices.
    RS = jax.numpy.arange(V)[:,None]     # Source vertex representatives.
    RT = jax.numpy.arange(1,2*E+1)%(2*E) # Target vertex representatives.
    BV = jax.numpy.eye(V, dtype = DTYPE) # Vertex basis.
    BS = BV[:,RE.at[RV].set(RS)]         # Source vertex basis.
    BT = BS[:,RT]                        # Target vertex basis.
    S = array(BS,field)     # Shape V 2E.
    T = array(BT,field)     # Shape V 2E.
    d0 = zeros((1,V),field) # Shape 1 V.
    d1 = (S-T)@(L-R)        # Shape V E.
    d2 = zeros((E,F),field) # Shape E F.
    d3 = zeros((F,1),field) # Shape F 1.
    return d0,d1,d2,d3

@jax.jit
def is_boundary(d): # d is a tuple of arrays.
    boundary = True
    for i in range(len(d)-1):
        boundary = boundary & (d[i+1]@d[i]).is_zero()
    return boundary

@jax.jit
def homology(d): # d is the boundary operator of a chain complex.
    H = ()
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Hi = ker.mod(im)
        H = H+(Hi,)
        ker = k
    return H

@jax.jit
def betti(d): # d is the boundary operator of a chain complex.
    B = ()
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Bi = ker.rankmod(im)
        B = B+(Bi,)
        ker = k
    return B

@jax.jit
def euler_characteristic(d): # d is the boundary operator of a chain complex.
    x = 0
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Bi = ker.rankmod(im)
        x = x-(-1)**(i%2)*Bi
        ker = k
    return x

# END TOPOLOGY