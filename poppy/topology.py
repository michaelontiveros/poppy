import jax
import functools
from poppy.constant import DTYPE
from poppy.array import array, zeros, ones

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

@functools.partial(jax.jit, static_argnums = 0)
def involution(n,perm): # Construct an involution from a permutation.
    return jax.numpy.arange(n).at[perm[:n//2]].set(perm[n//2:]).at[perm[n//2:]].set(perm[:n//2])

@functools.partial(jax.jit, static_argnums = (0,1))
def rotation(V,D): # Rotate half-edges around vertices.
    return jax.numpy.tile(jax.numpy.arange(1,D+1)%D,V)+jax.numpy.repeat(jax.numpy.arange(0,V*D,D),D)

@functools.partial(jax.jit, static_argnums = (0,1))
def generator(V,D,perm): # Permute half-edges.
    return rotation(V,D)[involution(V*D,perm)]

@jax.jit
def iterate(ap,i):
    a,perm = ap
    return (a[perm],perm),a[perm]

@functools.partial(jax.jit, static_argnums = (0,1))
def orbit(V,D,a,perm):
    return jax.lax.scan(iterate, (a,generator(V,D,perm)), a)[1]

@functools.partial(jax.jit, static_argnums = (0,1))
def unique_jit(V,D,a):
    return jax.numpy.unique(a, size = V*D, fill_value = V*D)

def unique(V,D,a):
    return jax.numpy.unique(jax.vmap(unique_jit, in_axes = (None,None,1))(V,D,a), axis = 0)

def graph(degree,perm,field): # The coboundary operator of an orientable regular ribbon graph.
    perm = jax.numpy.array(perm, dtype = DTYPE) # Half-edge identifications.
    H = len(perm) # Number of half-edges.
    E = H//2      # Number of edges
    D = degree    # Vertex degree.
    V = H//D      # Number of vertices.
    RE = jax.numpy.arange(H)             # Edge representatives.
    RF = unique(V,D, orbit(V,D,RE,perm)) # Face representatives.
    F = len(RF)   # Number of faces.
    RS = jax.numpy.arange(F)[:,None]     # Source face representatives.
    RT = rotation(V,D)                   # Target face representatives.
    BF = jax.numpy.eye(F, dtype = DTYPE)                 # Face basis.
    BS = BF[:,RE.at[RF].set(RS)]                         # Source face basis.
    BT = BS[:,RT]                                        # Target face basis.
    BE = jax.numpy.eye(H, dtype = DTYPE)                 # Edge basis.
    BV = jax.numpy.sum(BE[perm].reshape((H,V,D)),axis=2) # Vertex basis.
    L = array(BE[:,perm[:E]],field) # Left half-edges.  H E.
    R = array(BE[:,perm[E:]],field) # Right half-edges. H E.
    S = array(BS,field)             # Source faces.     F H.
    T = array(BT,field)             # Target faces.     F H.
    dV = array(BV,field)    # H V.
    dE = L-R                # H E.
    dF = S-T                # F H.
    d3 = zeros((1,F),field) # 1 F.
    d2 = dF@dE              # F E.
    d1 = dE.t()@dV          # E V.
    d0 = zeros((V,1),field) # V 1.
    return d3,d2,d1,d0

@jax.jit
def boundary(d): # d is a tuple of arrays.
    b = True
    for i in range(len(d)-1):
        b = b & (d[i+1]@d[i]).vanishes()
    return b

@jax.jit
def homology(d): 
    # Homology groups.
    # d is the boundary operator of a chain complex.
    H = ()
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Hi = ker.mod(im)
        H = H+(Hi,)
        ker = k
    return H

@jax.jit
def betti(d): 
    # Betti numbers.
    # d is the boundary operator of a chain complex.
    B = ()
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Bi = ker.rankmod(im)
        B = B+(Bi,)
        ker = k
    return B

@jax.jit
def euler(d): 
    # Euler characteristic.
    # d is the boundary operator of a chain complex.
    X = 0
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Bi = ker.rankmod(im)
        X = X-(-1)**(i%2)*Bi
        ker = k
    return X