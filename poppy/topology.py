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
    D = degree 
    H = len(perm)                            # Number of half-edges.
    V = H//D                                 # Number of vertices.
    RE = jax.numpy.arange(H)                 # Edge representatives.
    BE = jax.numpy.eye(H, dtype = DTYPE)     # Edge basis.
    L = array(BE[:,perm[:H//2]],field)       # Left edges.
    R = array(BE[:,perm[H//2:]],field)       # Right edges.
    RF = unique(V, D, orbit(V, D, RE, perm)) # Face representatives.
    F = len(RF)                              # Number of faces.
    RS = jax.numpy.arange(F)[:,None]         # Source face representatives.
    RT = rotation(V,D)                       # Target face representatives.
    BF = jax.numpy.eye(F, dtype = DTYPE)     # Face basis.
    BS = BF[:,RE.at[RF].set(RS)]             # Source face basis.
    BT = BS[:,RT]                            # Target face basis.
    S = array(BS,field)     # Shape F H.
    T = array(BT,field)     # Shape F H.
    FE = array(jax.numpy.sum(BE[perm].reshape((H,V,D)),axis=2),field) # Shape H V.
    d3 = zeros((1,F),field) # Shape 1 F.
    d2 = (S-T)@(L-R)        # Shape F E.
    d1 = (L-R).t()@FE       # Shape E V.
    d0 = zeros((V,1),field) # Shape V 1.
    return d3,d2,d1,d0

@jax.jit
def boundary(d): # d is a tuple of arrays.
    b = True
    for i in range(len(d)-1):
        b = b & (d[i+1]@d[i]).vanishes()
    return b

@jax.jit
def homology(d): # d is the boundary operator of a chain complex.
    H = ()
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Hi = ker.mod(im)
        H = H+(Hi,)
        ker = k
    return H # Homology groups.

@jax.jit
def betti(d): # d is the boundary operator of a chain complex.
    B = ()
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Bi = ker.rankmod(im)
        B = B+(Bi,)
        ker = k
    return B # Betti numbers.

@jax.jit
def euler(d): # d is the boundary operator of a chain complex.
    X = 0
    ker = d[0].ker()
    for i in range(1,len(d)):
        k,im = d[i].kerim()
        Bi = ker.rankmod(im)
        X = X-(-1)**(i%2)*Bi
        ker = k
    return X # Euler characteristic.