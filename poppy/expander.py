import jax 
import functools
from poppy.linear import DTYPE
from poppy.field import field, POLYNOMIAL
from poppy.ring import Z2, M2
from poppy.group import psl2mod, pgl2mod 

# BEGIN EXPANDER

@functools.partial(jax.jit, static_argnums = 0)
def pS1(p): # A point on the circle x*x + y*y = (p-1) mod p.
    def norm(a):
        return a[0]*a[0]+a[1]*a[1]
    R = Z2(p)
    N = jax.vmap(norm)(R)%p
    M = jax.numpy.where(N == (p-1), N, -1)
    return R[jax.numpy.argmax(M)]

@functools.partial(jax.jit, static_argnums = (0,1))
def S3(p,r): # The integer three sphere x*x + y*y + z*z + t*t = r*r.
    def norm(a):
        return a[0,0]*a[0,0]+a[0,1]*a[0,1]+a[1,0]*a[1,0]+a[1,1]*a[1,1]
    R = M2(2*r)-r
    N = jax.vmap(norm)(R)
    return R, jax.numpy.where(N == p,N,-1)

def S(p,q): # p+1 generators for the group PSL2q if p is a quadratic residue mod q, else PGL2q.
    xy = pS1(q)
    x, y = xy[0], xy[1]
    def f(s):
        a, b, c, d = s[0,0], s[0,1], s[1,0], s[1,1]
        return jax.numpy.array([[a+b*x+d*y, c+d*x-b*y],[-c+d*x-b*y, a-b*x-d*y]], dtype = DTYPE)%q
    R, i = S3(p,1+int(jax.numpy.sqrt(p)))
    i = jax.numpy.where(i >= 0)
    R0 = R[i,0,0]
    R1 = R[i,0,1]
    Sp = R[i[0][jax.numpy.where((R0 > 0) & ((R0%2)==1))[1]]] if (p % 4) == 1 else R[i[0][jax.numpy.where((R0 >= 0) & ((R0%2) == 0) & ((R0 > 0) | (R1 > 0)))[1]]]
    return jax.vmap(f)(Sp)

def lps(p,q): # The Lubotzky-Phillips-Sarnak expander graph is a p+1-regular Cayley graph for the group PSL2q or PGL2q.
    assert (p in POLYNOMIAL) and (q in POLYNOMIAL) and (p != q) and (p > 2) and (q > 2) and (q*q > 4*p)
    f = field(q,1)
    l = f.leg()[p%q]

    @jax.jit
    def normpgl(A):
        a, b = A[0,0], A[0,1]
        sa = jax.numpy.sign(a)
        c = f.INV[sa*a + (1-sa)*b]
        return (c*A)%q

    @jax.jit
    def normpsl(A):
        a, b = A[0,0], A[0,1]
        sa = jax.numpy.sign(a)
        sqa = jax.numpy.astype(jax.numpy.sign((q/2)-jax.numpy.astype(a, jax.numpy.float64)), DTYPE)
        sqb = jax.numpy.astype(jax.numpy.sign((q/2)-jax.numpy.astype(b, jax.numpy.float64)), DTYPE)
        s = sa*sqa + (1-sa)*sqb
        return (s*A)%q

    @jax.jit
    def norm(A):
        return jax.lax.cond(l == 1, normpsl, normpgl, A)
    V = jax.vmap(norm)(S(p,q))
    
    @jax.jit
    def enc(a):
        return jax.numpy.sum(a.ravel() * q**jax.numpy.arange(4), dtype = jax.numpy.uint32)

    @jax.jit
    def dec(x):
        d = x//q**3
        c = (x-d*q**3)//q**2
        b = (x-d*q**3-c*q**2)//q
        a = (x-d*q**3-c*q**2-b*q)
        return jax.numpy.array([[a,b],[c,d]], dtype = DTYPE)

    @jax.jit 
    def mul(x):
        a = jax.vmap(norm)(jax.numpy.tensordot(dec(x), V, axes = (1,1)).swapaxes(0,1)%q)
        return jax.vmap(enc)(a)

    G, i = psl2mod(q) if l == 1 else pgl2mod(q)
    graph = jax.vmap(mul)(G)
    return graph, i

# END EXPANDER