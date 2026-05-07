import jax 
import functools
from poppy.constant import INT, POLYNOMIAL
from poppy.modular import mod
from poppy.field import field
from poppy.ring import Z2, M2
from poppy.gl2 import psl2mod, pgl2mod 

@functools.partial(jax.jit, static_argnums = (0,1))
def pS1(p, dtype): # A point on the circle x*x + y*y = (p-1) mod p.
    def norm(a):
        return a[0]*a[0]+a[1]*a[1]
    R = Z2(p, dtype)
    N = jax.vmap(norm)(R)%p
    M = jax.numpy.where(N == (p-1), N, -1)
    return R[jax.numpy.argmax(M)]

@functools.partial(jax.jit, static_argnums = (0,1,2))
def S3(p,r, dtype): # The integer three sphere x*x + y*y + z*z + t*t = r*r.
    def norm(a):
        return a[0,0]*a[0,0]+a[0,1]*a[0,1]+a[1,0]*a[1,0]+a[1,1]*a[1,1]
    R = M2(2*r, dtype)-r
    N = jax.vmap(norm)(R)
    return R, jax.numpy.where(N == p,N,-1)

def S(p,q, dtype): # p+1 generators for the group PSL2q if p is a quadratic residue mod q, else PGL2q.
    xy = pS1(q, dtype)
    x, y = xy[0], xy[1]
    def f(s):
        a, b, c, d = s[0,0], s[0,1], s[1,0], s[1,1]
        return mod(jax.numpy.array([[a+b*x+d*y, c+d*x-b*y],[-c+d*x-b*y, a-b*x-d*y]], dtype = dtype),q)
    R, i = S3(p,1+int(jax.numpy.sqrt(p)), dtype)
    i = jax.numpy.where(i >= 0)
    R0 = R[i,0,0]
    R1 = R[i,0,1]
    Sp = R[i[0][jax.numpy.where((R0 > 0) & ((R0%2)==1))[1]]] if (p % 4) == 1 else R[i[0][jax.numpy.where((R0 >= 0) & ((R0%2) == 0) & ((R0 > 0) | (R1 > 0)))[1]]]
    return jax.vmap(f)(Sp)

def lps(p,q, dtype = 'float32'): # The Lubotzky-Phillips-Sarnak expander graph is a p+1-regular Cayley graph for the group PSL2q or PGL2q.
    assert (p in POLYNOMIAL) and (q in POLYNOMIAL) and (p != q) and (p > 2) and (q > 2) and (q*q > 4*p)
    f = field(q, dtype = dtype)
    l = f.leg()[p%q]

    @jax.jit
    def normpgl(A):
        a, b = A[0,0], A[0,1]
        sa = jax.numpy.sign(a)
        c = f.inv[(sa*a + (1-sa)*b).astype(INT)]
        return mod(c*A,q)

    @jax.jit
    def normpsl(A):
        a, b = A[0,0], A[0,1]
        sa = jax.numpy.sign(a)
        sqa = jax.numpy.sign((q/2)-a).astype(dtype)
        sqb = jax.numpy.sign((q/2)-b).astype(dtype)
        s = sa*sqa + (1-sa)*sqb
        return mod(s*A,q)

    @jax.jit
    def norm(A):
        return jax.lax.cond(l == 1, normpsl, normpgl, A)
    V = jax.vmap(norm)(S(p,q, dtype))
    
    @jax.jit
    def enc(a):
        return jax.numpy.sum(a.ravel() * q**jax.numpy.arange(4, dtype = INT), dtype = dtype)

    @jax.jit
    def dec(x):
        d = x//q**3
        c = (x-d*q**3)//q**2
        b = (x-d*q**3-c*q**2)//q
        a = (x-d*q**3-c*q**2-b*q)
        return jax.numpy.array([[a,b],[c,d]], dtype = dtype)

    @jax.jit 
    def mul(x):
        a = jax.vmap(norm)(mod(jax.numpy.tensordot(dec(x), V, axes = (1,1)).swapaxes(0,1),q))
        return jax.vmap(enc)(a)

    G, i = psl2mod(q, dtype) if l == 1 else pgl2mod(q, dtype)
    graph = jax.vmap(mul)(G)
    return graph, i
