import jax
import conway_polynomials
import functools
from poppy.modular import mulmod, matmulmod, negmod
from poppy.linear import invmod, DTYPE, BLOCKSIZE

# BEGIN FIELD

# A finite field is a polynomial ring modulo an irreducible polynomial.
POLYNOMIAL = conway_polynomials.database()

p12 = 3329
p16 = 65521
p22 = 4194191
p30 = 999999733

POLYNOMIAL[p22] = {}
POLYNOMIAL[p30] = {}

class field:
    def __init__(self, p, n, inv = True):    
        self.p = p # Field characteristic.
        self.n = n # Field degree.
        self.q = p**n if n*jax.numpy.log2(p) < 63 else None # Field order.
        self.INV = self.inv() if inv else None # Multiplicative inverse mod p.
        self.BASIS = self.basis() # Power basis.
        self.DUAL = self.dual()   # Dual basis.

    def __repr__(self):
        return f'field {self.q}'
     
    def inv(self):
        @jax.jit
        def mul(a,b):
            return mulmod(a, b, self.p)
        @functools.partial(jax.jit, static_argnums = 1)
        def inv_jit(ABC, i):
            C = mul(ABC[i-2, 0], ABC[i-2, 2])
            ABC = ABC.at[i-1, 2].set(C)   
            return ABC, mul(ABC[i-1, 1], C)
        @jax.jit
        def inv_scan():    
            A = jax.numpy.arange(1, self.p, dtype = DTYPE)
            AA = jax.numpy.concatenate([jax.numpy.ones(1, dtype = DTYPE), jax.numpy.flip(A[1:])])
            B = jax.numpy.flip(jax.lax.associative_scan(mul,AA))
            C = jax.numpy.ones(self.p - 1, dtype = DTYPE).at[0].set(self.p - 1)
            ABC = jax.numpy.vstack([A,B,C]).T       
            return jax.numpy.concatenate([jax.numpy.zeros(1, dtype = DTYPE), jax.lax.scan(inv_jit, ABC, A)[1]])
        return inv_scan()
   
    def basis(self):
        @jax.jit
        def id(a,i):
            return a
        stack = jax.vmap(id, (None,0))
        @jax.jit
        def neg(a):
            return negmod(a, self.p)
        @jax.jit
        def matmul(a,b):
            return matmulmod(a,b, self.p)
        # V is the vector of subleading coefficients of the irreducible polynomial.
        V = jax.numpy.array(POLYNOMIAL[self.p][self.n][:-1], dtype = DTYPE)
        # M is a matrix root of the irreducible polynomial.
        M = jax.numpy.zeros((self.n,self.n), dtype = DTYPE).at[:-1,1:].set(jax.numpy.eye(self.n-1, dtype = DTYPE)).at[-1].set(neg(V))
        # B is the array of powers of M.
        B = jax.lax.associative_scan(matmul, stack(M,jax.numpy.arange(self.n, dtype = DTYPE)).at[0].set(jax.numpy.eye(self.n, dtype = DTYPE)))
        return B
  
    def dual(self):
        A = jax.numpy.array(POLYNOMIAL[self.p][self.n][:-1], dtype = DTYPE)
        R = self.BASIS[1]
        Ri = invmod(R, self.INV, BLOCKSIZE)
        DD = jax.numpy.zeros((self.n,self.n,self.n), dtype = DTYPE).at[0,:,:].set((-Ri*A[0])%self.p)
        def dualscan(b,i):
            b = b.at[i].set((Ri@b[i-1]-Ri*A[i])%self.p)
            return b, b[i]
        DD = jax.lax.scan(dualscan,DD,jax.numpy.arange(1,self.n))[0]
        C = jax.numpy.tensordot(DD,self.BASIS,axes = ([0,2],[0,1]))%self.p
        Ci = invmod(C, self.INV, BLOCKSIZE)
        D = DD@(Ci.reshape((1,self.n,self.n)))%self.p
        return D   

    def leg(self):
        R = jax.numpy.arange(self.p)
        return (-jax.numpy.ones(self.p, dtype = DTYPE)).at[(R*R) % self.p].set(1).at[0].set(0)

def flatten_field(f):
    children = (f.BASIS, f.DUAL, f.INV)
    aux_data = (f.p, f.n, f.q)
    return (children, aux_data)
def unflatten_field(aux_data, children):
    f = object.__new__(field)
    f.BASIS, f.DUAL, f.INV = children
    f.p, f.n, f.q = aux_data
    return f
jax.tree_util.register_pytree_node(field, flatten_field, unflatten_field)

# END FIELD