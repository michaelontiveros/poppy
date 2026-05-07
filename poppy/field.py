import jax
import functools
from poppy.constant import INT, POLYNOMIAL, BLOCKSIZE
from poppy.modular import mod
from poppy.linear import inv1, dot33

class field:
    def __init__(self, p, n = 1, inverse = True, dtype = 'float32'):    
        self.p = p # Field characteristic.
        self.n = n # Field degree.
        self.q = p**n if n*jax.numpy.log2(p) < 63 else None # Field order.
        self.dtype = dtype
        self.inv = self._inv() if inverse else None # Multiplicative inverse mod p.
        self.basis = self._basis() # Power basis.
        self.dual = self._dual()   # Dual basis.

    def __repr__(self):
        return f'FIELD {self.q}'
     
    def _inv(self):
        def mul(a,b):
            return mod(a*b, self.p)
        def inv_jit(ABC, i):
            C = mul(ABC[i-2, 0], ABC[i-2, 2])
            ABC = ABC.at[i-1, 2].set(C)   
            return ABC, mul(ABC[i-1, 1], C)
        @jax.jit
        def inv_scan():    
            A = jax.numpy.arange(1, self.p, dtype = self.dtype)
            AA = jax.numpy.concatenate([jax.numpy.ones(1, dtype = self.dtype), jax.numpy.flip(A[1:])])
            B = jax.numpy.flip(jax.lax.associative_scan(mul,AA))
            C = jax.numpy.ones(self.p - 1, dtype = self.dtype).at[0].set(self.p - 1)
            ABC = jax.numpy.vstack([A,B,C]).T       
            return jax.numpy.concatenate([jax.numpy.zeros(1, dtype = self.dtype), jax.lax.scan(inv_jit, ABC, A.astype(INT))[1]])
        return inv_scan()
   
    def _basis(self):
        def id(a,i):
            return a
        stack = jax.vmap(id, (None,0))
        def neg(a):
            return mod(-a, self.p)
        def matmul(a,b):
            return mod(a@b, self.p)
        # V is the vector of subleading coefficients of the irreducible polynomial.
        V = jax.numpy.array(POLYNOMIAL[self.p][self.n][:-1], dtype = self.dtype)
        # M is a matrix root of the irreducible polynomial.
        M = jax.numpy.zeros((self.n,self.n), dtype = self.dtype).at[:-1,1:].set(jax.numpy.eye(self.n-1, dtype = self.dtype)).at[-1].set(neg(V))
        # B is the array of powers of M.
        B = jax.lax.associative_scan(matmul, stack(M,jax.numpy.arange(self.n, dtype = self.dtype)).at[0].set(jax.numpy.eye(self.n, dtype = self.dtype)))
        return B
  
    def _dual(self):
        A = jax.numpy.array(POLYNOMIAL[self.p][self.n][:-1], dtype = self.dtype)
        R = self.basis[1]
        Ri = inv1(R, self.inv, BLOCKSIZE)
        DD = jax.numpy.zeros((self.n,self.n,self.n), dtype = self.dtype).at[0,:,:].set(mod(-Ri*A[0],self.p))
        def dual_scan(b,i):
            b = b.at[i].set(mod(Ri@b[i-1]-Ri*A[i],self.p))
            return b, b[i]
        DD = jax.lax.scan(dual_scan,DD,jax.numpy.arange(1,self.n))[0]
        C = dot33(DD,self.basis,self.p)
        Ci = inv1(C, self.inv, BLOCKSIZE)
        D = mod(DD@(Ci.reshape((1,self.n,self.n))),self.p)
        return D

    def leg(self):
        R = jax.numpy.arange(self.p)
        return (-jax.numpy.ones(self.p, dtype = self.dtype)).at[(R*R) % self.p].set(1).at[0].set(0)

# BEGIN REGISTER FIELD
def flatten_field(f):
    children = (f.basis, f.dual, f.inv)
    aux_data = (f.p, f.n, f.q, f.dtype)
    return (children, aux_data)
def unflatten_field(aux_data, children):
    f = object.__new__(field)
    f.basis, f.dual, f.inv = children
    f.p, f.n, f.q, f.dtype = aux_data
    return f
jax.tree_util.register_pytree_node(field, flatten_field, unflatten_field)
# END REGISTER FIELD
