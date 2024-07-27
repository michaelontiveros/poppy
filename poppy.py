# BEGIN POPPY
# BEGIN INIT

import jax
import conway_polynomials
import functools
import matplotlib.pyplot

# 64 bit integer arrays encode numbers in finite fields.
jax.config.update("jax_enable_x64", True)
DTYPE = jax.numpy.int64

# A finite field is a polynomial ring modulo an irreducible polynomial.
POLYNOMIAL = conway_polynomials.database()

p12 = 3329
p16 = 65521
p22 = 4194191
p30 = 999999733

POLYNOMIAL[p22] = {}
POLYNOMIAL[p30] = {}

# The pseudo random number generator has a default seed.
SEED = 0 

# END INIT
# BEGIN MODULAR ARITHMETIC

@functools.partial(jax.jit, static_argnums = 1)
def negmod(a,p):
    return (-a)%p

@functools.partial(jax.jit, static_argnums = 2)
def addmod(a,b,p):
    return (a+b)%p

@functools.partial(jax.jit, static_argnums = 2)
def submod(a,b,p):
    return (a-b)%p

@functools.partial(jax.jit, static_argnums = 2)
def mulmod(a,b,p):
    return (a*b)%p

@functools.partial(jax.jit, static_argnums = 2)
def matmulmod(a,b,p):
    return (a@b)%p

# END MODULAR ARITHMETIC
# BEGIN LINEAR ALGEBRA

@jax.jit
def transpose(a):
    return a.swapaxes(-2,-1) 

@jax.jit
def tracemod(a,p):
    return jax.numpy.trace(a, axis1=-2, axis2=-1)%p

@jax.jit
def mtrsm(a,b,p): # Triangular solve mod p.
    R = jax.numpy.arange(len(a), dtype = DTYPE) # a has shape (r,c).
    def mtrsm_vmap(bb): # bb is the matrix b.
        def mtrsm_scan(bc): # bc is a column of bb.
            def f(x,j):
                x = x.at[j].set((bc[j] - jax.numpy.dot(a[j], x)) % p)
                return x, x[j]  
            return jax.lax.scan(f, jax.numpy.where(R == 0, bc[0], 0), R[1:])[0] # scan the rows of a.
        return jax.vmap(mtrsm_scan)(bb.T).T  # vmap the columns of b.
    return mtrsm_vmap(b)

@jax.jit
def mgetrf2(aperm, inv): # Sequential lu decomposition mod p.
    p = 1+inv[-1] # p is prime.
    I = jax.numpy.arange(aperm.shape[0])
    J = jax.numpy.arange(aperm.shape[1]-1)
    R = jax.numpy.arange(min(len(I),len(J)))
    def f(ap, i):
        j = jax.numpy.argmax(jax.numpy.where(I >= i, ap[:,i], -1)) # Search column i for j.
        ap = ap.at[[i,j],:].set( ap[[j,i],:] ) # Swap rows i and j.
        ap = ap.at[:,i].set( jax.numpy.where( I > i, (ap[:,i] * inv[ ap[i,i] ]) % p, ap[:,i] ) )  # Scale column i.
        ap = ap.at[:,:-1].set((ap[:,:-1] - jax.numpy.where((I[:,None] > i) & (J[None,:] > i), jax.numpy.outer(ap[:,i], ap[i,:-1]), 0)) % p) # Update block D.     
        return ap, i
    return jax.lax.scan(f, aperm, R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def mgetrf(a, inv, b): # Blocked lu decompposition mod p.
    p = 1+inv[-1]
    m = min(a.shape)
    perm = jax.numpy.arange(len(a))
    for i in range(0, m, b):
        bb = min(m-i, b)
        ap = mgetrf2(jax.numpy.hstack([a[i:, i:i+bb], jax.numpy.arange(i,len(a)).reshape((-1,1))]), inv)
        perm = perm.at[i:].set(perm[ap[:,-1]])
        a = a.at[i:,:].set(a[ap[:,-1], :]) # Swap rows.
        a = a.at[i:, i:i+bb].set( ap[:,:-1])  # Update block C.
        a = a.at[i:i+bb, i+bb:].set(mtrsm( a[i:i+bb, i:i+bb], a[i:i+bb, i+bb:], p )) # Update block B.
        a = a.at[i+bb:, i+bb:].set((a[i+bb:, i+bb:] - jax.lax.dot(a[i+bb: , i:i+bb], a[i:i+bb, i+bb:])) % p) # Update block D.
    l = jax.numpy.fill_diagonal(jax.numpy.tril(a), 1, inplace = False)
    u = jax.numpy.tril(a.T).T
    d = jax.numpy.diagonal(u)
    iperm = jax.numpy.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm
mgetrf_vmap = jax.vmap(mgetrf, in_axes = (0, None, None))

def invmod(a, inv, b): # Matrix inverse mod p.
    if len(a) == 1:
        return inv[a[0,0]].reshape((1,1))
    p = 1+inv[-1]
    I = jax.numpy.eye(len(a), dtype = DTYPE)
    l, u, d, iperm = mgetrf(a, inv, b)
    D = inv[d]
    L = mtrsm(l, I, p) # L = 1/l.
    U = mtrsm((D*u%p).T, D*I, p).T # U = 1/u.      
    return (U@L%p)[:,iperm]

def invmod_vmap(a, inv, b): # Matrix inverse mod p.
    if a.shape[1] == 1:
        return inv[a[:,0,0]].reshape((a.shape[0],1,1))
    p = 1+inv[-1]
    I = jax.numpy.eye(a.shape[1], dtype = DTYPE)
    def inverse(A):
        l, u, d, iperm = mgetrf(A, inv, b)
        D = inv[d]
        L = mtrsm(l, I, p) # L = 1/l.
        U = mtrsm((D*u%p).T, D*I, p).T # U = 1/u.      
        return (U@L%p)[:,iperm]
    return jax.vmap(inverse)(a)

@jax.jit
def ftrsm(a,b,p): # Triangular solve over a finite field.
    R = jax.numpy.arange(len(a), dtype = DTYPE) # a has shape (r,c,n,n).
    ZERO = jax.numpy.zeros((b.shape[-1],b.shape[-1]), dtype = DTYPE) # b has shape (c,d,n,n).
    def ftrsm_vmap(bb): # bb is the array b.
        def ftrsm_scan(bc): # bc has shape (c,n,n). it is a column of bb.
            def f(x,j):
                x = x.at[j].set((bc[j] - jax.numpy.tensordot(a[j], x, axes = ([0,2],[0,1]))) % p)
                return x, x[j]  
            return jax.lax.scan( f, jax.numpy.where( R[:,None,None] == 0, bc[0], ZERO ), R[1:] )[0] # scan the rows of a.
        return jax.vmap(ftrsm_scan)(bb.swapaxes(0,1)).swapaxes(0,1)  # vmap the columns of b.
    return ftrsm_vmap(b)

@jax.jit
def fgetrf2(aperm, inv): # Sequential lu decomposition over a finite field.
    a, perm, parity = aperm
    p = 1+inv[-1] # p is prime.
    I = jax.numpy.arange(a.shape[0])
    J = jax.numpy.arange(a.shape[1])
    R = jax.numpy.arange(min(len(I),len(J)))
    def f(ap, i):
        a, perm, parity = ap # a has shape (r,c,n,n). perm has shape (r,). parity has shape (1,).
        ai = a[:,i,:,:].reshape((len(I),-1)).max(axis = 1)
        j = jax.numpy.argmax(jax.numpy.where(I >= i, ai, -1)) # Search column i for j.
        a = a.at[[i,j],:,:,:].set(a[[j,i],:,:,:]) # Swap rows i and j.
        perm = perm.at[[i,j],].set(perm[[j,i],]) # Record swap.
        a = a.at[:,i,:,:].set(jax.numpy.where(I[:,None,None] > i, (jax.numpy.tensordot(a[:,i,:,:], invmod(a[i,i,:,:], inv, 32), axes = (2,0))) % p, a[:,i,:,:])) # Scale column i.
        a = a.at[:,:,:,:].set((a[:,:,:,:] - jax.numpy.where((I[:,None,None,None] > i) & (J[None,:,None,None] > i), jax.numpy.tensordot(a[:,i,:,:], a[i,:,:,:], axes = (2,1)).swapaxes(1,2), 0)) % p) # Update block D.    
        parity = (parity + jax.numpy.count_nonzero(i-j)) % 2
        return (a, perm, parity), j
    return jax.lax.scan(f, aperm, R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def fgetrf(a, inv, b): # Blocked lu decompposition over a finite field.
    p = 1+inv[-1]
    r,c = a.shape[0], a.shape[1]
    m = min(r,c)
    R = jax.numpy.arange(r)
    perm = jax.numpy.arange(r)
    parity = jax.numpy.zeros(1, dtype = DTYPE)
    for i in range(0, m, b):
        bb = min(m-i, b)
        ai, permi, pari = fgetrf2((a[i:,i:i+bb,:,:], R[i:], 0), inv) # a has shape (r-i,bb,n,n). pi has shape (r-i,)
        parity = (parity + pari) % 2
        perm = perm.at[i:].set(perm[permi])
        a = a.at[i:,:,:,:].set(a[permi,:,:,:]) # Swap rows.
        a = a.at[i:,i:i+bb,:,:].set(ai)  # Update block C.
        a = a.at[i:i+bb,i+bb:,:,:].set(ftrsm(a[i:i+bb,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], p)) # Update block B.
        a = a.at[i+bb:,i+bb:,:,:].set((a[i+bb:,i+bb:,:,:] - jax.numpy.tensordot(a[i+bb: ,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], axes = ([1,3],[0,2])).swapaxes(1,2)) % p) # Update block D.
    I = jax.numpy.eye(a.shape[-1], dtype = DTYPE)
    l = jax.numpy.where((R[:,None,None,None] - R[None,:,None,None]) > 0, a, 0)
    l = jax.numpy.where(R[:,None,None,None] == R[None,:,None,None], I, l)
    u = jax.numpy.where((R[:,None,None,None] - R[None,:,None,None]) <= 0, a, 0)
    d = jax.numpy.diagonal(a, offset = 0, axis1 = 0, axis2 = 1).swapaxes(0,2).swapaxes(1,2)
    iperm = jax.numpy.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm, parity
fgetrf_vmap = jax.vmap(fgetrf, in_axes = (0, None, None))

def fdet(a, inv, p, b): # Matrix determinant over a finite field.
    def matmul(A,B):
        return matmulmod(A,B,p)
    l, u, d, iperm, parity = fgetrf(a, inv, b)
    return jax.numpy.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p

def fdet_vmap(a, inv, p, b): # Matrix determinant over a finite field.
    def matmul(A,B):
        return matmulmod(A,B,p)
    def det(A):
        l, u, d, iperm, parity = fgetrf(A, inv, b)
        return jax.numpy.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p
    return jax.vmap(det)(a)

# END LINEAR ALGEBRA
# BEGIN FIELD

class field:
    def __init__(self, p, n, inv = True):    
        self.p = p
        self.n = n
        self.q = p**n if n*jax.numpy.log2(p) < 63 else None
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
        # X is the array of powers of M.
        X = jax.lax.associative_scan(matmul, stack(M,jax.numpy.arange(self.n, dtype = DTYPE)).at[0].set(jax.numpy.eye(self.n, dtype = DTYPE)))
        return X

  
    def dual(self):
        A = jax.numpy.array(POLYNOMIAL[self.p][self.n][:-1], dtype = DTYPE)
        R = self.BASIS[1]
        Ri = invmod(R,self.INV,32)
        DD = jax.numpy.zeros((self.n,self.n,self.n), dtype = DTYPE).at[0,:,:].set((-Ri*A[0])%self.p)
        def dualscan(b,i):
            b = b.at[i].set((Ri@b[i-1]-Ri*A[i])%self.p)
            return b, b[i]
        DD = jax.lax.scan(dualscan,DD,jax.numpy.arange(1,self.n))[0]
        C = jax.numpy.tensordot(DD,self.BASIS,axes = ([0,2],[0,1]))%self.p
        Ci = invmod(C,self.INV,32)
        D = DD@(Ci.reshape((1,self.n,self.n)))%self.p
        return D   

    def leg(self):
        R = jax.numpy.arange(self.p)
        return (-jax.numpy.ones(self.p, dtype = DTYPE)).at[(R*R) % self.p].set(1).at[0].set(0)

def flatten_field(f):
    children = (f.BASIS, f.INV)
    aux_data = (f.p, f.n, f.q)
    return (children, aux_data)
def unflatten_field(aux_data, children):
    f = object.__new__(field)
    f.BASIS, f.INV = children
    f.p, f.n, f.q = aux_data
    return f
jax.tree_util.register_pytree_node(field, flatten_field, unflatten_field)

# END FIELD
# BEGIN RESHAPE

@functools.partial(jax.jit, static_argnums = 1)
def block(a,f): 
    s = a.shape
    n = f.n 
    return a.reshape(s[:-2] + (s[-2]//n, n, s[-1]//n, n)).swapaxes(-2, -3)

@functools.partial(jax.jit, static_argnums = 1)
def ravel(a,f): 
    n = f.n
    return block(a,f).reshape((len(a), -1, n, n))

@functools.partial(jax.jit, static_argnums = (1,2))
def unravel(i,f,s):   
    n = f.n    
    return i.reshape(s[:-2] + (s[-2]//n, s[-1]//n, n, n)).swapaxes(-2, -3).reshape(s)

# END RESHAPE
# BEGIN LIFT/PROJECT

@functools.partial(jax.jit, static_argnums = 1)
def int2vec(i,f):
    return jax.numpy.floor_divide(jax.numpy.expand_dims(i,3)*jax.numpy.ones(f.n, dtype = DTYPE).reshape((1,1,1,f.n)), jax.numpy.power(f.p*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.arange(f.n, dtype = DTYPE)).reshape((1,1,1,f.n)))%f.p

@functools.partial(jax.jit, static_argnums = 1)
def vec2int(v,f):
    return jax.numpy.sum(v*jax.numpy.power(f.p*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.arange(f.n, dtype = DTYPE)).reshape((1,1,1,f.n)), axis = -1, dtype = DTYPE)

@functools.partial(jax.jit, static_argnums = 1)
def vec2mat(v,f):
    return jax.numpy.tensordot(v,f.BASIS, axes = ([-1],[0]))%f.p

@jax.jit
def mat2vec(m):
    return m.swapaxes(-1,-2).T[0].T

@functools.partial(jax.jit, static_argnums = 1)
def int2mat(i,f):
    return vec2mat(int2vec(i,f),f)

@functools.partial(jax.jit, static_argnums = 1)
def mat2int(m,f):
    return vec2int(mat2vec(m),f)

# END LIFT/PROJECT
# BEGIN ARRAY

class array:
    def __init__(self, a, dtype):
        if type(a) == int:
            a = jax.numpy.array([a], dtype = DTYPE)
        elif type(a) == list:
            a = jax.numpy.array(a, dtype = DTYPE)
        if len(a.shape) == 1:
            a = a.reshape((a.shape[0],1,1))
        elif len(a.shape) == 2:
            a = a.reshape((1,a.shape[0],a.shape[1]))
        elif len(a.shape) > 3:
            print('ERROR: poppy arrays are three dimensional.')
            return
        self.field = dtype 
        self.shape = a.shape
        self.VEC = int2vec(a, self.field)

    def __repr__(self):
        return f'shape {self.shape[0]} {self.shape[1]} {self.shape[2]} over ' + repr(self.field) 

    def __neg__(self):
        a = object.__new__(array)
        a.field = self.field
        a.shape = self.shape
        a.VEC = negmod(self.VEC, self.field.p)
        return a
    
    def __add__(self, a):
        a = array1(a,self.field) if type(a) == int else a
        b = object.__new__(array)
        b.field = self.field
        b.VEC = addmod(self.VEC, a.VEC, self.field.p)
        b.shape = b.VEC.shape[:-1]
        return b
    def __radd__(self, a):
        return self.__add__(a)
 
    def __sub__(self, a):
        a = array1(a,self.field) if type(a) == int else a
        b = object.__new__(array)
        b.field = self.field
        b.VEC = submod(self.VEC, a.VEC, self.field.p)
        b.shape = b.VEC.shape[:-1]
        return b
    def __rsub__(self, a):
        return self.__sub__(a)
    
    def __mul__(self, a):
        a = array1(a,self.field) if type(a) == int else a
        b = object.__new__(array)
        b.field = self.field
        b.VEC = (jax.numpy.expand_dims(self.VEC,3)@vec2mat(a.VEC,a.field))[:,:,:,0,:]%self.field.p
        b.shape = b.VEC.shape[:-1]
        return b
    def __rmul__(self, a):
        return self.__mul__(a)
    
    def __matmul__(self, a):
        def matmul(b,c):
            return jax.numpy.tensordot(b,vec2mat(c,self.field), axes = ([1,2],[0,2]))%self.field.p
        b = object.__new__(array)
        b.field = self.field
        b.VEC = jax.vmap(matmul)(self.VEC, a.VEC)
        b.shape = b.VEC.shape[:-1]
        return b

    def lift(self):
        return vec2mat(self.VEC, self.field)

    def proj(self):
        return vec2int(self.VEC, self.field)

    def trace(self):  
        a = object.__new__(array)
        a.field = self.field
        a.VEC = jax.numpy.trace(self.VEC, axis1 = 1, axis2 = 2)%self.field.p
        a.shape = a.VEC.shape[:-1]
        return a

    def det(self):
        a = object.__new__(array)
        a.field = self.field
        a.VEC = mat2vec(fdet_vmap(vec2mat(self.VEC, self.field), self.field.INV, self.field.p, 32))
        a.shape = a.VEC.shape[:-1]
        return a

    def lu(self):
        return mgetrf_vmap(vec2mat(self.VEC, self.field).swapaxes(-2,-3).reshape((self.shape[0],self.shape[1]*self.field.n,self.shape[2]*self.field.n)), self.field.INV, 32)

    def lu_block(self):
        return fgetrf_vmap(vec2mat(self.VEC, self.field), self.field.INV, 32)

    def inv(self):
        a = object.__new__(array)
        a.field = self.field
        a.shape = self.shape
        a.VEC = mat2vec(block(invmod_vmap(vec2mat(self.VEC, self.field).swapaxes(-2,-3).reshape((self.shape[0],self.shape[1]*self.field.n,self.shape[2]*self.field.n)), self.field.INV, 32),self.field))
        return a

    def rank(self):
        return jax.numpy.count_nonzero(self.lu()[2], axis = 1)

def flatten_array(a):
    children = (a.shape, a.VEC)
    aux_data = (a.field,)
    return (children, aux_data)
def unflatten_array(aux_data, children):
    a = object.__new__(array)
    a.shape, a.VEC = children
    a.field, = aux_data
    return a
jax.tree_util.register_pytree_node(array, flatten_array, unflatten_array)

# END ARRAY
# BEGIN RANDOM

def key(s = SEED):
    return jax.random.key(s)

def random(shape, f, s = SEED): 
    SHAPE = (shape,1,1) if type(shape) == int else (shape[0],1,1) if len(shape) == 1 else (1,shape[0],shape[1]) if len(shape) == 2 else shape
    a = jax.random.randint(key(s), SHAPE+(f.n,), 0, f.p, dtype = DTYPE)
    b = object.__new__(array)
    b.field = f
    b.shape = SHAPE
    b.VEC = a
    return b

# END RANDOM
# BEGIN PLOT

def plot(a, title = '', size = 6, cmap = 'twilight_shifted'):
    matplotlib.rc('figure', figsize=(size,size))
    matplotlib.pyplot.matshow(a.reshape((-1,a.shape[-1])).T, cmap = cmap, interpolation = 'none')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()

# END PLOT
# BEGIN RINGS

@functools.partial(jax.jit, static_argnums = 0)
def Z2(q): # The finite ring Z/q x Z/q.
    Zq = jax.numpy.arange(q)
    Z2q = jax.numpy.array([jax.numpy.tile(Zq,q), jax.numpy.repeat(Zq,q)]).T
    return Z2q

@functools.partial(jax.jit, static_argnums = 0)
def M2(q): # The finite ring M_2( Z/q ).
    Z2q = Z2(q)
    M2q = jax.numpy.array([jax.numpy.tile(Z2q.T,q*q).T, jax.numpy.repeat(Z2q.T,q*q).reshape(2,-1).T]).swapaxes(0,1).reshape(-1,2,2)
    return M2q

# END RINGS
# BEGIN ENCODE/DECODE 2D

@functools.partial(jax.jit, static_argnums = 1)
def encode(a,q): # a has shape (2,2) over Zq for q < 256.
    return jax.numpy.sum(a.ravel() * q**jax.numpy.arange(4), dtype = jax.numpy.uint32)

@functools.partial(jax.jit, static_argnums = 1)
def decode(x,q): # x is nonnegative and q < 256.
    d = x//q**3
    c = (x-d*q**3)//q**2
    b = (x-d*q**3-c*q**2)//q
    a = (x-d*q**3-c*q**2-b*q)
    return jax.numpy.array([[[a,b],[c,d]]], dtype = DTYPE)

# END ENCODE/DECODE 2D
# BEGIN GROUPS

def gl2(f): # The general linear group GL_2( F ).

    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q),f)
        return mat2int((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    assert f.q < 256
    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where(m2d > 0, m2c, 0))[0]
    gl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return gl2c, idx

def sl2(f): # The special linear group SL_2( F ).

    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q),f)
        return mat2int((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    assert f.q < 256
    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where(m2d == 1, m2c, 0))[0]
    sl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return sl2c, idx

def pgl2(f): # The projective general linear group PGL_2( F ).
    assert f.q < 256

    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q),f)
        return mat2int((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    @jax.jit
    def normed(x):
        A = decode(x,f.q)[0]
        a, b = A[0,0], A[0,1]
        return (a == 1) | ((a == 0) & (b == 1))

    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    m2n = jax.vmap(normed)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where((m2d > 0) & m2n, m2c, 0))[0]
    pgl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return pgl2c, idx

def pgl2mod(q): # The projective general linear group PGL_2( Z/q ).
    assert q < 256

    @jax.jit
    def det(x):
        a = decode(x,q)[0]
        return (a[0,0]*a[1,1]-a[0,1]*a[1,0])%q

    @jax.jit
    def normed(x):
        A = decode(x,q)[0]
        a, b = A[0,0], A[0,1]
        return (a == 1) | ((a == 0) & (b == 1))

    m2c = jax.numpy.arange(q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c)
    m2n = jax.vmap(normed)(m2c)
    i = jax.numpy.nonzero(jax.numpy.where((m2d > 0) & m2n, m2c, 0))[0]
    pgl2c = m2c[i]
    idx = q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return pgl2c, idx

def psl2(f): # The projective special linear group PSL_2( F ).
    assert f.q < 256
    if f.p == 2:
        return pgl2(f)

    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q),f)
        return mat2int((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    @jax.jit
    def normed(x):
        A = decode(x,f.q)[0]
        a,b = A[0,0], A[0,1]
        return ((a != 0) & (a < f.q/2)) | ((a == 0) & (b < f.q/2))

    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    m2n = jax.vmap(normed)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where((m2d == 1) & m2n, m2c, 0))[0]
    psl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return psl2c, idx

def psl2mod(q): # The projective special linear group PSL_2( Z/q ).
    assert q < 256
    if q == 2:
        return pgl2mod(q)

    @jax.jit
    def det(x):
        a = decode(x,q)[0]
        return (a[0,0]*a[1,1]-a[0,1]*a[1,0])%q

    @jax.jit
    def normed(x):
        A = decode(x,q)[0]
        a, b = A[0,0], A[0,1]
        return ((a != 0) & (a < q/2)) | ((a == 0) & (b < q/2))

    m2c = jax.numpy.arange(q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c)
    m2n = jax.vmap(normed)(m2c)
    i = jax.numpy.nonzero(jax.numpy.where((m2d == 1) & m2n, m2c, 0))[0]
    psl2c = m2c[i]
    idx = q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return psl2c, idx

# END GROUPS
# BEGIN EXPANDERS

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

def S(p,q): # p+1 generators for the group PSL2q, if p is a quadratic residue mod q, else PGL2q.
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
        sqa = jax.numpy.astype(jax.numpy.sign((q/2)-jax.numpy.astype(a, jax.numpy.float64)), jax.numpy.int64)
        sqb = jax.numpy.astype(jax.numpy.sign((q/2)-jax.numpy.astype(b, jax.numpy.float64)), jax.numpy.int64)
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

# END EXPANDERS
# END POPPY
