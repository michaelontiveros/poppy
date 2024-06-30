# BEGIN POPPY
# BEGIN INIT

import jax
import jax.numpy as jnp
import conway_polynomials
import functools

# 64 bit integer arrays encode numbers in finite fields.
jax.config.update("jax_enable_x64", True)
DTYPE = jnp.int64

# Finite fields are polynomial rings modulo Conway polynomials.
CONWAY = conway_polynomials.database()

# The pseudo random number generator has a default seed.
SEED = 0 

# END INIT
# BEGIN MODULAR ARITHMETIC

@functools.partial(jax.jit, static_argnums = 1)
def pneg(a,p):
    return (-a)%p

@functools.partial(jax.jit, static_argnums = 2)
def padd(a, b, p):
    return (a+b)%p

@functools.partial(jax.jit, static_argnums = 2)
def psub(a, b, p):
    return (a-b)%p

@functools.partial(jax.jit, static_argnums = 2)
def pmul(a, b, p):
    return (a*b)%p

@functools.partial(jax.jit, static_argnums = 2)
def pmatmul(a, b, p):
    return (a@b)%p

pmatmul_vmap = jax.vmap(pmatmul, in_axes = (None, 0, None))

# END MODULAR ARITHMETIC
# BEGIN LINEAR ALGEBRA

@jax.jit
def ptrsm(a, b, p): # triangular solve mod p.
    
    R = jnp.arange(len(a), dtype = DTYPE) # a has shape (r,c).

    def ptrsm_vmap(bb): # bb is the matrix b.
        def ptrsm_scan(bc): # bc is a column of bb.
            def f(x,j):
                x = x.at[j].set( (bc[j] - jnp.dot( a[j], x )) % p )
                return x, x[j]  
            return jax.lax.scan( f, jnp.where( R == 0, bc[0], 0 ), R[1:] )[0] # scan the rows of a.
        return jax.vmap(ptrsm_scan)(bb.T).T  # vmap the columns of b.

    return ptrsm_vmap(b)

@jax.jit
def qtrsm(a, b, p): # triangular solve over a finite field.
    
    R = jnp.arange(len(a), dtype = DTYPE) # a has shape (r,c,n,n).
    ZERO = jnp.zeros((b.shape[-1],b.shape[-1]), dtype = jnp.int64) # b has shape (c,d,n,n).

    def ptrsm_vmap(bb): # bb is the array b.
        def ptrsm_scan(bc): # bc has shape (c,n,n). it is a column of bb.
            def f(x,j):
                x = x.at[j].set((bc[j] - jnp.tensordot(a[j], x, axes = ([0,2],[0,1]))) % p)
                return x, x[j]  
            return jax.lax.scan( f, jnp.where( R[:,None,None] == 0, bc[0], ZERO ), R[1:] )[0] # scan the rows of a.
        return jax.vmap(ptrsm_scan)(bb.swapaxes(0,1)).swapaxes(0,1)  # vmap the columns of b.

    return ptrsm_vmap(b)

@jax.jit
def pgetrf2(aperm, inv): # sequential lu decomposition mod p.
    
    p = 1+inv[-1] # p is prime.
    I = jnp.arange(aperm.shape[0])
    J = jnp.arange(aperm.shape[1]-1)
    R = jnp.arange(min(len(I),len(J)))
  
    def f(ap, i):
        j = jnp.argmax(jnp.where(I >= i, ap[:,i], -1)) # search column i for j.
        ap = ap.at[[i,j],:].set( ap[[j,i],:] ) # swap rows i and j.
        ap = ap.at[:,i].set( jnp.where( I > i, (ap[:,i] * inv[ ap[i,i] ]) % p, ap[:,i] ) )  # scale column i.
        ap = ap.at[:,:-1].set((ap[:,:-1] - jnp.where((I[:,None] > i) & (J[None,:] > i), jnp.outer(ap[:,i], ap[i,:-1]), 0)) % p) # update block D.     
        return ap, i

    return jax.lax.scan(f, aperm, R, unroll = False)[0]

@jax.jit
def qgetrf2(aperm, inv): # sequential lu decomposition over a finite field.
    
    a, perm, parity = aperm
    p = 1+inv[-1] # p is prime.
    I = jnp.arange(a.shape[0])
    J = jnp.arange(a.shape[1])
    R = jnp.arange(min(len(I),len(J)))
  
    def f(ap, i):
        a, perm, parity = ap # a has shape (r,c,n,n). perm has shape (r,). parity has shape (1,).
        #ai = a[:,i,0,0]
        ai = a[:,i,:,:].reshape((len(I),-1)).max(axis = 1)
        j = jnp.argmax(jnp.where(I >= i, ai, -1)) # search column i for j.
        perm = perm.at[[i,j],].set(perm[[j,i],]) # swap rows i and j.
        a = a.at[[i,j],:,:,:].set( a[[j,i],:,:,:] ) # swap rows i and j.
        a = a.at[:,i,:,:].set( jnp.where( I[:,None,None] > i, (jnp.tensordot( a[:,i,:,:], pinv(a[i,i,:,:], inv, 32), axes = (2,0))) % p, a[:,i,:,:] ) )  # scale column i.
        a = a.at[:,:,:,:].set((a[:,:,:,:] - jnp.where((I[:,None,None,None] > i) & (J[None,:,None,None] > i), jnp.tensordot(a[:,i,:,:], a[i,:,:,:], axes = (2,1)).swapaxes(1,2), 0)) % p) # update block D.    
        parity = (parity + jnp.count_nonzero(i-j)) % 2
        return (a, perm, parity), j

    return jax.lax.scan(f, aperm, R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def pgetrf(a, inv, b): # blocked lu decompposition mod p.
    
    p = 1+inv[-1]
    m = min(a.shape)
    perm = jnp.arange(len(a))
  
    for i in range(0, m, b):
        bb = min(m-i, b)
        ap = pgetrf2(jnp.hstack([a[i:, i:i+bb], jnp.arange(i,len(a)).reshape((-1,1))]), inv)
        perm = perm.at[i:].set(perm[ap[:,-1]])
        a = a.at[i:,:].set(a[ap[:,-1], :]) # swap rows.
        a = a.at[i:, i:i+bb].set( ap[:,:-1])  # update block C.
        a = a.at[i:i+bb, i+bb:].set(ptrsm( a[i:i+bb, i:i+bb], a[i:i+bb, i+bb:], p )) # update block B.
        a = a.at[i+bb:, i+bb:].set((a[i+bb:, i+bb:] - jax.lax.dot(a[i+bb: , i:i+bb], a[i:i+bb, i+bb:])) % p) # update block D.

    l = jnp.fill_diagonal(jnp.tril(a), 1, inplace = False)
    u = jnp.tril(a.T).T
    d = jnp.diagonal(u)
    iperm = jnp.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm

@functools.partial(jax.jit, static_argnums = 2)
def qgetrf(a, inv, b): # blocked lu decompposition over a finite field.
    
    p = 1+inv[-1]
    r,c = a.shape[0], a.shape[1]
    m = min(r,c)
    R = jnp.arange(r)
    perm = jnp.arange(r)
    parity = jnp.zeros(1, dtype = jnp.int64)
  
    for i in range(0, m, b):
        bb = min(m-i, b)
        ai, permi, pari = qgetrf2((a[i:,i:i+bb,:,:], R[i:], 0), inv) # a has shape (r-i,bb,n,n). pi has shape (r-i,)
        parity = (parity + pari) % 2
        perm = perm.at[i:].set(perm[permi])
        a = a.at[i:,:,:,:].set(a[permi,:,:,:]) # swap rows.
        a = a.at[i:,i:i+bb,:,:].set(ai)  # update block C.
        a = a.at[i:i+bb,i+bb:,:,:].set(qtrsm(a[i:i+bb,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], p)) # update block B.
        a = a.at[i+bb:,i+bb:,:,:].set((a[i+bb:,i+bb:,:,:] - jnp.tensordot(a[i+bb: ,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], axes = ([1,3],[0,2])).swapaxes(1,2)) % p) # update block D.

    I = jnp.eye(a.shape[-1], dtype = jnp.int64)
    l = jnp.where((R[:,None,None,None] - R[None,:,None,None]) > 0, a, 0)
    l = jnp.where(R[:,None,None,None] == R[None,:,None,None], I, l)
    u = jnp.where((R[:,None,None,None] - R[None,:,None,None]) <= 0, a, 0)
    d = jnp.diagonal(a, offset = 0, axis1 = 0, axis2 = 1).swapaxes(0,2).swapaxes(1,2)
    iperm = jnp.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm, parity

def pinv(a, inv, b): # matrix inverse mod p.
    
    if len(a) == 1:
        return inv[a[0,0]].reshape((1,1))

    p = 1+inv[-1]
    I = jnp.eye(len(a), dtype = DTYPE)
    l, u, d, iperm = pgetrf(a, inv, b)
    D = inv[d]
    L = ptrsm(l, I, p) # L = 1/l.
    U = ptrsm((D*u%p).T, D*I, p).T # U = 1/u.      
    return (U@L%p)[:,iperm]

@functools.partial(jax.jit, static_argnums = (2,3))
def qinv(a, inv, p, b): # matrix inverse over a finite field.
    
    if len(a) == 1:
        return pinv(a[0,0,:,:], inv, b)

    p = 1+inv[-1]
    R = jnp.arange(len(a))
    In = jnp.eye(a.shape[-1],a.shape[-1], dtype = DTYPE)
    Zn = jnp.zeros((a.shape[-1],a.shape[-1]), dtype = DTYPE)
    I = jnp.where(R[:,None,None,None] - R[None,:,None,None] == 0, In, Zn)
    l, u, d, iperm, parity = qgetrf(a, inv, b)

    def dinv(a):
        return pinv(a, inv, b)

    @jax.jit
    def matmul_vmap(aa,bb):
        return pmatmul_vmap(aa,bb,p)

    D = jax.vmap(dinv)(d)
    L = qtrsm(l, I, p) # L = 1/l.
    U = qtrsm(jax.vmap(matmul_vmap)(D,u).swapaxes(0,1), jax.vmap(matmul_vmap)(D,I), p).swapaxes(0,1) # U = 1/u.      
    return (jnp.tensordot(U, L, axes = ([1,3],[0,2])) % p)[:,iperm,:,:]

def qdet(a, inv, p, b): # matrix determinant over a finite field.
    l, u, d, iperm, parity = qgetrf(a, inv, b)
    def matmul(a,b):
        return pmatmul(a,b,p)
    return jnp.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p

# END LINEAR ALGEBRA
# BEGIN FIELD

class field:
    def __init__(self, p, n):    
        
        self.p = p
        self.n = n
        self.q = p ** n if n*jnp.log2(p) < 63 else None
        self.CONWAY = CONWAY[p][n]
        self.RANGE = jnp.arange(n, dtype = DTYPE)
        self.ONE = jnp.ones(  n, dtype = DTYPE)
        self.I = jnp.eye(   n, dtype = DTYPE)
        self.BASIS = jnp.power( p * self.ONE, self.RANGE )
        self.X = self.x() # powers of the Conway matrix.
        self.INV = self.inv() # multiplicative inverses mod p.
        self.LEG = self.leg() # legendre symbol mod p.

    def __repr__(self):
        return f'field order  {self.q}.'
        
    def x(self):

        @jax.jit
        def id(a,i):
            return a

        stack = jax.vmap(id, (None,0))
        
        @jax.jit
        def neg(a):
            return pneg(a, self.p)
        
        @jax.jit
        def matmul(a,b):
            return pmatmul(a, b, self.p)
        
        @jax.jit
        def x_scan():
            
            # V is the vector of subleading coefficients of the Conway polynomial.
            V = jnp.array(self.CONWAY[:-1], dtype = DTYPE)
            
            # M is the companion matrix of the Conway polynomial.
            M = jnp.zeros((self.n,self.n), dtype = DTYPE).at[:-1,1:].set(self.I[1:,1:]).at[-1].set(neg(V))
            
            # X is the array of powers of the companion matrix.
            X = jax.lax.associative_scan(matmul, stack(M,self.RANGE).at[0].set(self.I))
            
            return X

        return x_scan()
    
    def inv(self):
        
        @jax.jit
        def mul(a,b):
            return pmul(a, b, self.p)
        
        @functools.partial(jax.jit, static_argnums = 1)
        def inv_jit(ABC, i):
            
            C = mul(ABC[i-2, 0], ABC[i-2, 2])
            ABC = ABC.at[i-1, 2].set(C)   
            return ABC, mul(ABC[i-1, 1], C)
        
        @jax.jit
        def inv_scan():    
            
            A = jnp.arange(1, self.p, dtype = DTYPE)
            AA = jnp.concatenate([self.ONE[:1], jnp.flip(A[1:])])
            B = jnp.flip(jax.lax.associative_scan(mul,AA))
            C = jnp.ones(self.p - 1, dtype = DTYPE).at[0].set(self.p - 1)
            ABC = jnp.vstack([A,B,C]).T       
            return jnp.concatenate([jnp.zeros(1, dtype = DTYPE), jax.lax.scan(inv_jit, ABC, A)[1]])
        
        return inv_scan()

    def leg(self):
        R = jnp.arange(self.p)
        return (-jnp.ones(self.p, dtype = jnp.int32)).at[(R*R) % self.p].set(1).at[0].set(0)

# END FIELD
# BEGIN RESHAPE OPERATIONS

@functools.partial(jax.jit, static_argnums = 1)
def block(m,f): 
    s = m.shape
    n = f.n 
    return m.reshape(s[:-2] + (s[-2]//n, n, s[-1]//n, n)).swapaxes(-2, -3)

@functools.partial(jax.jit, static_argnums = 1)
def ravel(m,f): 
    n = f.n
    return block(m,f).reshape((-1, n, n))

@functools.partial(jax.jit, static_argnums = (1,2))
def unravel(i, f, s):   
    n = f.n    
    return i.reshape(s[:-2] + (s[-2]//n, s[-1]//n, n, n)).swapaxes(-2, -3).reshape(s)

# END RESHAPE OPERATIONS
# BEGIN EN/DE-CODING OPERATIONS

@functools.partial(jax.jit, static_argnums = 1)
def i2v(i,f):
    return jnp.floor_divide(i * f.ONE, f.BASIS) % f.p

@functools.partial(jax.jit, static_argnums = 1)
def v2i(v,f):
    return jnp.sum(v * f.BASIS, dtype = DTYPE)

@functools.partial(jax.jit, static_argnums = 1)
def v2m(v,f):
    return jnp.dot(v, f.X) % f.p

@functools.partial(jax.jit, static_argnums = 1)
def m2v(m,f):
    return m[0]

@functools.partial(jax.jit, static_argnums = 1)
def i2m(i,f):
    return v2m(i2v(i,f), f)

@functools.partial( jax.jit, static_argnums = 1 )
def m2i(m,f):
    return v2i(m[0], f)

i2v_vmap = jax.vmap(i2v, in_axes = (0, None))
v2i_vmap = jax.vmap(v2i, in_axes = (0, None))
v2m_vmap = jax.vmap(v2m, in_axes = (0, None))
m2v_vmap = jax.vmap(m2v, in_axes = (0, None))
i2m_vmap = jax.vmap(i2m, in_axes = (0, None))
m2i_vmap = jax.vmap(m2i, in_axes = (0, None))

@functools.partial(jax.jit, static_argnums = 1)
def lift(i,f):  
    m = i2m_vmap(i.ravel(), f)
    s = (1, 1, i.shape) if len(i.shape) == 1 else (1, i.shape[0], i.shape[1]) if len(i.shape) == 2 else i.shape
    #s = i.shape
    n = f.n 
    return m.reshape(s + (n,n)).swapaxes(-2, -3).reshape(s[:-2] + (s[-2]*n, s[-1]*n))

@functools.partial(jax.jit, static_argnums = 1)
def proj(m,f):      
    i = m2i_vmap(ravel(m,f), f)
    s = m.shape[:-2] + (m.shape[-2]//f.n, m.shape[-1]//f.n)  
    return i.reshape(s)

# END EN/DE-CODING OPERATIONS
# BEGIN ARRAY

class array:
    def __init__(self, a, dtype = field(2,1), lifted = False):
    
        if len(a.shape) > 3:
            print('ERROR: poppy arrays are three dimensional')
            return

        self.field = dtype 
        self.shape = (1, 1, a.shape[0]) if len(a.shape) == 1 else (1, a.shape[0], a.shape[1]) if len(a.shape) == 2 else a.shape
        self.shape = (self.shape[-3], self.shape[-2]//self.field.n, self.shape[-1]//self.field.n) if lifted else self.shape
        self.lift = a if lifted else lift(a, self.field)

    def __repr__(self):
        return f'array shape {self.shape}.\n' + repr(self.field) 

    def __neg__(self):
        return array(pneg(self.lift, self.field.p), dtype = self.field, lifted = True)
    
    def __add__(self, a):
        return array(padd(self.lift, a.lift, self.field.p), dtype = self.field, lifted = True)
    
    def __sub__(self, a):
        return array(psub(self.lift, a.lift, self.field.p), dtype = self.field, lifted = True)

    def __mul__(self, a):    
        if self.shape[-1]*self.shape[-2] == 1:      
            b = pmatmul_vmap(self.lift, ravel(a.lift, self.field), self.field.p).reshape(a.lift.shape)        
            return array(b, dtype = self.field, lifted = True)      
        if a.shape[-1]*a.shape[-2] == 1:         
            b = pmatmul_vmap(a.lift, ravel(self.lift, self.field), self.field.p).reshape(self.lift.shape)                     
            return array(b, dtype = self.field, lifted = True)
    
    def __matmul__(self, a):
        return array(pmatmul(self.lift, a.lift, self.field.p), dtype = self.field, lifted = True)

    def proj(self):
        return proj(self.lift, self.field)

    def trace(self):  
        T = jnp.trace(block(self.lift, self.field)) % self.field.p     
        return array(T, dtype = self.field, lifted = True)

    def det(self):    
        return array(qdet(block(self.lift, self.field), self.field.INV, self.field.p, 32), dtype = self.field, lifted = True)

    def lu(self):
        return pgetrf(self.lift, self.field.INV, 32)

    def lu_block(self):
        return qgetrf(block(self.lift, self.field), self.field.INV, 32)

    def inv(self):    
        return array(pinv(self.lift, self.field.INV, 32), dtype = self.field, lifted = True)

    def inv_block(self):    
        return qinv(block(self.lift, self.field), self.field.INV, self.field.p, 32)
    
    def inv_scan(self):   
        
        assert len(self.shape) > 1
        assert self.shape[-1] == self.shape[-2]
        
        @functools.partial(jax.jit, static_argnums = 1)
        def row_reduce_jit(a,j):     
            
            mask = jnp.where(jnp.arange(self.lift.shape[0], dtype = DTYPE) < j, 0, 1)
            i = jnp.argmax(mask*a[:,j] != 0)
            
            if a.at[i,j] == 0:
                return a, a[j]
        
            Rj, Ri = a[j], (a[i]*self.field.INV[a[i,j]]) % self.field.p
            a = a.at[i].set(Rj).at[j].set(Ri)
            A = (a - jnp.outer(a[:,j].at[j].set(0), a[j])) % self.field.p         
            return A, A[j]
        
        @jax.jit
        def inv_jit():          
            
            N = self.lift.shape[0]
            I = jnp.eye(N, dtype = DTYPE)
            MI = jnp.hstack([self.lift, I])
            RANGE = jnp.arange(N, dtype = DTYPE)          
            return jax.lax.scan(row_reduce_jit, MI, RANGE)[0][:, N:]
        
        INV = inv_jit()       
        return array(INV, dtype = self.field, lifted = True)

# END ARRAY
# BEGIN RANDOM

SEED = 0

def key(s = SEED):
    return jax.random.PRNGKey(s)

def random(shape, f, s = SEED): 
    SHAPE = (1, 1, shape) if type(shape) == int else shape
    MAX = f.q if f.n*jnp.log2(f.p) < 63 else jnp.iinfo(jnp.int64).max
    a = jax.random.randint(key(s), SHAPE, 0, MAX, dtype = DTYPE)
    return array(a,f)

# END RANDOM
# BEGIN 2D OBJECTS

@functools.partial(jax.jit, static_argnums = 0)
def v2(q):
  R1 = jnp.arange(q)
  R2 = jnp.array( [ jnp.tile(R1,len(R1)), jnp.repeat(R1,len(R1))]).T
  return R2 # all 2d vectors with nonnegative integer entries less than q.

@functools.partial(jax.jit, static_argnums = 0)
def m2(q):
  R1 = jnp.arange(q)
  R2 = jnp.array( [ jnp.tile(R1,len(R1)), jnp.repeat(R1,len(R1))]).T
  R3 = jnp.array( [ jnp.tile(R2.T,len(R2)).T, jnp.repeat(R2.T,len(R2)).reshape(2,-1).T]).swapaxes(0,1).reshape(-1,2,2)
  return R3 # all 2x2 matrices with nonnegative integer entries less than q.

@functools.partial(jax.jit, static_argnums = 1)
def encode(a,q): # q < 256.
  return jnp.sum(a.ravel() * q**jnp.arange(4), dtype = jnp.uint32)

@functools.partial(jax.jit, static_argnums = 1)
def decode(x,q): # q < 256.
    d = x//q**3
    c = (x-d*q**3)//q**2
    b = (x-d*q**3-c*q**2)//q
    a = (x-d*q**3-c*q**2-b*q)
    return jnp.array([[a,b],[c,d]], dtype = jnp.int64)

def gl2(f):
  assert f.q < 256

  @jax.jit
  def det(x):
    a = lift(decode(x,f.q),f)
    return proj((a[0,0]@a[1,1]-a[0,1]@a[1,0])%f.p,f).ravel()

  m2c = jnp.arange(f.q**4, dtype = jnp.uint32)
  m2d = jax.vmap(det)(m2c.reshape((-1,1,1))).squeeze()
  i = jnp.nonzero(jnp.where(m2d > 0, m2c, 0))[0]
  gl2c = m2c[i]
  id = f.q**4*jnp.ones(len(m2c), dtype = jnp.uint32)
  id = id.at[i].set(jnp.arange(len(i), dtype = jnp.uint32))
  return gl2c, id

def sl2(f):
  assert f.q < 256

  @jax.jit
  def det(x):
    a = lift(decode(x,f.q),f)
    return proj((a[0,0]@a[1,1]-a[0,1]@a[1,0])%f.p,f).ravel()

  m2c = jnp.arange(f.q**4, dtype = jnp.uint32)
  m2d = jax.vmap(det)(m2c.reshape((-1,1,1))).squeeze()
  i = jnp.nonzero(jnp.where(m2d == 1, m2c, 0))[0]
  sl2c = m2c[i]
  id = f.q**4*jnp.ones(len(m2c), dtype = jnp.uint32)
  id = id.at[i].set(jnp.arange(len(i), dtype = jnp.uint32))
  return sl2c, id

def pgl2(f):
  assert f.q < 256

  @jax.jit
  def det(x):
    a = lift(decode(x,f.q),f)
    return proj((a[0,0]@a[1,1]-a[0,1]@a[1,0])%f.p,f).ravel()

  @jax.jit
  def normed(x):
    A = lift(decode(x,f.q),f)
    a, b = proj(A[0,0],f).ravel(), proj(A[0,1],f).ravel()
    return (a == 1) | ((a == 0) & (b == 1))

  m2c = jnp.arange(f.q**4, dtype = jnp.uint32)
  m2d = jax.vmap(det)(m2c.reshape((-1,1,1))).squeeze()
  m2n = jax.vmap(normed)(m2c.reshape((-1,1,1))).squeeze()
  i = jnp.nonzero(jnp.where((m2d > 0) & m2n, m2c, 0))[0]
  pgl2c = m2c[i]
  id = f.q**4*jnp.ones(len(m2c), dtype = jnp.uint32)
  id = id.at[i].set(jnp.arange(len(i), dtype = jnp.uint32))
  return pgl2c, id

def pgl2mod(q):
  assert q < 256

  @jax.jit
  def det(x):
    a = decode(x,q)
    return (a[0,0]*a[1,1]-a[0,1]*a[1,0])%q

  @jax.jit
  def normed(x):
    A = decode(x,q)
    a, b = A[0,0], A[0,1]
    return (a == 1) | ((a == 0) & (b == 1))

  m2c = jnp.arange(q**4, dtype = jnp.uint32)
  m2d = jax.vmap(det)(m2c)
  m2n = jax.vmap(normed)(m2c)
  i = jnp.nonzero(jnp.where((m2d > 0) & m2n, m2c, 0))[0]
  pgl2c = m2c[i]
  id = q**4*jnp.ones(len(m2c), dtype = jnp.uint32)
  id = id.at[i].set(jnp.arange(len(i), dtype = jnp.uint32))
  return pgl2c, id

def psl2(f):
  assert f.q < 256

  @jax.jit
  def det(x):
    a = lift(decode(x,f.q),f)
    return proj((a[0,0]@a[1,1]-a[0,1]@a[1,0])%f.p,f).ravel()

  @jax.jit
  def normed(x):
    A = lift(decode(x,f.q),f)
    a, b = proj(A[0,0],f).ravel(), proj(A[0,1],f).ravel()
    return (a < f.q//2) | ((a == 0) & (b < f.q//2))

  m2c = jnp.arange(f.q**4, dtype = jnp.uint32)
  m2d = jax.vmap(det)(m2c.reshape((-1,1,1))).squeeze()
  m2n = jax.vmap(normed)(m2c.reshape((-1,1,1))).squeeze()
  i = jnp.nonzero(jnp.where((m2d == 1) & m2n, m2c, 0))[0]
  psl2c = m2c[i]
  id = f.q**4*jnp.ones(len(m2c), dtype = jnp.uint32)
  id = id.at[i].set(jnp.arange(len(i), dtype = jnp.uint32))
  return psl2c, id

def psl2mod(q):
  assert q < 256

  @jax.jit
  def det(x):
    a = decode(x,q)
    return (a[0,0]*a[1,1]-a[0,1]*a[1,0])%q

  @jax.jit
  def normed(x):
    A = decode(x,q)
    a, b = A[0,0], A[0,1]
    return (a < q//2) | ((a == 0) & (b < q//2))

  m2c = jnp.arange(q**4, dtype = jnp.uint32)
  m2d = jax.vmap(det)(m2c)
  m2n = jax.vmap(normed)(m2c)
  i = jnp.nonzero(jnp.where((m2d == 1) & m2n, m2c, 0))[0]
  psl2c = m2c[i]
  id = q**4*jnp.ones(len(m2c), dtype = jnp.uint32)
  id = id.at[i].set(jnp.arange(len(i), dtype = jnp.uint32))
  return psl2c, id

@functools.partial(jax.jit, static_argnums = 0)
def pS1(p):

  def norm(a):
    return a[0]*a[0]+a[1]*a[1]

  R = v2(p)
  N = jax.vmap(norm)(R)%p
  M = jnp.where(N == (p-1), N, -1)
  return R[jnp.argmax(M)] # a point on the circle x*x + y*y = (p-1) mod p.

@functools.partial(jax.jit, static_argnums = (0,1))
def S3(p,r):

  def norm(a):
    return a[0,0]*a[0,0]+a[0,1]*a[0,1]+a[1,0]*a[1,0]+a[1,1]*a[1,1]
    
  R = m2(2*r)-r
  N = jax.vmap(norm)(R)
  return R, jnp.where(N == p,N,-1) # the integer three sphere x*x + y*y + z*z + t*t = p.

def S(p,q):
  xy = pS1(q)
  x, y = xy[0], xy[1]

  def f(s):
    a, b, c, d = s[0,0], s[0,1], s[1,0], s[1,1]
    return jnp.array([[a+b*x+d*y, c+d*x-b*y],[-c+d*x-b*y, a-b*x-d*y]], dtype = jnp.int64)%q

  R, i = S3(p,1+int(jnp.sqrt(p)))
  i = jnp.where(i >= 0)
  R0 = R[i,0,0]
  R1 = R[i,0,1]
  Sp = R[i[0][jnp.where((R0 > 0) & ((R0%2)==1))[1]]] if (p % 4) == 1 else R[i[0][jnp.where((R0 >= 0) & ((R0%2) == 0) & ((R0 > 0) | (R1 > 0)))[1]]]
  return jax.vmap(f)(Sp) # generate the group PSL2q if p is a quadratic residue mod q, else PGL2q.

def lps(p,q): # the Lubotzky-Phillips-Sarnak expander graph is a (p+1)-regular Cayley graph for the group PSL2q or PGL2q.
  assert (p in CONWAY) & (q in CONWAY) & (p != q) & (p > 2) & (q > 2) & (q*q > 4*p)
  f = field(q,1)

  @jax.jit
  def normpgl(A):
    a, b = A[0,0], A[0,1]
    sa = jnp.sign(a)
    c = f.INV[sa*a + (1-sa)*b]
    return c*A%q

  @jax.jit
  def normpsl(A):
    a, b = A[0,0], A[0,1]
    sa, sqa, sqb = jnp.sign(a), jnp.sign(q//2 - a), jnp.sign(q//2 - b)
    s = sa*sqa + (1-sa)*sqb
    return s*A%q

  norm = normpsl if f.LEG[p] == 1 else normpgl
  V = jax.vmap(norm)(S(p,q))

  @jax.jit
  def enc(a):
    return jnp.sum(a.ravel() * q**jnp.arange(4), dtype = jnp.uint32)

  @jax.jit
  def dec(x):
    d = x//q**3
    c = (x-d*q**3)//q**2
    b = (x-d*q**3-c*q**2)//q
    a = (x-d*q**3-c*q**2-b*q)
    return jnp.array([[a,b],[c,d]], dtype = jnp.int64)

  @jax.jit 
  def mul(x):
    a = jax.vmap(norm)(jnp.tensordot(dec(x), V, axes = (1,1)).swapaxes(0,1)%q)
    return jax.vmap(enc)(a)

  G, id = psl2mod(q) if f.LEG[p] == 1 else pgl2mod(q)
  graph = jax.vmap(mul)(G)
  return graph, id

# END 2D OBJECTS
# END POPPY