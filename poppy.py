# BEGIN POPPY
# BEGIN INIT

import jax
import conway_polynomials
import functools

# 64 bit integer arrays encode numbers in finite fields.
jax.config.update("jax_enable_x64", True)
DTYPE = jax.numpy.int64

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
def padd(a,b,p):
    return (a+b)%p

@functools.partial(jax.jit, static_argnums = 2)
def psub(a,b,p):
    return (a-b)%p

@functools.partial(jax.jit, static_argnums = 2)
def pmul(a,b,p):
    return (a*b)%p

@functools.partial(jax.jit, static_argnums = 2)
def pmatmul(a,b,p):
    return (a@b)%p

# END MODULAR ARITHMETIC
# BEGIN LINEAR ALGEBRA

@jax.jit
def ptrsm(a,b,p): # Triangular solve mod p.
    
    R = jax.numpy.arange(len(a), dtype = DTYPE) # a has shape (r,c).

    def ptrsm_vmap(bb): # bb is the matrix b.
        def ptrsm_scan(bc): # bc is a column of bb.
            def f(x,j):
                x = x.at[j].set((bc[j] - jax.numpy.dot(a[j], x)) % p)
                return x, x[j]  
            return jax.lax.scan(f, jax.numpy.where(R == 0, bc[0], 0), R[1:])[0] # scan the rows of a.
        return jax.vmap(ptrsm_scan)(bb.T).T  # vmap the columns of b.

    return ptrsm_vmap(b)

@jax.jit
def qtrsm(a,b,p): # Triangular solve over a finite field.
    
    R = jax.numpy.arange(len(a), dtype = DTYPE) # a has shape (r,c,n,n).
    ZERO = jax.numpy.zeros((b.shape[-1],b.shape[-1]), dtype = DTYPE) # b has shape (c,d,n,n).

    def ptrsm_vmap(bb): # bb is the array b.
        def ptrsm_scan(bc): # bc has shape (c,n,n). it is a column of bb.
            def f(x,j):
                x = x.at[j].set((bc[j] - jax.numpy.tensordot(a[j], x, axes = ([0,2],[0,1]))) % p)
                return x, x[j]  
            return jax.lax.scan( f, jax.numpy.where( R[:,None,None] == 0, bc[0], ZERO ), R[1:] )[0] # scan the rows of a.
        return jax.vmap(ptrsm_scan)(bb.swapaxes(0,1)).swapaxes(0,1)  # vmap the columns of b.

    return ptrsm_vmap(b)

@jax.jit
def pgetrf2(aperm, inv): # Sequential lu decomposition mod p.
    
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

@jax.jit
def qgetrf2(aperm, inv): # Sequential lu decomposition over a finite field.
    
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
        a = a.at[:,i,:,:].set(jax.numpy.where(I[:,None,None] > i, (jax.numpy.tensordot(a[:,i,:,:], pinv(a[i,i,:,:], inv, 32), axes = (2,0))) % p, a[:,i,:,:])) # Scale column i.
        a = a.at[:,:,:,:].set((a[:,:,:,:] - jax.numpy.where((I[:,None,None,None] > i) & (J[None,:,None,None] > i), jax.numpy.tensordot(a[:,i,:,:], a[i,:,:,:], axes = (2,1)).swapaxes(1,2), 0)) % p) # Update block D.    
        parity = (parity + jax.numpy.count_nonzero(i-j)) % 2
        return (a, perm, parity), j

    return jax.lax.scan(f, aperm, R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def pgetrf(a, inv, b): # Blocked lu decompposition mod p.
    
    p = 1+inv[-1]
    m = min(a.shape)
    perm = jax.numpy.arange(len(a))
  
    for i in range(0, m, b):
        bb = min(m-i, b)
        ap = pgetrf2(jax.numpy.hstack([a[i:, i:i+bb], jax.numpy.arange(i,len(a)).reshape((-1,1))]), inv)
        perm = perm.at[i:].set(perm[ap[:,-1]])
        a = a.at[i:,:].set(a[ap[:,-1], :]) # Swap rows.
        a = a.at[i:, i:i+bb].set( ap[:,:-1])  # Update block C.
        a = a.at[i:i+bb, i+bb:].set(ptrsm( a[i:i+bb, i:i+bb], a[i:i+bb, i+bb:], p )) # Update block B.
        a = a.at[i+bb:, i+bb:].set((a[i+bb:, i+bb:] - jax.lax.dot(a[i+bb: , i:i+bb], a[i:i+bb, i+bb:])) % p) # Update block D.

    l = jax.numpy.fill_diagonal(jax.numpy.tril(a), 1, inplace = False)
    u = jax.numpy.tril(a.T).T
    d = jax.numpy.diagonal(u)
    iperm = jax.numpy.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm

pgetrf_vmap = jax.vmap(pgetrf, in_axes = (0, None, None))

@functools.partial(jax.jit, static_argnums = 2)
def qgetrf(a, inv, b): # Blocked lu decompposition over a finite field.
    
    p = 1+inv[-1]
    r,c = a.shape[0], a.shape[1]
    m = min(r,c)
    R = jax.numpy.arange(r)
    perm = jax.numpy.arange(r)
    parity = jax.numpy.zeros(1, dtype = DTYPE)
  
    for i in range(0, m, b):
        bb = min(m-i, b)
        ai, permi, pari = qgetrf2((a[i:,i:i+bb,:,:], R[i:], 0), inv) # a has shape (r-i,bb,n,n). pi has shape (r-i,)
        parity = (parity + pari) % 2
        perm = perm.at[i:].set(perm[permi])
        a = a.at[i:,:,:,:].set(a[permi,:,:,:]) # Swap rows.
        a = a.at[i:,i:i+bb,:,:].set(ai)  # Update block C.
        a = a.at[i:i+bb,i+bb:,:,:].set(qtrsm(a[i:i+bb,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], p)) # Update block B.
        a = a.at[i+bb:,i+bb:,:,:].set((a[i+bb:,i+bb:,:,:] - jax.numpy.tensordot(a[i+bb: ,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], axes = ([1,3],[0,2])).swapaxes(1,2)) % p) # Update block D.

    I = jax.numpy.eye(a.shape[-1], dtype = DTYPE)
    l = jax.numpy.where((R[:,None,None,None] - R[None,:,None,None]) > 0, a, 0)
    l = jax.numpy.where(R[:,None,None,None] == R[None,:,None,None], I, l)
    u = jax.numpy.where((R[:,None,None,None] - R[None,:,None,None]) <= 0, a, 0)
    d = jax.numpy.diagonal(a, offset = 0, axis1 = 0, axis2 = 1).swapaxes(0,2).swapaxes(1,2)
    iperm = jax.numpy.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm, parity

qgetrf_vmap = jax.vmap(qgetrf, in_axes = (0, None, None))

def pinv(a, inv, b): # Matrix inverse mod p.
    
    if a.shape[1] == 1:
        return inv[a[:,0,0]].reshape((a.shape[0],1,1))

    p = 1+inv[-1]
    I = jax.numpy.eye(a.shape[1], dtype = DTYPE)
   
    def inverse(A):
        l, u, d, iperm = pgetrf(A, inv, b)
        D = inv[d]
        L = ptrsm(l, I, p) # L = 1/l.
        U = ptrsm((D*u%p).T, D*I, p).T # U = 1/u.      
        return (U@L%p)[:,iperm]
    
    return jax.vmap(inverse)(a)

def qdet(a, inv, p, b): # Matrix determinant over a finite field.
    l, u, d, iperm, parity = qgetrf(a, inv, b)
    def matmul(a,b):
        return pmatmul(a,b,p)
    return jax.numpy.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p

# END LINEAR ALGEBRA
# BEGIN FIELD

class field:
    def __init__(self, p, n):    
        
        self.p = p
        self.n = n
        self.q = p ** n if n*jax.numpy.log2(p) < 63 else None
        self.CONWAY = CONWAY[p][n]
        self.RANGE = jax.numpy.arange(n, dtype = DTYPE)
        self.ONE = jax.numpy.ones(  n, dtype = DTYPE)
        self.I = jax.numpy.eye(   n, dtype = DTYPE)
        self.BASIS = jax.numpy.power( p * self.ONE, self.RANGE )
        self.X = self.x()     # Powers of the Conway matrix.
        self.INV = self.inv() # Multiplicative inverse mod p.
        self.LEG = self.leg() # Legendre symbol mod p.

    def __repr__(self):
        return f'the field of order {self.q}'
        
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
            V = jax.numpy.array(self.CONWAY[:-1], dtype = DTYPE)
            
            # M is the companion matrix of the Conway polynomial.
            M = jax.numpy.zeros((self.n,self.n), dtype = DTYPE).at[:-1,1:].set(self.I[1:,1:]).at[-1].set(neg(V))
            
            # X is the array of powers of M.
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
            
            A = jax.numpy.arange(1, self.p, dtype = DTYPE)
            AA = jax.numpy.concatenate([self.ONE[:1], jax.numpy.flip(A[1:])])
            B = jax.numpy.flip(jax.lax.associative_scan(mul,AA))
            C = jax.numpy.ones(self.p - 1, dtype = DTYPE).at[0].set(self.p - 1)
            ABC = jax.numpy.vstack([A,B,C]).T       
            return jax.numpy.concatenate([jax.numpy.zeros(1, dtype = DTYPE), jax.lax.scan(inv_jit, ABC, A)[1]])
        
        return inv_scan()

    def leg(self):
        R = jax.numpy.arange(self.p)
        return (-jax.numpy.ones(self.p, dtype = DTYPE)).at[(R*R) % self.p].set(1).at[0].set(0)

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
def i2v(i,f):
    return jax.numpy.floor_divide(i*f.ONE, f.BASIS) % f.p

@functools.partial(jax.jit, static_argnums = 1)
def v2i(v,f):
    return jax.numpy.sum(v*f.BASIS, dtype = DTYPE)

@functools.partial(jax.jit, static_argnums = 1)
def v2m(v,f):
    return jax.numpy.dot(v, f.X) % f.p

@functools.partial(jax.jit, static_argnums = 1)
def m2v(m,f):
    return m[0]

@functools.partial(jax.jit, static_argnums = 1)
def i2m(i,f):
    return v2m(i2v(i,f), f)

@functools.partial( jax.jit, static_argnums = 1 )
def m2i(m,f):
    return v2i(m[0], f)

i2m_vmap = jax.vmap(i2m, in_axes = (0, None))
m2i_vmap = jax.vmap(m2i, in_axes = (0, None))

#@functools.partial(jax.jit, static_argnums = 1)
def lift(i,f):  
    if type(i) == int:
        i = jax.numpy.array(i, dtype = DTYPE).reshape((1,1,1))
    m = i2m_vmap(i.ravel(), f)
    s = (i.shape[0], 1, 1) if len(i.shape) == 1 else (1, i.shape[0], i.shape[1]) if len(i.shape) == 2 else i.shape
    n = f.n 
    return m.reshape(s + (n,n)).swapaxes(-2, -3).reshape(s[:-2] + (s[-2]*n, s[-1]*n))

@functools.partial(jax.jit, static_argnums = 1)
def proj(a,f):      
    i = jax.vmap(m2i_vmap, in_axes = (0, None))(ravel(a,f), f)
    s = a.shape[:-2] + (a.shape[-2]//f.n, a.shape[-1]//f.n)  
    return i.reshape(s)

# END LIFT/PROJECT
# BEGIN ARRAY

class array:
    def __init__(self, a, dtype = field(2,1), lifted = False):
    
        if type(a) == int:
            a = jax.numpy.array([a], dtype = DTYPE)
        elif type(a) == list:
            a = jax.numpy.array(a, dtype = DTYPE)
        if len(a.shape) > 3:
            print('ERROR: poppy arrays are three dimensional.')
            return

        self.field = dtype 
        self.shape = (a.shape[0], 1, 1) if len(a.shape) == 1 else (1, a.shape[0], a.shape[1]) if len(a.shape) == 2 else a.shape
        self.shape = (self.shape[0], self.shape[1] // self.field.n, self.shape[2] // self.field.n) if lifted else self.shape
        self.REP = a if lifted else lift(a, self.field)

    def __repr__(self):
        return f'shape {self.shape[0]} {self.shape[1]} {self.shape[2]} over ' + repr(self.field) 

    def __neg__(self):
        return array(pneg(self.REP, self.field.p), dtype = self.field, lifted = True)
    
    def __add__(self, a):
        if type(a) == int:
            b = jax.vmap(padd, in_axes = (0,None,None))(ravel(self.REP, self.field), lift(a, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if a.shape == self.shape:
            return array(padd(self.REP, a.REP, self.field.p), dtype = self.field, lifted = True)
        if a.shape[0] == 1:
            if a.shape[-1]*a.shape[-2] == 1:
                b = jax.vmap(padd, in_axes = (0,None,None))(ravel(self.REP, self.field), a.REP, self.field.p)
                return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
            b = jax.vmap(jax.vmap(padd, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if self.shape[0] == 1:
            if self.shape[-1]*self.shape[-2] == 1:
                b = jax.vmap(padd, in_axes = (0,None,None))(ravel(a.REP, self.field), self.REP, self.field.p)
                return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)
            b = jax.vmap(jax.vmap(padd, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(a.REP, self.field), ravel(self.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if a.shape[-1]*a.shape[-2] == 1:
            b = jax.vmap(jax.vmap(padd, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        b = jax.vmap(jax.vmap(padd, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(a.REP, self.field), self.REP, self.field.p)
        return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)

    def __radd__(self, a):
        if type(a) == int:
            b = jax.vmap(padd, in_axes = (0,None,None))(ravel(self.REP, self.field), lift(a, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
 
    def __sub__(self, a):
        if type(a) == int:
            b = jax.vmap(psub, in_axes = (0,None,None))(ravel(self.REP, self.field), lift(a, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if a.shape == self.shape:
            return array(psub(self.REP, a.REP, self.field.p), dtype = self.field, lifted = True)
        if a.shape[0] == 1:
            if a.shape[-1]*a.shape[-2] == 1:
                b = jax.vmap(psub, in_axes = (0,None,None))(ravel(self.REP, self.field), a.REP, self.field.p)
                return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
            b = jax.vmap(jax.vmap(psub, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if self.shape[0] == 1:
            if self.shape[-1]*self.shape[-2] == 1:
                b = jax.vmap(psub, in_axes = (None,0,None))(self.REP, ravel(a.REP, self.field), self.field.p)
                return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)
            b = jax.vmap(jax.vmap(psub, in_axes = (None,0,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if a.shape[-1]*a.shape[-2] == 1:
            b = jax.vmap(jax.vmap(psub, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        b = jax.vmap(jax.vmap(psub, in_axes = (None,0,None)), in_axes = (0,0,None))(self.REP, ravel(a.REP, self.field), self.field.p)
        return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)

    def __rsub__(self, a):
        if type(a) == int:
            b = jax.vmap(psub, in_axes = (None,0,None))(lift(a, self.field), ravel(self.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)

    def __mul__(self, a):
        if type(a) == int:
            b = jax.vmap(pmatmul, in_axes = (0,None,None))(ravel(self.REP, self.field), lift(a, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if a.shape == self.shape:
            b = jax.vmap(pmatmul, in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)
        if a.shape[0] == 1:
            if a.shape[-1]*a.shape[-2] == 1:
                b = jax.vmap(pmatmul, in_axes = (0,None,None))(ravel(self.REP, self.field), a.REP, self.field.p)
                return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
            b = jax.vmap(jax.vmap(pmatmul, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if self.shape[0] == 1:
            if self.shape[-1]*self.shape[-2] == 1:
                b = jax.vmap(pmatmul, in_axes = (0,None,None))(ravel(a.REP, self.field), self.REP, self.field.p)
                return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)
            b = jax.vmap(jax.vmap(pmatmul, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(a.REP, self.field), ravel(self.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        if a.shape[-1]*a.shape[-2] == 1:
            b = jax.vmap(jax.vmap(pmatmul, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(self.REP, self.field), ravel(a.REP, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
        b = jax.vmap(jax.vmap(pmatmul, in_axes = (0,None,None)), in_axes = (0,0,None))(ravel(a.REP, self.field), self.REP, self.field.p)
        return array(unravel(b, self.field, a.REP.shape), dtype = self.field, lifted = True)

    def __rmul__(self, a):
        if type(a) == int:
            b = jax.vmap(pmatmul, in_axes = (0,None,None))(ravel(self.REP, self.field), lift(a, self.field), self.field.p)
            return array(unravel(b, self.field, self.REP.shape), dtype = self.field, lifted = True)
    
    def __matmul__(self, a):
        return array(pmatmul(self.REP, a.REP, self.field.p), dtype = self.field, lifted = True)

    def lift(self):
        return self.REP

    def proj(self):
        return proj(self.REP, self.field)

    def trace(self):  
        return array(jax.numpy.trace(block(self.REP, self.field), axis1 = 1, axis2 = 2) % self.field.p, dtype = self.field, lifted = True)

    def det(self):
        b = block(self.REP, self.field)
        def d(i):
            return qdet(b[i], self.field.INV, self.field.p, 32)
        return array(jax.vmap(d)(jax.numpy.arange(self.shape[0])), dtype = self.field, lifted = True)

    def lu(self):
        return pgetrf_vmap(self.REP, self.field.INV, 32)

    def lu_block(self):
        return qgetrf_vmap(block(self.REP, self.field), self.field.INV, 32)

    def inv(self):
        return array(pinv(self.REP, self.field.INV, 32), dtype = self.field, lifted = True)

# END ARRAY
# BEGIN RANDOM

def key(s = SEED):
    return jax.random.PRNGKey(s)

def random(shape, f, s = SEED): 
    SHAPE = (shape,1,1) if type(shape) == int else shape
    MAX = f.q if f.n*jax.numpy.log2(f.p) < 63 else jax.numpy.iinfo(DTYPE).max
    a = jax.random.randint(key(s), SHAPE, 0, MAX, dtype = DTYPE)
    return array(a,f)

# END RANDOM
# BEGIN RINGS

@functools.partial(jax.jit, static_argnums = 0)
def Z2(q):
    Zq = jax.numpy.arange(q)
    Z2q = jax.numpy.array([jax.numpy.tile(Zq,q), jax.numpy.repeat(Zq,q)]).T
    return Z2q # The finite ring Z/q x Z/q.

@functools.partial(jax.jit, static_argnums = 0)
def M2(q):
    Z2q = Z2(q)
    M2q = jax.numpy.array([jax.numpy.tile(Z2q.T,q*q).T, jax.numpy.repeat(Z2q.T,q*q).reshape(2,-1).T]).swapaxes(0,1).reshape(-1,2,2)
    return M2q # The finite ring M_2( Z/q ).

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
    return jax.numpy.array([[a,b],[c,d]], dtype = DTYPE)

# END ENCODE/DECODE 2D
# BEGIN GROUPS

def gl2(f):

    @jax.jit
    def det(x):
        a = block(lift(decode(x,f.q),f),f)
        return proj((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    assert f.q < 256
    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where(m2d > 0, m2c, 0))[0]
    gl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return gl2c, idx

def sl2(f):

    @jax.jit
    def det(x):
        a = block(lift(decode(x,f.q),f),f)
        return proj((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    assert f.q < 256
    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where(m2d == 1, m2c, 0))[0]
    sl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return sl2c, idx

def pgl2(f):
    assert f.q < 256

    @jax.jit
    def det(x):
        a = block(lift(decode(x,f.q),f),f)
        return proj((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    @jax.jit
    def normed(x):
        A = block(lift(decode(x,f.q),f),f)
        a, b = proj(A[:,0,0,:,:],f).ravel(), proj(A[:,0,1,:,:],f).ravel()
        return (a == 1) | ((a == 0) & (b == 1))

    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    m2n = jax.vmap(normed)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where((m2d > 0) & m2n, m2c, 0))[0]
    pgl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return pgl2c, idx

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

    m2c = jax.numpy.arange(q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c)
    m2n = jax.vmap(normed)(m2c)
    i = jax.numpy.nonzero(jax.numpy.where((m2d > 0) & m2n, m2c, 0))[0]
    pgl2c = m2c[i]
    idx = q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return pgl2c, idx

def psl2(f):
    assert f.q < 256
    if f.p == 2:
        return pgl2(f)

    @jax.jit
    def det(x):
        a = block(lift(decode(x,f.q),f),f)
        return proj((a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:])%f.p,f).ravel()

    @jax.jit
    def normed(x):
        A = block(lift(decode(x,f.q),f),f)
        a, b = proj(A[:,0,0,:,:],f).ravel(), proj(A[:,0,1,:,:],f).ravel()
        return ((a != 0) & (a < f.q/2)) | ((a == 0) & (b < f.q/2))

    m2c = jax.numpy.arange(f.q**4, dtype = jax.numpy.uint32)
    m2d = jax.vmap(det)(m2c).squeeze()
    m2n = jax.vmap(normed)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where((m2d == 1) & m2n, m2c, 0))[0]
    psl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = jax.numpy.uint32)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = jax.numpy.uint32))
    return psl2c, idx

def psl2mod(q):
    assert q < 256
    if q == 2:
        return pgl2mod(q)

    @jax.jit
    def det(x):
        a = decode(x,q)
        return (a[0,0]*a[1,1]-a[0,1]*a[1,0])%q

    @jax.jit
    def normed(x):
        A = decode(x,q)
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
# BEGIN EXPANDER

@functools.partial(jax.jit, static_argnums = 0)
def pS1(p):

    def norm(a):
        return a[0]*a[0]+a[1]*a[1]

    R = Z2(p)
    N = jax.vmap(norm)(R)%p
    M = jax.numpy.where(N == (p-1), N, -1)
    return R[jax.numpy.argmax(M)] # A point on the circle x*x + y*y = (p-1) mod p.

@functools.partial(jax.jit, static_argnums = (0,1))
def S3(p,r): # r = sqrt(p).

    def norm(a):
        return a[0,0]*a[0,0]+a[0,1]*a[0,1]+a[1,0]*a[1,0]+a[1,1]*a[1,1]
    
    R = M2(2*r)-r
    N = jax.vmap(norm)(R)
    return R, jax.numpy.where(N == p,N,-1) # The integer three sphere x*x + y*y + z*z + t*t = p.

def S(p,q):
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
    return jax.vmap(f)(Sp) # p+1 generators for the group PSL2q, if p is a quadratic residue mod q, else PGL2q.

def lps(p,q): # The Lubotzky-Phillips-Sarnak expander graph is a p+1-regular Cayley graph for the group PSL2q or PGL2q.
    assert (p in CONWAY) and (q in CONWAY) and (p != q) and (p > 2) and (q > 2) and (q*q > 4*p)
    f = field(q,1)
    l = f.LEG[p%q]

    @jax.jit
    def normpgl(A):
        a, b = A[0,0], A[0,1]
        sa = jax.numpy.sign(a)
        c = f.INV[sa*a + (1-sa)*b]
        return c*A%q

    @jax.jit
    def normpsl(A):
        a, b = A[0,0], A[0,1]
        sa, sqa, sqb = jax.numpy.sign(a), jax.numpy.sign((q//2)-a), jax.numpy.sign((q//2)-b)
        s = sa*sqa + (1-sa)*sqb
        return s*A%q

    @jax.jit
    def norm(A):
        return jax.lax.cond(l==1, normpsl, normpgl, A)
    #norm = normpsl if f.LEG[p%q] == 1 else normpgl
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

    G, i = psl2mod(q) if f.LEG[p] == 1 else pgl2mod(q)
    graph = jax.vmap(mul)(G)
    return graph, i

# END EXPANDER
# END POPPY
