import jax
import functools
from poppy.modular import matmulmod

# BEGIN LINEAR ALGEBRA

# 64 bit integer arrays encode numbers in finite fields.
jax.config.update("jax_enable_x64", True)
DTYPE = jax.numpy.int64

# Linear algebra subroutines are blocked.
BLOCKSIZE = 32

@jax.jit
def transpose(a):
    return a.swapaxes(-2,-1) 

@jax.jit
def tracemod(a,p):
    return jax.numpy.trace(a, axis1=-2, axis2=-1)%p

@jax.jit
def mtrsm(a,b,p): # Triangular solve mod p.
    R = jax.numpy.arange(len(a), dtype = DTYPE) # a has shape r c.
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
def mgetrf1(a, inv, b): # Blocked lu decompposition mod p.
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

mgetrf = jax.vmap(mgetrf1, in_axes = (0, None, None))

def invmod1(a, inv, b): # Matrix inverse mod p.
    if len(a) == 1:
        return inv[a[0,0]].reshape((1,1))
    p = 1+inv[-1]
    I = jax.numpy.eye(len(a), dtype = DTYPE)
    l, u, d, iperm = mgetrf1(a, inv, b)
    D = inv[d]
    L = mtrsm(l, I, p) # L = 1/l.
    U = mtrsm((D*u%p).T, D*I, p).T # U = 1/u.      
    return (U@L%p)[:,iperm]

def invmod(a, inv, b): # Matrix inverse mod p.
    if a.shape[1] == 1:
        return inv[a[:,0,0]].reshape((a.shape[0],1,1))
    p = 1+inv[-1]
    I = jax.numpy.eye(a.shape[1], dtype = DTYPE)
    def inverse(A):
        l, u, d, iperm = mgetrf1(A, inv, b)
        D = inv[d]
        L = mtrsm(l, I, p) # L = 1/l.
        U = mtrsm((D*u%p).T, D*I, p).T # U = 1/u.      
        return (U@L%p)[:,iperm]
    return jax.vmap(inverse)(a)

@jax.jit
def trsm(a,b,p): # Triangular solve over a finite field.
    R = jax.numpy.arange(len(a), dtype = DTYPE) # a has shape r c n n.
    ZERO = jax.numpy.zeros((b.shape[-1],b.shape[-1]), dtype = DTYPE) # b has shape c d n n.
    def trsm_vmap(bb): # bb is the array b.
        def trsm_scan(bc): # bc has shape c n n. it is a column of bb.
            def f(x,j):
                x = x.at[j].set((bc[j] - jax.numpy.tensordot(a[j], x, axes = ([0,2],[0,1]))) % p)
                return x, x[j]  
            return jax.lax.scan( f, jax.numpy.where( R[:,None,None] == 0, bc[0], ZERO ), R[1:] )[0] # scan the rows of a.
        return jax.vmap(trsm_scan)(bb.swapaxes(0,1)).swapaxes(0,1)  # vmap the columns of b.
    return trsm_vmap(b)

@jax.jit
def getrf2(aperm, inv): # Sequential lu decomposition over a finite field.
    a, perm, parity = aperm
    p = 1+inv[-1] # p is prime.
    I = jax.numpy.arange(a.shape[0]).reshape((-1,1,1,1))
    J = jax.numpy.arange(a.shape[1]).reshape((1,-1,1,1))
    R = jax.numpy.arange(min(a.shape[0],a.shape[1]))
    def eliminate(ap, i):
        a, perm, parity = ap # a has shape r c n n. perm has shape r. parity has shape 1.
        ai = a[:,i,:,:].reshape((a.shape[0],-1)).max(axis = 1)
        j = jax.numpy.argmax(jax.numpy.where(I[:,0,0,0] >= i, ai, -1)) # Search column i for j.
        a = a.at[[i,j],:,:,:].set(a[[j,i],:,:,:]) # Swap rows i and j.
        perm = perm.at[[i,j],].set(perm[[j,i],]) # Record swap.
        a = a.at[:,i,:,:].set(jax.numpy.where(I[:,:,:,0] > i, (jax.numpy.tensordot(a[:,i,:,:], invmod1(a[i,i,:,:], inv, BLOCKSIZE), axes = (2,0))) % p, a[:,i,:,:])) # Scale column i.
        a = a.at[:,:,:,:].set((a[:,:,:,:] - jax.numpy.where((I > i) & (J > i), jax.numpy.tensordot(a[:,i,:,:], a[i,:,:,:], axes = (2,1)).swapaxes(1,2), 0)) % p) # Update block D.    
        parity = (parity + jax.numpy.count_nonzero(i-j)) % 2
        return (a, perm, parity), j
    return jax.lax.scan(eliminate, aperm, R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def getrf1(a, inv, b): # Blocked lu decompposition over a finite field.
    p = 1+inv[-1]
    r,c = a.shape[0], a.shape[1]
    m = min(r,c)
    R = jax.numpy.arange(r)
    perm = jax.numpy.arange(r)
    parity = jax.numpy.zeros(1, dtype = DTYPE)
    for i in range(0, m, b):
        bb = min(m-i, b)
        ai, permi, pari = getrf2((a[i:,i:i+bb,:,:], R[i:], 0), inv) # a has shape r-i bb n n. R has shape r-i.
        parity = (parity + pari) % 2
        perm = perm.at[i:].set(perm[permi])
        a = a.at[i:,:,:,:].set(a[permi,:,:,:]) # Swap rows.
        a = a.at[i:,i:i+bb,:,:].set(ai)  # Update block C.
        a = a.at[i:i+bb,i+bb:,:,:].set(trsm(a[i:i+bb,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], p)) # Update block B.
        a = a.at[i+bb:,i+bb:,:,:].set((a[i+bb:,i+bb:,:,:] - jax.numpy.tensordot(a[i+bb: ,i:i+bb,:,:], a[i:i+bb,i+bb:,:,:], axes = ([1,3],[0,2])).swapaxes(1,2)) % p) # Update block D.
    I = jax.numpy.eye(a.shape[-1], dtype = DTYPE)
    l = jax.numpy.where((R[:,None,None,None] - R[None,:,None,None]) > 0, a, 0)
    l = jax.numpy.where(R[:,None,None,None] == R[None,:,None,None], I, l)
    u = jax.numpy.where((R[:,None,None,None] - R[None,:,None,None]) <= 0, a, 0)
    d = jax.numpy.diagonal(a, offset = 0, axis1 = 0, axis2 = 1).swapaxes(0,2).swapaxes(1,2)
    iperm = jax.numpy.arange(len(perm))
    iperm = iperm.at[perm].set(iperm)  
    return l, u, d, iperm, parity

getrf = jax.vmap(getrf1, in_axes = (0, None, None))

def fdet(a, inv, p, b): # Matrix determinant over a finite field.
    def matmul(A,B):
        return matmulmod(A,B,p)
    def det(A):
        l, u, d, iperm, parity = getrf1(A, inv, b)
        return jax.numpy.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p
    return jax.vmap(det)(a)

@jax.jit
def gje2(apiv, inv): # Sequential Gauss-Jordan elimination over a finite field.
    a, piv = apiv # a has shape b r c n n. piv has shape b c.
    b,r,c,n,n = a.shape
    p = 1+inv[-1] # p is prime.
    I = jax.numpy.arange(r).reshape((r,1,1,1)) # Row indices.
    J = jax.numpy.arange(c).reshape((1,c,1,1)) # Column indices.
    def searchpiv(pv): # pv[i] = j+1 if the array a has a pivot at ij, else 0. 
        i = c-jax.numpy.argmax(jax.numpy.flip(jax.numpy.sign(pv)))
        i = jax.numpy.where(i==c, 0,i)
        i = jax.numpy.where(i>=r, r-1,i)
        return i
    def searchcol(aj,i): # search column j below row i for index k.  
        return jax.numpy.argmax(jax.numpy.where(I.reshape((1,-1)) >= i, aj, -1), axis = 1)
    def swaprows(a,i,j):
        return a.at[[i,j],:,:,:].set(a[[j,i],:,:,:])
    def updatepiv(pv,i,j,k,aj): # pv[i] = j+1 if the array a has a pivot at ij, else 0. 
        mask = jax.numpy.where(pv[i]>0,0,1)
        pv = pv.at[i].set(mask*(j+1)*jax.numpy.sign(aj[k])+(1-mask)*pv[i])
        return jax.numpy.where(J.squeeze() >= r, 0, pv)
    def extractpiv(pv,a,i,j):
        return jax.numpy.where(pv[i,None,None] != 0, a[i,j,:,:], jax.numpy.eye(n,dtype = DTYPE)[None,:,:])
    def updateblock(a,i,j):
        return a.at[:,:,:,:].set((a[:,:,:,:] - jax.numpy.where((I != i) & (J >= j), jax.numpy.einsum('jkl,mln->jmkn',a[:,j,:,:], a[i,:,:,:]), 0)) % p)    
    def eliminate(ap, j):
        a, piv = ap # a has shape b r c n n. piv has shape b c.
        aj = a[:,:,j,:,:].reshape((b,r,-1)).max(axis = 2) 
        i = jax.vmap(searchpiv)(piv).reshape((b,))  
        k = jax.vmap(searchcol)(aj,i).reshape((b,)) 
        a = jax.vmap(swaprows)(a,i,k)
        piv = jax.vmap(updatepiv, in_axes = (0,0,None,0,0))(piv,i,j,k,aj)
        d = jax.vmap(extractpiv, in_axes = (0,0,0,None))(piv,a,i,j).reshape((b,n,n)) 
        a = a.at[:,i,:,:,:].set(jax.numpy.einsum('brcin,bnm->brcim', a[:,i,:,:,:], invmod(d, inv, BLOCKSIZE)))%p # Scale row i.
        a = jax.vmap(updateblock, in_axes = (0,0,None))(a,i,j)
        return (a, piv), i
    def permute(a,sgnpiv):
        perm = jax.numpy.argsort(sgnpiv, axis = 0, descending = True)
        rank = jax.numpy.sum(sgnpiv)
        rref = a[perm[:r],:,:,:][:,perm,:,:]
        return perm,rank,rref
    a,piv = jax.lax.scan(eliminate, apiv, jax.numpy.arange(c), unroll = False)[0] # Scan the columns of a.
    perm,rank,rref = jax.vmap(permute)(a,jax.numpy.sign(piv)) # Move the pivots to the front of a.
    return piv,perm,rank,rref

@jax.jit
def kerim(a,f): # Matrix kernel and image over a finite field.
    b,r,c,n,_=a.shape
    m = min(r,c)
    apiv = (a, jax.numpy.zeros((b,c), dtype = DTYPE))
    piv,perm,rank,rref = gje2(apiv,f.INV)
    rank =             rank.reshape((b,1,1,1,1))
    I = jax.numpy.arange(c).reshape((1,c,1,1,1))
    J = jax.numpy.arange(c).reshape((1,1,c,1,1))
    i = jax.numpy.arange(n).reshape((1,1,1,n,1))
    j = jax.numpy.arange(n).reshape((1,1,1,1,n))
    ker = jax.numpy.zeros((b,c,c,n,n), dtype = DTYPE)
    iim = jax.numpy.zeros((b,c,c,n,n), dtype = DTYPE)
    ker = ker.at[:,:m,:,:,:].set(jax.numpy.where(J >= rank, -rref[:,:m,:,:,:], 0))%f.p
    ker = ker.at[:,:,:,:,:].set(jax.numpy.where((J >= rank) & (I == J) & (i == j), 1, ker[:,:,:,:,:]))
    iim = iim.at[:,:m,:,:,:].set(jax.numpy.where((I[:,:m] < rank) & (J < rank), rref[:,:m,:,:,:],0))
    def swap(ki,pm):
        return ki[pm]
    ker = jax.vmap(swap)(ker,perm)
    iim = jax.vmap(swap)(iim,perm)
    im = jax.numpy.einsum('ijklm,iknmt->ijnlt',a,iim)%f.p
    return ker,im,rank

# END LINEAR ALGEBRA