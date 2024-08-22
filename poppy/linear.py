import jax
import functools
from poppy.constant import DTYPE, BLOCKSIZE
from poppy.modular import matmulmod

@jax.jit
def trace(a,p):
    # Field trace.
    # a has shape b r c n n.
    return jax.numpy.trace(a, axis1 = -2, axis2 = -1)%p # b r c.
@jax.jit
def trace4(a,p):
    # Matrix trace.
    # a has shape b r c n.
    return jax.numpy.trace(a, axis1 = 1, axis2 = 2)%p # b n.

@jax.jit
def dot33(a,b,p):
    # a has shape r n n.
    # b has shape r n n.
    return jax.numpy.einsum('rni,rim->nm',a,b)%p # n n.
    #return jax.numpy.tensordot(a,b, axes = ([0,2],[0,1]))%p # n n.
@jax.jit
def mul32(a,b,p):
    # a has shape r c n.
    # b has shape n n.
    return jax.numpy.einsum('rcm,mn->rcn',a,b)%p # r c n.
    #return jax.numpy.tensordot(a,b, axes = (2,0))%p # r c n.
@jax.jit
def mul53(a,b,p):
    # a has shape b r c n n.
    # b has shape b n n.
    return jax.numpy.einsum('brcni,bim->brcnm',a,b)%p # b r c n n.
@jax.jit
def mul45(a,b,p):
    # a has shape b r c n.
    # b has shape b r c n n.
    return jax.numpy.einsum('brci,brcin->brcn',a,b)%p
@jax.jit
def matmul44(a,b,p): 
    # a has shape r c n n.
    # b has shape c t n n.
    return jax.numpy.einsum('rcni,ctim->rtnm',a,b)%p # r t n n.
    #return jax.numpy.tensordot(a,b, axes = ([1,3],[0,2])).swapaxes(1,2)%p # r t n n.
@jax.jit
def matmul55(a,b,p):
    # a has shape b r c n n. 
    # b has shape b c t n n.
    return jax.numpy.einsum('brcni,bctim->brtnm',a,b)%p # b r t n n.
    #return jax.vmap(matmul44, in_axes = (0,0,None))(a,b,p) # b r t n n.
@jax.jit
def matmul34(a,b,p): 
    # a has shape r c n.
    # b has shape c t n n.
    return jax.numpy.einsum('rci,ctin->rtn',a,b)%p # r t n.
    #return jax.numpy.tensordot(a,b, axes = ([1,2],[0,2]))%p # r t n.
@jax.jit
def matmul45(a,b,p):
    # a has shape b r c n.
    # b has shape b c t n n.
    return jax.numpy.einsum('brci,bctin->brtn',a,b)%p # b r t n.
    #return jax.vmap(matmul34, in_axes = (0,0,None))(a,b,p) # b r t n.
@jax.jit
def outer33(a,b,p):
    # a has shape r n n.
    # b has shape c n n.
    return jax.numpy.einsum('rni,cim->rcnm', a,b)%p # r c n n.
    #return jax.numpy.tensordot(a,b, axes = (2,1)).swapaxes(1,2)%p # r c n n.
@jax.jit
def outer44(a,b,p):
    # a has shape b r n n.
    # b has shape b c n n.
    return jax.numpy.einsum('brni,bcim->brcnm', a,b)%p # b r c n n.

@jax.jit
def mtrsm(a,b,p): 
    # Triangular solve mod p.
    # Assume unit diagonals.
    r,c = a.shape # b has shape r t.
    COL = jax.numpy.zeros(c, dtype = DTYPE)
    ROW = jax.numpy.arange(1,r, dtype = DTYPE)
    def mtrsm_vmap(bt): 
        # bt is b transpose.
        def mtrsm_scan(col): 
            # col is a column of b.
            def eliminate(cc,j):
                # cc is a column of the solution.
                cc = cc.at[j].set((col[j]-jax.numpy.dot(a[j],cc))%p)
                return cc, cc[j]  
            return jax.lax.scan(eliminate, COL.at[0].set(col[0]), ROW)[0] # scan the rows of a.
        return jax.vmap(mtrsm_scan)(bt)  # vmap the columns of b.
    return mtrsm_vmap(b.T).T

@jax.jit
def mgetrf2(aperm,inverse): 
    # Sequential lu decomposition mod p.
    r,c = aperm.shape
    p = 1+inverse[-1] # inverse has shape p.
    I = jax.numpy.arange(r)
    J = jax.numpy.arange(c-1)
    R = jax.numpy.arange(min(r,c-1))
    def eliminate(ap,i):
        j = jax.numpy.argmax(jax.numpy.where(I >= i, ap[:,i], -1)) # Search column i for j.
        ap = ap.at[[i,j],:].set(ap[[j,i],:]) # Swap rows i and j.
        ap = ap.at[:,i].set(jax.numpy.where(I>i,(ap[:,i]*inverse[ap[i,i]])%p,ap[:,i]))  # Scale column i.
        ap = ap.at[:,:-1].set((ap[:,:-1]-jax.numpy.where((I[:,None]>i)&(J[None,:]>i),jax.numpy.outer(ap[:,i],ap[i,:-1]),0))%p) # Update block.     
        return ap,i
    return jax.lax.scan(eliminate, aperm,R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def mgetrf1(a,inverse,bs): 
    # Blocked lu decompposition mod p.
    # bs is block size.
    r,c = a.shape
    p = 1+inverse[-1] # inverse has shape p.
    m = min(r,c)
    perm = jax.numpy.arange(r)
    for i in range(0,m,bs):
        bb = min(m-i,bs)
        ap = mgetrf2(jax.numpy.hstack([a[i:,i:i+bb],jax.numpy.arange(i,r).reshape((-1,1))]), inverse)
        perm = perm.at[i:].set(perm[ap[:,-1]]) # Update permutation.
        a = a.at[i:,:].set(a[ap[:,-1],:]) # Swap rows.
        a = a.at[i:,i:i+bb].set(ap[:,:-1])  # Update block C.
        a = a.at[i:i+bb,i+bb:].set(mtrsm(a[i:i+bb,i:i+bb], a[i:i+bb,i+bb:], p)) # Update block B.
        a = a.at[i+bb:,i+bb:].set((a[i+bb:,i+bb:] - jax.lax.dot(a[i+bb:,i:i+bb],a[i:i+bb,i+bb:]))%p) # Update block D.
    l = jax.numpy.fill_diagonal(jax.numpy.tril(a), 1, inplace = False)
    u = jax.numpy.tril(a.T).T
    d = jax.numpy.diagonal(u)
    iperm = jax.numpy.arange(r)
    iperm = iperm.at[perm].set(iperm)  
    return l,u,d,iperm

# Batched blocked lu decomposition mod p.
mgetrf = jax.vmap(mgetrf1, in_axes = (0,None,None))

def inv1(a,i,bs): 
    # Matrix inverse mod p.
    # i has shape p.
    # bs is block size.
    r,c = a.shape
    if r == 1:
        return i[a[0,0]].reshape((1,1))
    p = 1+i[-1]
    I = jax.numpy.eye(r, dtype = DTYPE)
    l,u,d,iperm = mgetrf1(a,i,bs)
    D = i[d]
    L = mtrsm(l,I,p) # L = 1/l.
    U = mtrsm((D*u%p).T,D*I,p).T # U = 1/u.      
    return (U@L%p)[:,iperm]

def inv(a,i,bs): 
    # Batched matrix inverse mod p.
    # i has shape p.
    # bs is block size.
    b,r,c = a.shape
    if r == 1:
        return i[a[:,0,0]].reshape((b,1,1))
    p = 1+i[-1]
    I = jax.numpy.eye(r, dtype = DTYPE)
    def inverse(A):
        l,u,d,iperm = mgetrf1(A,i,bs)
        D = i[d]
        L = mtrsm(l,I,p) # L = 1/l.
        U = mtrsm((D*u%p).T,D*I,p).T # U = 1/u.      
        return (U@L%p)[:,iperm]
    return jax.vmap(inverse)(a)

@jax.jit
def trsm(a,b,p): 
    # Triangular solve over a finite field.
    # Assume unit diagonals.
    # p is the field characteristic.
    r,c,n,_ = a.shape # b has shape r t n n.
    COL = jax.numpy.arange(c, dtype = DTYPE)
    ROW = jax.numpy.arange(1,r, dtype = DTYPE)
    ZERO = jax.numpy.zeros((n,n), dtype = DTYPE)
    def trsm_vmap(bt): 
        # bt is b transpose.
        def trsm_scan(col): 
            # col is a column of b.
            def eliminate(cc,j):
                # cc is a column of the solution.
                cc = cc.at[j].set((col[j]-dot33(a[j],cc,p))%p)
                return cc, cc[j]  
            return jax.lax.scan(eliminate, jax.numpy.where(COL[:,None,None] == 0,col[0],ZERO), ROW)[0] # scan the rows of a.
        return jax.vmap(trsm_scan)(bt)  # vmap the columns of b.
    return trsm_vmap(b.swapaxes(0,1)).swapaxes(0,1)

@jax.jit
def getrf2(aperm,inverse): 
    # Sequential lu decomposition over a finite field.
    a,perm,parity = aperm
    r,c,n,n = a.shape
    p = 1+inverse[-1] # p is the field characteristic.
    I = jax.numpy.arange(r).reshape((-1,1,1,1))
    J = jax.numpy.arange(c).reshape((1,-1,1,1))
    R = jax.numpy.arange(min(r,c))
    def eliminate(ap, i):
        a, perm, parity = ap # perm has shape r. parity has shape 1.
        ai = a[:,i,:,:].reshape((r,-1)).max(axis = 1)
        j = jax.numpy.argmax(jax.numpy.where(I[:,0,0,0] >= i, ai, -1)) # Search column i for j.
        a = a.at[[i,j],:,:,:].set(a[[j,i],:,:,:]) # Swap rows i and j.
        perm = perm.at[[i,j],].set(perm[[j,i],]) # Record swap.
        a = a.at[:,i,:,:].set(jax.numpy.where(I[:,:,:,0]>i,mul32(a[:,i,:,:],inv1(a[i,i,:,:],inverse,BLOCKSIZE),p),a[:,i,:,:])) # Scale column i.
        a = a.at[:,:,:,:].set((a - jax.numpy.where((I > i)&(J > i), outer33(a[:,i,:,:],a[i,:,:,:],p), 0))%p) # Update block.   
        parity = (parity + jax.numpy.count_nonzero(i-j)) % 2
        return (a,perm,parity), j
    return jax.lax.scan(eliminate, aperm, R, unroll = False)[0]

@functools.partial(jax.jit, static_argnums = 2)
def getrf1(a,inverse,bs): 
    # Blocked lu decompposition over a finite field.
    # bs is block size.
    p = 1+inverse[-1] # p is the field characteristic.
    r,c,n,_ = a.shape
    m = min(r,c)
    R = jax.numpy.arange(r)
    perm = jax.numpy.arange(r)
    parity = jax.numpy.zeros(1, dtype = DTYPE)
    for i in range(0, m, bs):
        bb = min(m-i, bs)
        ai, permi, pari = getrf2((a[i:,i:i+bb,:,:], R[i:], 0), inverse) # a has shape r-i bb n n. R has shape r-i.
        parity = (parity+pari)%2
        perm = perm.at[i:].set(perm[permi])
        a = a.at[i:,:,:,:].set(a[permi,:,:,:]) # Swap rows.
        a = a.at[i:,i:i+bb,:,:].set(ai)  # Update block C.
        a = a.at[i:i+bb,i+bb:,:,:].set(trsm(a[i:i+bb,i:i+bb,:,:],a[i:i+bb,i+bb:,:,:],p)) # Update block B.
        a = a.at[i+bb:,i+bb:,:,:].set((a[i+bb:,i+bb:,:,:]-matmul44(a[i+bb:,i:i+bb,:,:],a[i:i+bb,i+bb:,:,:],p))%p) # Update block D.
    I = jax.numpy.eye(n, dtype = DTYPE)
    i = R[:,None,None,None]
    j = R[None,:,None,None]
    l = jax.numpy.where(i >  j, a, 0)
    l = jax.numpy.where(i == j, I, l)
    u = jax.numpy.where(i <= j, a, 0)
    d = jax.numpy.diagonal(a, offset = 0, axis1 = 0, axis2 = 1).swapaxes(0,2).swapaxes(1,2)
    iperm = jax.numpy.arange(r)
    iperm = iperm.at[perm].set(iperm)  
    return l,u,d,iperm,parity

# Batched blocked lu decomposition over a finite field.
getrf = jax.vmap(getrf1, in_axes = (0, None, None))

def det(a,i,p,bs): 
    # Batched matrix determinant over a finite field.
    # a has shape b r c n n.
    # i has shape p.
    # p is the field characteristic.
    # bs is block size.
    def mul(A,B):
        return matmulmod(A,B,p)
    def det1(A):
        l,u,d,iperm,parity = getrf1(A,i,bs)
        return jax.numpy.power(-1, parity) * jax.lax.associative_scan(mul,d)[-1]% p
    return jax.vmap(det1)(a)

@jax.jit
def gje2(apiv,inverse): 
    # Sequential Gauss-Jordan elimination over a finite field.
    # inverse has shape p.
    a,piv = apiv # piv has shape b c.
    b,r,c,n,_ = a.shape
    p = 1+inverse[-1] # p is the field characteristic.
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
        return jax.numpy.where(pv[i,None,None] != 0, a[i,j,:,:],jax.numpy.eye(n,dtype = DTYPE)[None,:,:])
    def updateblock(a,i,j):
        return a.at[:,:,:,:].set((a-jax.numpy.where((I != i)&(J >= j), outer33(a[:,j,:,:],a[i,:,:,:],p), 0))%p)     
    def eliminate(ap, j):
        a, piv = ap
        aj = a[:,:,j,:,:].reshape((b,r,-1)).max(axis = 2) 
        i = jax.vmap(searchpiv)(piv).reshape((b,))  
        k = jax.vmap(searchcol)(aj,i).reshape((b,)) 
        a = jax.vmap(swaprows)(a,i,k)
        piv = jax.vmap(updatepiv, in_axes = (0,0,None,0,0))(piv,i,j,k,aj)
        d = jax.vmap(extractpiv, in_axes = (0,0,0,None))(piv,a,i,j).reshape((b,n,n)) 
        a = a.at[:,i,:,:,:].set(mul53(a[:,i,:,:,:], inv(d, inverse, BLOCKSIZE),p)) # Scale row i.
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
def kerim(a,f): 
    # Matrix kernel and image over a finite field.
    b,r,c,n,_ = a.shape
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
    ker = ker.at[:,:m,:,:,:].set(jax.numpy.where(J >= rank, -rref[:,:m,:,:,:],0))%f.p
    ker = ker.at[:,:,:,:,:].set(jax.numpy.where((J >= rank)&(I == J)&(i == j), 1,ker))
    iim = iim.at[:,:m,:,:,:].set(jax.numpy.where((I[:,:m] < rank)&(J < rank), rref[:,:m,:,:,:],0))
    def swap(ki,pm):
        return ki[pm]
    ker = jax.vmap(swap)(ker,perm)
    iim = jax.vmap(swap)(iim,perm)
    im = matmul55(a,iim,f.p)
    return ker,im,rank