# BEGIN POPPY
# BEGIN INIT

import jax
import conway_polynomials
import functools
import matplotlib.pyplot

# 64 bit integer arrays encode numbers in finite fields.
jax.config.update("jax_enable_x64", True)
DTYPE = jax.numpy.int64

# Finite fields are polynomial rings modulo Conway polynomials.
CONWAY = conway_polynomials.database()

p12 = 3329
p22 = 4194191
p30 = 999999733

CONWAY[p22] = {}
CONWAY[p30] = {}

CONWAY[p12][256] = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

CONWAY[p22][32] = (1210382, 2465089, 2473115, 1675214, 1402336, 2768999, 1423446, 3813950, 300273, 2698959, 2739930, 288062, 1937372, 1497279, 3718707, 684317, 3086020, 1187785, 3245176, 4101125, 3053698, 2428306, 430958, 1959819, 2481505, 3556404, 3611824, 136721, 56755, 4193498, 4193819, 1, 1)
CONWAY[p22][64] = (2113114, 2865230, 1576056, 3726964, 2877627, 499207, 2035891, 2007157, 625689, 3100185, 3324040, 546690, 676409, 1083835, 2204012, 3956530, 3639404, 951145, 1430843, 343435, 1535584, 1706721, 2557097, 1812042, 2660473, 551889, 1964759, 3794689, 3016365, 3639000, 3307938, 1631856, 3950905, 2203751, 401388, 237219, 1096646, 192834, 4078359, 2208065, 2343083, 1293731, 642048, 1067968, 2116063, 4046513, 623914, 2313263, 309590, 884598, 1852277, 2448870, 1416168, 2308687, 1232242, 2956988, 3264982, 3833720, 1675882, 217763, 65065, 4193348, 4193813, 1, 1)
CONWAY[p22][128] = (3133421, 88403, 1589587, 2909025, 1947932, 1225311, 424283, 3045769, 3412942, 3528628, 4067083, 707002, 2893288, 2504860, 3737776, 1938463, 2834128, 567607, 708408, 935357, 1223815, 2525367, 3882712, 609233, 3433020, 1753395, 1362370, 1963203, 3076617, 682623, 104210, 1393179, 3754654, 4156996, 2072991, 1637153, 552476, 4193193, 2501505, 266823, 2921479, 65530, 1222221, 331871, 1756060, 2932245, 2693668, 2025618, 586524, 3172278, 721227, 3807584, 2185003, 2248009, 3223478, 3680719, 962420, 1599177, 3498371, 575936, 3619948, 2315835, 587007, 1446277, 2441306, 738019, 731121, 538389, 119078, 2514615, 3518614, 2674450, 3554783, 2056540, 3985179, 1671120, 413524, 1193541, 3528041, 3585378, 4144966, 1482613, 2006414, 389176, 725556, 4019057, 1475793, 2966325, 3851978, 2460650, 1157580, 759005, 3229412, 3644368, 3216267, 2984962, 3462836, 1013948, 2942350, 2050904, 21951, 4101626, 2340097, 2712950, 3742225, 1101759, 807746, 3043570, 3123790, 3665199, 1964294, 4075652, 654006, 3222199, 738778, 2471354, 2311314, 423868, 2499755, 171152, 316179, 2179489, 513830, 251663, 69250, 4193309, 4193810, 1, 1)

CONWAY[p30][32] = (27583, 93555, 999979589, 999800744, 175914, 999685076, 132437, 999673390, 496535, 151090, 431212, 999462160, 98317, 33297, 541045, 999835792, 999862880, 999688933, 75534, 54402, 95912, 999983150, 999989224, 999983931, 1854, 1071, 1385, 999999634, 999999680, 999999673, 2, 1, 1) # PARI/GP: Vecrev(lift(Vec(ffinit(999999733,32))))
CONWAY[p30][64] = (880158359, 421910987, 238580159, 47835162, 125188131, 203934989, 164672785, 362892651, 887918584, 521609740, 24133069, 86039701, 214120832, 23365488, 654306687, 69149380, 344957381, 692130410, 243093192, 201020705, 834659236, 673379853, 23445898, 85914969, 19546546, 16408258, 696013949, 475723044, 330268037, 347300024, 273007531, 55530684, 328005007, 674587458, 392709403, 158715956, 586029521, 996777825, 312436560, 717852169, 909508614, 398983312, 982554754, 131964166, 713742405, 445760565, 663413759, 920919556, 952784420, 934855743, 6164104, 3531442, 4536664, 999671374, 999818632, 999782267, 11326, 6031, 6793, 999999506, 999999616, 999999609, 2, 1, 1) # PARI/GP: Vecrev(lift(Vec(ffinit(999999733,64))))
CONWAY[p30][128] = (928209554, 689007997, 472485176, 311285921, 272458838, 229002643, 526199429, 748788741, 158402309, 925169829, 607190820, 266515213, 911332241, 21650296, 407449831, 606489086, 265113041, 867081435, 180429632, 925706999, 112686331, 958957946, 754893271, 68756580, 276636789, 167171103, 188832257, 264467404, 463644290, 383082157, 754617689, 567989677, 304813600, 401831854, 85865675, 624795044, 613047041, 773771412, 485323633, 741466547, 214401332, 23836307, 469318615, 15716832, 584372832, 199675878, 604520832, 883896696, 68443640, 320810623, 46618158, 84119126, 344029114, 384386949, 460884580, 318695963, 855693480, 232903525, 308055698, 160381091, 466451351, 835227787, 883525455, 636838051, 399737446, 845304369, 649162109, 365256802, 543813349, 814514710, 564586964, 716267650, 590065529, 979017259, 231493101, 874990194, 525337367, 613803546, 253612586, 736002404, 833094381, 517615489, 775532462, 57085789, 907036708, 682687804, 30153072, 898793307, 976984826, 819505151, 366020328, 27957222, 240548646, 189925218, 484884196, 554094810, 248828094, 392437155, 542166060, 323232758, 942826697, 540641589, 490212731, 976754548, 97293091, 681872372, 657942597, 179669288, 889075267, 671880645, 444029812, 53206735, 646778118, 534319291, 254877140, 967882434, 58219934, 785914891, 694008030, 165003883, 616862256, 960237312, 992125181, 251663, 69250, 999998851, 999999352, 1, 1) # PARI/GP: Vecrev(lift(Vec(ffinit(999999733,128))))

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
def transpose(a):
    return a.swapaxes(-2,-1) 

@jax.jit
def ptrace(a,p):
    return jax.numpy.trace(a, axis1=-2, axis2=-1)%p

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

def pinv(a, inv, b): # Matrix inverse mod p.
    if len(a) == 1:
        return inv[a[0,0]].reshape((1,1))
    p = 1+inv[-1]
    I = jax.numpy.eye(len(a), dtype = DTYPE)
    l, u, d, iperm = pgetrf(a, inv, b)
    D = inv[d]
    L = ptrsm(l, I, p) # L = 1/l.
    U = ptrsm((D*u%p).T, D*I, p).T # U = 1/u.      
    return (U@L%p)[:,iperm]

def pinv_vmap(a, inv, b): # Matrix inverse mod p.
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

def qdet(a, inv, p, b): # Matrix determinant over a finite field.
    def matmul(A,B):
        return pmatmul(A,B,p)
    l, u, d, iperm, parity = qgetrf(a, inv, b)
    return jax.numpy.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p

def qdet_vmap(a, inv, p, b): # Matrix determinant over a finite field.
    def matmul(A,B):
        return pmatmul(A,B,p)
    def det(A):
        l, u, d, iperm, parity = qgetrf(A, inv, b)
        return jax.numpy.power(-1, parity) * jax.lax.associative_scan(matmul, d)[-1]% p
    return jax.vmap(det)(a)

# END LINEAR ALGEBRA
# BEGIN FIELD

class field:
    def __init__(self, p, n, inv = True):    
        self.p = p
        self.n = n
        self.q = p**n
        self.q = p ** n if n*jax.numpy.log2(p) < 63 else None
        self.INV = self.inv() if inv else None # Multiplicative inverse mod p.
        self.BASIS = self.basis()     # Powers of the Conway matrix.
        self.DUALBASIS = self.dualbasis()

    def __repr__(self):
        return f'field {self.q}'
     
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
            return pneg(a, self.p)
        
        @jax.jit
        def matmul(a,b):
            return pmatmul(a,b, self.p)
        
        # V is the vector of subleading coefficients of the Conway polynomial.
        V = jax.numpy.array(CONWAY[self.p][self.n][:-1], dtype = DTYPE)
        # M is the companion matrix of the Conway polynomial.
        M = jax.numpy.zeros((self.n,self.n), dtype = DTYPE).at[:-1,1:].set(jax.numpy.eye(self.n-1, dtype = DTYPE)).at[-1].set(neg(V))
        # X is the array of powers of M.
        X = jax.lax.associative_scan(matmul, stack(M,jax.numpy.arange(self.n, dtype = DTYPE)).at[0].set(jax.numpy.eye(self.n, dtype = DTYPE)))
        return X

  
    def dualbasis(self):
        A = jax.numpy.array(CONWAY[self.p][self.n][:-1], dtype = DTYPE)
        R = self.BASIS[1]
        Ri = pinv(R,self.INV,32)
        DD = jax.numpy.zeros((self.n,self.n,self.n), dtype = DTYPE).at[0,:,:].set((-Ri*A[0])%self.p)
        def dualscan(b,i):
            b = b.at[i].set((Ri@b[i-1]-Ri*A[i])%self.p)
            return b, b[i]
        DD = jax.lax.scan(dualscan,DD,jax.numpy.arange(1,self.n))[0]
        C = jax.numpy.tensordot(DD,self.BASIS,axes = ([0,2],[0,1]))%self.p
        Ci = pinv(C,self.INV,32)
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
def i2v(i,f):
    return jax.numpy.floor_divide(i*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.power(f.p*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.arange(f.n, dtype = DTYPE))) % f.p

@functools.partial(jax.jit, static_argnums = 1)
def v2i(v,f):
    return jax.numpy.sum(v*jax.numpy.power(f.p*jax.numpy.ones(f.n, dtype = DTYPE), jax.numpy.arange(f.n, dtype = DTYPE)), dtype = DTYPE)

@functools.partial(jax.jit, static_argnums = 1)
def v2m(v,f):
    return jax.numpy.dot(v, f.BASIS) % f.p

@functools.partial(jax.jit, static_argnums = 1)
def m2v(m,f):
    return m[0]

@functools.partial(jax.jit, static_argnums = 1)
def i2m(i,f):
    return v2m(i2v(i,f), f)

@functools.partial( jax.jit, static_argnums = 1)
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
        if self.shape[0] == 1:
            return array(qdet(block(self.REP, self.field)[0], self.field.INV, self.field.p, 32), dtype = self.field, lifted = True)
        return array(qdet_vmap(block(self.REP, self.field), self.field.INV, self.field.p, 32), dtype = self.field, lifted = True)

    def lu(self):
        if self.shape[0] == 1:
            return pgetrf(self.REP[0], self.field.INV, 32)
        return pgetrf_vmap(self.REP, self.field.INV, 32)

    def lu_block(self):
        if self.shape[0] == 1:
            return qgetrf(block(self.REP, self.field)[0], self.field.INV, 32)
        return qgetrf_vmap(block(self.REP, self.field), self.field.INV, 32)

    def inv(self):
        if self.shape[0] == 1:
            return array(pinv(self.REP[0], self.field.INV, 32), dtype = self.field, lifted = True)
        return array(pinv_vmap(self.REP, self.field.INV, 32), dtype = self.field, lifted = True)

    def rank(self):
        return jax.numpy.count_nonzero(self.lu()[2], axis = 1)

def flatten_array(a):
    children = (a.shape, a.REP)
    aux_data = (a.field,)
    return (children, aux_data)
def unflatten_array(aux_data, children):
    a = object.__new__(array)
    a.shape, a.REP = children
    a.field, = aux_data
    return a
jax.tree_util.register_pytree_node(array, flatten_array, unflatten_array)

# END ARRAY
# BEGIN RANDOM

def key(s = SEED):
    return jax.random.key(s)

def random(shape, f, s = SEED): 
    SHAPE = (shape,1,1) if type(shape) == int else (shape[0],1,1) if len(shape) == 1 else (shape[0],shape[1],1) if len(shape) == 2 else shape
    a = jax.random.randint(key(s), SHAPE+(f.n,), 0, f.p, dtype = DTYPE)
    return array(unravel(jax.vmap(v2m, in_axes = (0,None))(a.reshape((-1,f.n)),f).reshape((SHAPE[0],-1,f.n,f.n)),f,SHAPE[:-2]+(SHAPE[-2]*f.n,SHAPE[-1]*f.n)),f, lifted = True)

# END RANDOM
# BEGIN PLOT

def plot(a, title = '', cmap = 'twilight_shifted'):
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
    return jax.numpy.array([[a,b],[c,d]], dtype = DTYPE)

# END ENCODE/DECODE 2D
# BEGIN GROUPS

def gl2(f): # The general linear group GL_2( F ).

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

def sl2(f): # The special linear group SL_2( F ).

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

def pgl2(f): # The projective general linear group PGL_2( F ).
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

def pgl2mod(q): # The projective general linear group PGL_2( Z/q ).
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

def psl2(f): # The projective special linear group PSL_2( F ).
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

def psl2mod(q): # The projective special linear group PSL_2( Z/q ).
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
# BEGIN EXPANDERS

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

    G, i = psl2mod(q) if l == 1 else pgl2mod(q)
    graph = jax.vmap(mul)(G)
    return graph, i

# END EXPANDERS
# END POPPY
