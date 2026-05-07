import jax
import functools
from poppy.constant import INT, BLOCKSIZE, SEED
from poppy.modular import mod
from poppy.linear import mul45, matmul34, trace4, trace, det, mgetrf, getrf, inv, kerim, gje2
from poppy.rep import int2vec, vec2int, vec2mat, mat2vec, block, unblock
from poppy.plot import plot

class array:
    def __init__(self, a, field):
        if type(a) == int:
            a = jax.numpy.array([a], dtype = field.dtype)
        elif type(a) == list:
            a = jax.numpy.array(a, dtype = field.dtype)
        if len(a.shape) == 1:
            a = a.reshape((a.shape[0],1,1))
        elif len(a.shape) == 2:
            a = a.reshape((1,a.shape[0],a.shape[1]))
        elif len(a.shape) > 3:
            print('ERROR: POPPY ARRAYS ARE THREE DIMENSIONAL.')
            return
        self.field = field
        self.shape = a.shape
        self.vec = int2vec(a, field)

    def new(self,v):
        a = object.__new__(array)
        a.field = self.field
        a.shape = v.shape[:-1]
        a.shape = (v.shape[0],1,1) if len(v.shape) == 2 else (1,v.shape[0],v.shape[1]) if len(v.shape) == 3 else v.shape[:-1]
        a.vec = v
        return a

    def __repr__(self):
        return f'SHAPE {self.shape[0]} {self.shape[1]} {self.shape[2]} OVER ' + repr(self.field) 

    def __getitem__(self,i):
        return self.new(self.vec[i])

    def __neg__(self):
        return self.new(mod(-self.vec, self.field.p))
    
    def __add__(self, a):
        a = array(a,self.field) if type(a) == int else a
        return self.new(mod(self.vec + a.vec, self.field.p))
    def __radd__(self, a):
        return self.__add__(a)
 
    def __sub__(self, a):
        a = array(a,self.field) if type(a) == int else a
        return self.new(mod(self.vec - a.vec, self.field.p))
    def __rsub__(self, a):
        return self.new(-self.vec).__add__(a)
    
    def __mul__(self, a):
        b = mul45(self.vec,vec2mat(a.vec,a.field),self.field.p) if type(a) == array else mod(self.vec*a,self.field.p)
        return self.new(b)
    def __rmul__(self, a):
        return self.__mul__(a)

    def __pow__(self,n):
        n = n % (self.field.q-1)
        assert n < 2**32
        c0 = n//256**3
        n = n - c0*256**3
        c1 = n//256**2
        n = n - c1*256**2
        c2 = n//256
        c3 = n - c2*256
        bits = jax.numpy.unpackbits(jax.numpy.array([c0,c1,c2,c3], dtype = jax.numpy.uint8))
        bits = jax.numpy.flip(bits)
        def f_xy(x,y):
            return x*y
        def f_y(x,y):
            return y
        x = self
        y = ones(self.shape,self.field)
        for i in range(len(bits)):
            y = jax.lax.cond(bits[i] > 0, f_xy, f_y, x, y)
            x = x*x
        return y

    def __truediv__(self,b):
        return self*b**(self.field.q-2)

    def __matmul__(self, a):
        def matmul(b,c):
            d = vec2mat(c,self.field)
            return matmul34(b,d,self.field.p)
        (ax0,ax1) = (0,0)
        b = self.vec
        c = a.vec
        if len(self.vec) == 1 and len(a.vec) > 1:
            ax0 = None
            b = b.reshape(b.shape[1:])
        if len(a.vec) == 1 and len(self.vec) > 1:
            ax1 = None
            c = c.reshape(c.shape[1:])
        return self.new(jax.vmap(matmul, in_axes = (ax0,ax1))(b,c))

    def lift(self):
        return vec2mat(self.vec, self.field)

    def proj(self):
        return vec2int(self.vec, self.field)

    def t(self):
        return self.new(self.vec.swapaxes(1,2))

    def ravel(self):
        return self.new(self.vec.reshape((self.shape[0],-1,1,self.field.n)))

    def reshape(self,shape):
        return self.new(self.vec.reshape((shape[0],shape[1],shape[2],self.field.n)))
    
    def concatenate(self):
        return self.new(self.vec.swapaxes(0,2))

    def separate(self):
        return self.new(self.vec.swapaxes(0,2))

    def hstack(self, a):
        return self.new(jax.numpy.concatenate([self.vec, a.vec], axis = 2))

    def stack(self, a):
        return self.new(jax.numpy.concatenate([self.vec, a.vec], axis = 0))

    def sum(self):
        return self.new(self.vec.sum(axis = 0, keepdims = True)%self.field.p)

    def mul(self):
        @jax.jit
        def mul_scan(a,b):
            c = matmul34(a,vec2mat(b,self.field),self.field.p)
            return c,c
        vec = jax.lax.scan(mul_scan,eye(self.shape[1],self.field).vec[0],self.vec)[0]
        return self.new(jax.numpy.expand_dims(vec,0))

    def duplicate(self,d):
        b,r,c = self.shape
        n = self.field.n
        return self.new(jax.numpy.tile(self.vec,d).reshape((b,r,c,d,n)).swapaxes(2,3).swapaxes(1,2).reshape((b*d,r,c,n)))

    def vanishes(self):
        def test(a):
            return jax.numpy.count_nonzero(a) == 0
        return jax.vmap(test)(self.vec)

    def trace(self):  
        return self.new(trace4(self.vec,self.field.p))

    def det(self):
        return self.new(mat2vec(det(vec2mat(self.vec,self.field), self.field.inv, self.field.p, BLOCKSIZE)))

    def frb(self):
        return self**self.field.p
   
    def lu(self):
        return mgetrf(unblock(vec2mat(self.vec,self.field)), self.field.inv, BLOCKSIZE)

    def lu_block(self):
        return getrf(vec2mat(self.vec, self.field), self.field.inv, BLOCKSIZE)

    def inv(self):
        return self.new(mat2vec(block(inv(unblock(vec2mat(self.vec,self.field)), self.field.inv, BLOCKSIZE),self.field)))

    def kerim(self):
        k,i,rank,prm = kerim(self.lift(),self.field)
        ker = self.new(mat2vec(k))
        im = self.new(mat2vec(i))
        return ker,im

    def rank(self):
        return kerim(self.lift(),self.field)[2].squeeze()

    def imprm(self):
        k,i,rank,prm = kerim(self.lift(),self.field)
        im = self.new(mat2vec(i))
        return im,prm

    def imrnk(self):
        k,i,rank,prm = kerim(self.lift(),self.field)
        im = self.new(mat2vec(i))
        return im,rank

    def ker(self):
        return self.kerim()[0]

    def im(self):
        return self.kerim()[1]

    def mod(self,b):
        c = self.shape[2]+b.shape[2] 
        def pivcol(piv0):
            mx = jax.numpy.max(piv0)
            zero = jax.numpy.zeros(c, dtype = self.field.dtype)
            sign = jax.numpy.sign(jax.numpy.arange(1, c+1)%c).astype(self.field.dtype)
            bit = jax.numpy.where(mx==c, 1, 0).astype(self.field.dtype)
            return zero.at[piv0-1].set(sign).at[-1].set(bit)
        ba = jax.numpy.concatenate([b.lift(),self.lift()], axis = 2)
        piv = jax.numpy.zeros((len(self.vec),self.vec.shape[2]+b.vec.shape[2]), dtype = INT)
        piv,_,_,_ = gje2((ba,piv),self.field.inv)
        mask = jax.vmap(pivcol)(piv)
        return self.new(mat2vec(mask[:,None,-self.shape[2]:,None,None]*self.lift()))

    def rankmod(self,b):
        c = self.shape[2]+b.shape[2]
        def pivcol(piv0):
            mx = jax.numpy.max(piv0)
            zero = jax.numpy.zeros(c, dtype = self.field.dtype)
            sign = jax.numpy.sign(jax.numpy.arange(1, c+1)%c).astype(self.field.dtype)
            bit = jax.numpy.where(mx==c, 1, 0).astype(self.field.dtype)
            return zero.at[piv0-1].set(sign).at[-1].set(bit)
        ba = jax.numpy.concatenate([b.lift(),self.lift()], axis = 2)
        piv = jax.numpy.zeros((len(self.vec),self.vec.shape[2]+b.vec.shape[2]), dtype = INT)
        piv,_,_,_ = gje2((ba,piv),self.field.inv)
        mask = jax.vmap(pivcol)(piv)
        return jax.numpy.sum(mask[:,-self.shape[2]:], axis = 1).astype(self.field.dtype)

    def dirsum(self,d):
        b = jax.numpy.zeros((self.shape[0],self.shape[1],d.shape[2],self.field.n), dtype = self.field.dtype)
        c = jax.numpy.zeros((self.shape[0],d.shape[1],self.shape[2],self.field.n), dtype = self.field.dtype)
        top = jax.numpy.concatenate([self.vec,b], axis = 2)
        bot = jax.numpy.concatenate([c,d.vec], axis = 2)
        return self.new(jax.numpy.concatenate([top,bot],axis = 1))
    
    def dirprod(self,b):
        return self.new(jax.numpy.einsum('ijkl,imnlr->ijmknr',self.vec, b.lift()).reshape((self.shape[0],self.shape[1]*b.shape[1],self.shape[2]*b.shape[2],self.field.n)))

    def plot(self, title = None, size = 10, dpi = 256, cmap = 'twilight_shifted'):
        title = title if title is not None else self.__repr__()
        return plot(self.vec, title = title, size = size, dpi = dpi, cmap = cmap)

    def plotlift(self, title = None, size = 10, dpi = 256, cmap = 'twilight_shifted'):
        title = title if title is not None else self.__repr__()
        return plot(self.lift().swapaxes(2,3), title = title, size = size, dpi = dpi, cmap = cmap)

    def plotproj(self, title = None, size = 10, dpi = 256, cmap = 'twilight_shifted'):
        title = title if title is not None else self.__repr__()
        return plot(self.proj(), title = title, size = size, dpi = dpi, cmap = cmap)
    
    def plottrace(self, title = None, size = 10, dpi = 256, cmap = 'twilight_shifted'):
        title = title if title is not None else self.__repr__()
        return plot(trace(self.lift(),self.field.p), title = title, size = size, dpi = dpi, cmap = cmap)

# BEGIN REGISTER ARRAY
def flatten_array(a):
    children = (a.vec, a.field)
    aux_data = (a.shape,)
    return (children, aux_data)
def unflatten_array(aux_data, children):
    a = object.__new__(array)
    a.vec, a.field = children
    a.shape, = aux_data
    return a
jax.tree_util.register_pytree_node(array, flatten_array, unflatten_array)
# END REGISTER ARRAY

# BEGIN NAMED ARRAYS
def zeros(shape,field):
    return array(jax.numpy.zeros(shape, dtype = field.dtype), field)
def ones(shape,field):
    return array(jax.numpy.ones(shape, dtype = field.dtype), field)
def arange(n,field):
    return array(jax.numpy.arange(n, dtype = field.dtype).reshape((1,n,1))%field.q, field)
def eye(shape,field):
    return array(jax.numpy.eye(shape, dtype = field.dtype), field)
# END NAMED ARRAYS 

# BEGIN RANDOM ARRAYS
def key(seed = SEED):
    return jax.random.key(seed)
def random(shape, field, seed = SEED): 
    shape = (shape,1,1) if type(shape) == int else (shape[0],1,1) if len(shape) == 1 else (1,shape[0],shape[1]) if len(shape) == 2 else shape
    sample = jax.random.randint(key(seed), shape+(field.n,), 0, field.p, dtype = INT).astype(field.dtype)
    a = object.__new__(array)
    a.field = field
    a.shape = shape
    a.vec = sample
    return a
# END RANDOM ARRAYS
