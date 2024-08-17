import jax
import functools
from poppy.constant import DTYPE, BLOCKSIZE, SEED
from poppy.modular import negmod, addmod, submod
from poppy.linear import matmul34, trace, det, mgetrf, getrf, inv, kerim, gje2
from poppy.rep import int2vec, vec2int, vec2mat, mat2vec, block, unblock
from poppy.plot import plot

class array:
    def __init__(self, a, field):
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
        self.field = field 
        self.shape = a.shape
        self.vec = int2vec(a, self.field)

    def new(self,v):
        a = object.__new__(array)
        a.field = self.field
        a.shape = v.shape[:-1]
        a.shape = (v.shape[0],1,1) if len(v.shape) == 2 else (1,v.shape[0],v.shape[1]) if len(v.shape) == 3 else v.shape[:-1]
        a.vec = v
        return a

    def __repr__(self):
        return f'shape {self.shape[0]} {self.shape[1]} {self.shape[2]} over ' + repr(self.field) 

    def __getitem__(self,i):
        return self.new(self.vec[i])

    def __neg__(self):
        return self.new(negmod(self.vec, self.field.p))
    
    def __add__(self, a):
        a = array(a,self.field) if type(a) == int else a
        return self.new(addmod(self.vec, a.vec, self.field.p))
    def __radd__(self, a):
        return self.__add__(a)
 
    def __sub__(self, a):
        a = array(a,self.field) if type(a) == int else a
        return self.new(submod(self.vec, a.vec, self.field.p))
    def __rsub__(self, a):
        return self.__sub__(a)
    
    def __mul__(self, a):
        a = array(a,self.field) if type(a) == int else a
        return self.new((jax.numpy.expand_dims(self.vec,3)@vec2mat(a.vec,a.field))[:,:,:,0,:]%self.field.p)
    def __rmul__(self, a):
        return self.__mul__(a)
    
    def __matmul__(self, a):
        def matmul(b,c):
            return matmul34(b,vec2mat(c,self.field),self.field.p)
        return self.new(jax.vmap(matmul)(self.vec, a.vec))

    def lift(self):
        return vec2mat(self.vec, self.field)

    def proj(self):
        return vec2int(self.vec, self.field)

    def t(self):
        return self.new(self.vec.swapaxes(1,2))

    def vanishes(self):
        def test(a):
            return jax.numpy.count_nonzero(a) == 0
        return jax.vmap(test)(self.vec)

    def trace(self):  
        return self.new(jax.numpy.trace(self.vec, axis1 = 1, axis2 = 2)%self.field.p)

    def det(self):
        return self.new(mat2vec(det(vec2mat(self.vec,self.field), self.field.INV, self.field.p, BLOCKSIZE)))

    def lu(self):
        return mgetrf(unblock(vec2mat(self.vec,self.field)), self.field.INV, BLOCKSIZE)

    def lu_block(self):
        return getrf(vec2mat(self.vec, self.field), self.field.INV, BLOCKSIZE)

    def inv(self):
        return self.new(mat2vec(block(inv(unblock(vec2mat(self.vec,self.field)), self.field.INV, BLOCKSIZE),self.field)))

    def rank(self):
        @jax.jit
        def unique_jit(a):
            return jax.numpy.unique(a, size = self.shape[1], fill_value = -1)
        unique = jax.vmap(unique_jit)
        return jax.numpy.count_nonzero(unique(jax.numpy.argmax(jax.numpy.sign(jax.numpy.max(block(self.lu()[1],self.field).swapaxes(1,2)[:,:,:,0,:],axis = 3)),axis = 1))+1,axis = 1)

    def kerim(self):
        k,i,rank = kerim(self.lift(),self.field)
        ker = self.new(mat2vec(k))
        im = self.new(mat2vec(i))
        return ker,im

    def ker(self):
        return self.kerim()[0]

    def im(self):
        return self.kerim()[1]

    def mod(self,b):
        c = self.shape[2]+b.shape[2]
        def pivcol(piv0):
            mx = jax.numpy.max(piv0)
            return jax.numpy.zeros(c, dtype = DTYPE).at[piv0-1].set(jax.numpy.sign(jax.numpy.arange(1,c+1)%c)).at[-1].set(jax.numpy.where(mx==c,1,0))
        ba = jax.numpy.concatenate([b.lift(),self.lift()], axis = 2)
        piv = jax.numpy.concatenate([self.vec[:,0,:,0]*0, b.vec[:,0,:,0]*0], axis = 1)
        piv,_,_,_ = gje2((ba,piv),self.field.INV)
        mask = jax.vmap(pivcol)(piv)
        return self.new(mat2vec(mask[:,None,-self.shape[2]:,None,None]*self.lift()))

    def rankmod(self,b):
        c = self.shape[2]+b.shape[2]
        def pivcol(piv0):
            mx = jax.numpy.max(piv0)
            return jax.numpy.zeros(c, dtype = DTYPE).at[piv0-1].set(jax.numpy.sign(jax.numpy.arange(1,c+1)%c)).at[-1].set(jax.numpy.where(mx==c,1,0))
        ba = jax.numpy.concatenate([b.lift(),self.lift()], axis = 2)
        piv = jax.numpy.concatenate([self.vec[:,0,:,0]*0, b.vec[:,0,:,0]*0], axis = 1)
        piv,_,_,_ = gje2((ba,piv),self.field.INV)
        mask = jax.vmap(pivcol)(piv)
        return jax.numpy.sum(mask[:,-self.shape[2]:], axis = 1)

    def dirsum(self,d):
        b = jax.numpy.zeros((self.shape[0],self.shape[1],d.shape[2],self.field.n), dtype = DTYPE)
        c = jax.numpy.zeros((self.shape[0],d.shape[1],self.shape[2],self.field.n), dtype = DTYPE)
        top = jax.numpy.concatenate([self.vec,b], axis = 2)
        bot = jax.numpy.concatenate([c,d.vec], axis = 2)
        return self.new(jax.numpy.concatenate([top,bot],axis = 1))
    
    def dirprod(self,b):
        return self.new(jax.numpy.einsum('ijkl,imnlr->ijmknr',self.vec, b.lift()).reshape((self.shape[0],self.shape[1]*b.shape[1],self.shape[2]*b.shape[2],self.field.n)))

    def plot(self, title = None, size = 10):
        title = title if title is not None else self.__repr__().upper()
        return plot(self.vec, title = title, size = size)

    def plotlift(self, title = None, size = 10):
        title = title if title is not None else self.__repr__().upper()
        return plot(self.lift().swapaxes(2,3), title = title, size = size)

    def plotproj(self, title = None, size = 10):
        title = title if title is not None else self.__repr__().upper()
        return plot(self.proj(), title = title, size = size)
    
    def plottrace(self, title = None, size = 10):
        title = title if title is not None else self.__repr__().upper()
        return plot(trace(self.lift(),self.field.p), title = title, size = size)

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
    return array(jax.numpy.zeros(shape, dtype = DTYPE), field)
def ones(shape,field):
    return array(jax.numpy.ones(shape, dtype = DTYPE), field)
def arange(n,field):
    return array(jax.numpy.arange(n, dtype = DTYPE).reshape((1,1,n))%field.q, field)
def eye(shape,field):
    return array(jax.numpy.eye(shape, dtype = DTYPE), field)
# END NAMED ARRAYS 

# BEGIN RANDOM ARRAYS
def key(seed = SEED):
    return jax.random.key(seed)
def random(shape, field, seed = SEED): 
    shape = (shape,1,1) if type(shape) == int else (shape[0],1,1) if len(shape) == 1 else (1,shape[0],shape[1]) if len(shape) == 2 else shape
    sample = jax.random.randint(key(seed), shape+(field.n,), 0, field.p, dtype = DTYPE)
    a = object.__new__(array)
    a.field = field
    a.shape = shape
    a.vec = sample
    return a
# END RANDOM ARRAYS