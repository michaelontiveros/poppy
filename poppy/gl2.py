import jax
import functools
from poppy.constant import INT
from poppy.rep import int2mat, mat2int
from poppy.modular import mod

@functools.partial(jax.jit, static_argnums = (1,2))
def encode(a,q,dtype): # a has shape 2 2 over Zq for q < 256.
    return jax.numpy.sum(a.ravel() * q**jax.numpy.arange(4), dtype = dtype)

@functools.partial(jax.jit, static_argnums = (1,2))
def decode(x,q,dtype): # x is nonnegative and q < 256.
    d = x//q**3
    c = (x-d*q**3)//q**2
    b = (x-d*q**3-c*q**2)//q
    a = (x-d*q**3-c*q**2-b*q)
    return jax.numpy.array([[[a,b],[c,d]]], dtype = dtype)

def gl2(f): # The general linear group GL_2( F ).
    assert f.q < 256
    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q, f.dtype),f)
        return mat2int(mod(a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:],f.p),f).ravel()
    m2c = jax.numpy.arange(f.q**4, dtype = f.dtype)
    m2d = jax.vmap(det)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where(m2d > 0, m2c, 0))[0].astype(INT)
    gl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = INT)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = INT))
    return gl2c, idx

def sl2(f): # The special linear group SL_2( F ).
    assert f.q < 256
    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q, f.dtype),f)
        return mat2int(mod(a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:],f.p),f).ravel()
    m2c = jax.numpy.arange(f.q**4, dtype = f.dtype)
    m2d = jax.vmap(det)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where(m2d == 1, m2c, 0))[0].astype(INT)
    sl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = INT)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = INT))
    return sl2c, idx

def pgl2(f): # The projective general linear group PGL_2( F ).
    assert f.q < 256
    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q, f.dtype),f)
        return mat2int(mod(a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:],f.p),f).ravel()
    @jax.jit
    def norm(x):
        A = decode(x,f.q, f.dtype)[0]
        a, b = A[0,0], A[0,1]
        return (a == 1) | ((a == 0) & (b == 1))
    m2c = jax.numpy.arange(f.q**4, dtype = f.dtype)
    m2d = jax.vmap(det)(m2c).squeeze()
    m2n = jax.vmap(norm)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where((m2d > 0) & m2n, m2c, 0))[0].astype(INT)
    pgl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = INT)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = INT))
    return pgl2c, idx

def pgl2mod(q, dtype): # The projective general linear group PGL_2( Z/q ).
    assert q < 256
    @jax.jit
    def det(x):
        a = decode(x,q, dtype)[0]
        return mod(a[0,0]*a[1,1]-a[0,1]*a[1,0],q)
    @jax.jit
    def norm(x):
        A = decode(x,q, dtype)[0]
        a, b = A[0,0], A[0,1]
        return (a == 1) | ((a == 0) & (b == 1))
    m2c = jax.numpy.arange(q**4, dtype = dtype)
    m2d = jax.vmap(det)(m2c)
    m2n = jax.vmap(norm)(m2c)
    i = jax.numpy.nonzero(jax.numpy.where((m2d > 0) & m2n, m2c, 0))[0].astype(INT)
    pgl2c = m2c[i]
    idx = q**4*jax.numpy.ones(len(m2c), dtype = INT)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = INT))
    return pgl2c, idx

def psl2(f): # The projective special linear group PSL_2( F ).
    assert f.q < 256
    if f.p == 2:
        return pgl2(f)
    @jax.jit
    def det(x):
        a = int2mat(decode(x,f.q, f.dtype),f)
        return mat2int(mod(a[:,0,0,:,:]@a[:,1,1,:,:]-a[:,0,1,:,:]@a[:,1,0,:,:],f.p),f).ravel()
    @jax.jit
    def norm(x):
        A = decode(x,f.q, f.dtype)[0]
        a,b = A[0,0], A[0,1]
        return ((a != 0) & (a < f.q/2)) | ((a == 0) & (b < f.q/2))
    m2c = jax.numpy.arange(f.q**4, dtype = f.dtype)
    m2d = jax.vmap(det)(m2c).squeeze()
    m2n = jax.vmap(norm)(m2c).squeeze()
    i = jax.numpy.nonzero(jax.numpy.where((m2d == 1) & m2n, m2c, 0))[0].astype(INT)
    psl2c = m2c[i]
    idx = f.q**4*jax.numpy.ones(len(m2c), dtype = INT)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = INT))
    return psl2c, idx

def psl2mod(q, dtype): # The projective special linear group PSL_2( Z/q ).
    assert q < 256
    if q == 2:
        return pgl2mod(q)
    @jax.jit
    def det(x):
        a = decode(x,q, dtype)[0]
        return mod(a[0,0]*a[1,1]-a[0,1]*a[1,0],q)
    @jax.jit
    def norm(x):
        A = decode(x,q, dtype)[0]
        a, b = A[0,0], A[0,1]
        return ((a != 0) & (a < q/2)) | ((a == 0) & (b < q/2))
    m2c = jax.numpy.arange(q**4, dtype = dtype)
    m2d = jax.vmap(det)(m2c)
    m2n = jax.vmap(norm)(m2c)
    i = jax.numpy.nonzero(jax.numpy.where((m2d == 1) & m2n, m2c, 0))[0].astype(INT)
    psl2c = m2c[i]
    idx = q**4*jax.numpy.ones(len(m2c), dtype = INT)
    idx = idx.at[i].set(jax.numpy.arange(len(i), dtype = INT))
    return psl2c, idx
