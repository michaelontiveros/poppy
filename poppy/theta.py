import jax
import functools
from poppy.constant import DTYPE

class theta: # N qudit stabilizer state over an odd prime field.
    def __init__(self, N, field, seed = 0):
        assert field.p > 2
        self.A = jax.numpy.eye(   N,    dtype = DTYPE) # Matrix.
        self.B = jax.numpy.zeros((N,N), dtype = DTYPE) # Matrix.
        self.C = jax.numpy.zeros((N,N), dtype = DTYPE) # Matrix.
        self.D = jax.numpy.eye(   N,    dtype = DTYPE) # Matrix.
        self.u = jax.numpy.zeros((N,),  dtype = DTYPE) # Vector
        self.v = jax.numpy.zeros((N,),  dtype = DTYPE) # Vector.
        self.r = jax.random.randint(jax.random.key(seed), (N*N,), 0,field.p, dtype = DTYPE)
        self.m = jax.numpy.ones((N*N,2), dtype = jax.numpy.int64)*N # Measurements.
        self.t = 0                                                  # Time.
        self.field = field

    def new(self, A,B,C,D,u,v):
        th = object.__new__(theta)
        th.u, th.A, th.B = u,A,B
        th.v, th.C, th.D = v,C,D
        th.r, th.m, th.t = self.r, self.m, self.t
        th.field = self.field
        return th

    def __repr__(self):
        return f'{len(self.A)} QUDIT STABILIZER STATE OVER ' + repr(self.field) 

    def Z(self, i,n=1): # Pauli Z gate n times on i.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        u = self.u.at[:].add((n*self.B[:,i].T).sum(axis = 0))%self.field.p
        v = self.v.at[:].add((n*self.D[:,i].T).sum(axis = 0))%self.field.p
        return self.new(self.A, self.B, self.C, self.D, u,v)

    def X(self, i,n=1): # Pauli X gate n times on i.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        u = self.u.at[:].add((-n*self.A[:,i].T).sum(axis = 0))%self.field.p
        v = self.v.at[:].add((-n*self.C[:,i].T).sum(axis = 0))%self.field.p
        return self.new(self.A, self.B, self.C, self.D, u,v)

    def Xl(self, i,n=1): # Pauli X gate n times on i on the left.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        u = self.u.at[i].add(-n.T.squeeze())%self.field.p
        return self.new(self.A, self.B, self.C, self.D, u, self.v)

    def Q(self, i,n=1): # Quadratic phase gate n times on i.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        A = self.A.at[:,i].add((n.T*self.B[:,i]).squeeze())%self.field.p
        C = self.C.at[:,i].add((n.T*self.D[:,i]).squeeze())%self.field.p
        return self.new(A, self.B, C, self.D, self.u, self.v)

    def Ql(self, i,n=1): # Quadratic phase gate n times on i on the left.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        C = self.C.at[i].add((n*self.A[i]).squeeze())%self.field.p
        D = self.D.at[i].add((n*self.B[i]).squeeze())%self.field.p
        v = self.v.at[i].add((n.T*self.u[i]).squeeze())%self.field.p
        return self.new(self.A, self.B, C,D, self.u, v)
    
    def S(self, i,j,n=1): # Sum gate n times on ij.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        A = self.A.at[:,i].add((-n.T*self.A[:,j]).squeeze())%self.field.p
        B = self.B.at[:,j].add(( n.T*self.B[:,i]).squeeze())%self.field.p
        C = self.C.at[:,i].add((-n.T*self.C[:,j]).squeeze())%self.field.p
        D = self.D.at[:,j].add(( n.T*self.D[:,i]).squeeze())%self.field.p
        return self.new(A,B,C,D, self.u, self.v)

    def Sl(self, i,j,n=1): # Sum gate n times on ij on the left.
        n = jax.numpy.array(n, dtype = DTYPE).reshape(-1,1)
        A = self.A.at[i].add((-n*self.A[j]).squeeze())%self.field.p
        B = self.B.at[i].add((-n*self.B[j]).squeeze())%self.field.p
        C = self.C.at[j].add(( n*self.C[i]).squeeze())%self.field.p
        D = self.D.at[j].add(( n*self.D[i]).squeeze())%self.field.p
        u = self.u.at[i].add((-n.T*(self.u[j])).sum(axis = 1))%self.field.p
        v = self.v.at[j].add(( n.T*(self.v[i])).sum(axis = 1))%self.field.p
        return self.new(A,B,C,D,u,v)

    def F(self, i): # Fourier gate on i.
        A = self.A.at[:,i].set( self.B[:,i])%self.field.p
        B = self.B.at[:,i].set(-self.A[:,i])%self.field.p
        C = self.C.at[:,i].set( self.D[:,i])%self.field.p
        D = self.D.at[:,i].set(-self.C[:,i])%self.field.p
        return self.new(A,B,C,D, self.u, self.v)

    def Fl(self, i): # Fourier gate on i on the left.
        A = self.A.at[i].set(-self.C[i])%self.field.p
        B = self.B.at[i].set(-self.D[i])%self.field.p
        C = self.C.at[i].set( self.A[i])%self.field.p
        D = self.D.at[i].set( self.B[i])%self.field.p
        u = self.u.at[i].set(-self.v[i])%self.field.p
        v = self.v.at[i].set( self.u[i])%self.field.p
        return self.new(A,B,C,D,u,v)

    def M1(self, i): # Measure one i.
        N = len(self.A)
        j = self.B[:,i].argmax()
        b = self.field.INV[self.B[j,i]]
        def collapse(th):
            ks = jax.numpy.where(jax.numpy.arange(N) != j, size = N-1)
            js = j*jax.numpy.ones(N-1, dtype = jax.numpy.int64)   
            n = (b*th.B[ks,i])%self.field.p  
            th = th.Sl(ks,js,n)      # Eliminate all but the jth number in the ith column of B.
            n = (-b*th.D[j,i])%self.field.p 
            th = th.Ql(j,n)          # Eliminate the jth number in the ith column of D.
            th = th.Fl(j)            # Collapse i.
            th = th.Xl(i,th.r[th.t]) # Randomize i.
            return th
        def identity(th): return th
        th = jax.lax.cond(b,collapse,identity,self)
        m = (-th.u@th.D[:,i])%self.field.p # Measure i.
        th.m = th.m.at[th.t].set(jax.numpy.array([i,m]))
        th.t = th.t+1
        return th

    def M(self, i): # Measure i.
        i = jax.numpy.array(i).reshape(-1,)
        def M_scan(th,k): return th.M1(k),k
        return jax.lax.scan(M_scan,self,i)[0]

    def MR1(self, i): # Measure then reset one i.
        th = self.M1(i)
        th = th.X(i,(-th.m[th.t-1,1])%th.field.p)
        return th

    def MR(self, i): # Measure then reset i.
        i = jax.numpy.array(i).reshape(-1,)
        def MR_scan(th,k): return th.MR1(k),k
        return jax.lax.scan(MR_scan,self,i)[0]

# BEGIN REGISTER THETA
def flatten_theta(th):
    children = (th.A, th.B, th.C, th.D, th.u, th.v, th.r, th.m, th.t)
    aux_data = (th.field,)
    return (children, aux_data)
def unflatten_theta(aux_data, children):
    th = object.__new__(theta)
    th.A, th.B, th.C, th.D, th.u, th.v, th.r, th.m, th.t = children
    th.field, = aux_data
    return th
jax.tree_util.register_pytree_node(theta, flatten_theta, unflatten_theta)
# END REGISTER THETA