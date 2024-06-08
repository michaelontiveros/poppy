import jax
import jax.numpy as jnp
import conway_polynomials
import functools

jax.config.update("jax_enable_x64", True)

CONWAY = conway_polynomials.database()
SEED   = 1

@jax.jit
def id( a, i ):
    return a

stack = jax.vmap( id, ( None, 0 ) )

@functools.partial( jax.jit, static_argnums = 2 )
def mulmodp( a, b, p ):
    return ( a * b ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def matmulmodp( a, b, p ):
    return ( a @ b ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def addmodp( a, b, p ):
    return ( a + b ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def submodp( a, b, p ):
    return ( a - b ) % p

@functools.partial( jax.jit, static_argnums = 1 )
def negmodp( a, p ):
    return ( -a ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def matmulmodp( a, b, p ):
    return ( a @ b ) % p

matmulmodp_vmap_im = jax.vmap( matmulmodp, in_axes = ( None, 0, None ) )
matmulmodp_vmap_mi = jax.vmap( matmulmodp, in_axes = ( 0, None, None ) )
matmulmodp_vmap_mm = jax.vmap( matmulmodp, in_axes = ( 0,    0, None ) )

class field:
    def __init__( self, p, n ):
        
        self.p = p
        self.n = n
        self.q = p ** n 
        self.CONWAY = CONWAY[ p ][ n ]
        self.RANGE  = jnp.arange(  n,      dtype = jnp.int64 )
        self.ONE    = jnp.ones(    n,      dtype = jnp.int64 )
        self.ZERO   = jnp.zeros( ( n, n ), dtype = jnp.int64 )
        self.I      = jnp.eye(     n,      dtype = jnp.int64 )
        self.BASIS  = jnp.power( p * self.ONE, self.RANGE )
        self.X      = self.companion( )
        self.INV    = self.inv( )
        
    def companion( self ):
        
        @jax.jit
        def matmul( a, b ):
            return matmulmodp( a, b, self.p )
        
        @jax.jit
        def companion_jit( ):
            
            # V is the vector of subleading coefficients of -1 * ( Conway polynomial ).
            V  = ( -jnp.array( self.CONWAY[ :-1 ], dtype = jnp.int64 ) ) % self.p
            
            # M is the companion matrix of the Conway polynomial.
            M  = self.ZERO.at[ : -1, 1 : ].set( self.I[ 1 : , 1 : ] ).at[ -1 ].set( V )
            
            # X is the array of powers of the companion matrix.
            X = jax.lax.associative_scan( matmul, stack( M, self.RANGE ).at[ 0 ].set( self.I ) )
            
            return X

        return companion_jit( )
    
    def inv( self ):
        
        @jax.jit
        def prod( a, b ):
            return mulmodp( a, b, self.p )
        
        @functools.partial( jax.jit, static_argnums = 1 )
        def INV( ABC, i ):
            
            C = prod( ABC[ i - 2, 0 ], ABC[ i - 2, 2 ] )
            ABC =  ABC.at[ i - 1, 2 ].set( C )
            
            return ABC, prod( ABC[ i - 1, 1 ], C )
        
        @jax.jit
        def inv_jit( ):
            
            A  = jnp.arange( 1, self.p, dtype = jnp.int64 )
            AA = jnp.concatenate( [ jnp.ones( 1, dtype = jnp.int64 ), jnp.flip( A[ 1 : ] ) ] )
            B  = jnp.flip( jax.lax.associative_scan( prod, AA ) )
            C  = jnp.ones( self.p - 1, dtype = jnp.int64 ).at[ 0 ].set( self.p - 1 )
            ABC = jnp.vstack( [ A, B, C ] ).transpose( )
            
            return jnp.concatenate( [ jnp.zeros( 1, dtype = jnp.int64 ), jax.lax.scan( INV, ABC, A )[ 1 ] ] )
        
        return inv_jit( )

    def __repr__( self ):
        return f'field order { self.q }.'


@functools.partial( jax.jit, static_argnums = 1 )
def i2v( i, F ):
    return jnp.floor_divide( i * F.ONE, F.BASIS ) % F.p

@functools.partial( jax.jit, static_argnums = 1 )
def v2i( v, F ):
    return jnp.sum( v * F.BASIS, dtype = jnp.int64 )

@functools.partial( jax.jit, static_argnums = 1 )
def v2m( v, F ):
    return jnp.einsum( 'i,ijk -> jk', v, F.X ) % F.p

@functools.partial( jax.jit, static_argnums = 1 )
def m2v( m, F ):
    return m[ 0 ]

@functools.partial( jax.jit, static_argnums = 1 )
def i2m( i, F ):
    return v2m( i2v( i, F ), F )

@functools.partial( jax.jit, static_argnums = 1 )
def m2i( m, F ):
    return v2i( m[ 0 ], F )

i2v_vmap = jax.vmap( i2v, in_axes = ( 0, None ) )
v2i_vmap = jax.vmap( v2i, in_axes = ( 0, None ) )
v2m_vmap = jax.vmap( v2m, in_axes = ( 0, None ) )
m2v_vmap = jax.vmap( m2v, in_axes = ( 0, None ) )
i2m_vmap = jax.vmap( i2m, in_axes = ( 0, None ) )
m2i_vmap = jax.vmap( m2i, in_axes = ( 0, None ) )

@functools.partial( jax.jit, static_argnums = 1 )
def block( m, F ):
    
    s = m.shape
    n = F.n
    
    return m.reshape( s[ : -1 ] + ( s[ -1 ] // n, n ) ) \
            .swapaxes( -2, -3 ) \
            .reshape( s[ : -2 ] + ( s[ -1 ] // n, s[ -2 ] // n, n, n ) ) \
            .swapaxes( -3, -4 ) 

@functools.partial( jax.jit, static_argnums = 1 )
def ravel( m, F ):
    
    n = F.n
    
    return block( m, F ).reshape( ( -1, n, n ) )

@functools.partial( jax.jit, static_argnums = ( 1, 2 ) )
def unravel( i, F, s ):
    return i.reshape( s[ : -2 ] + ( s[ -2 ] // F.n, s[ -1 ] // F.n ) )  

@functools.partial( jax.jit, static_argnums = 1 )
def lift( i, F ):
    
    s = i.shape if len( i.shape ) > 1 else ( i.shape[ 0 ], 1 ) if len( i.shape ) == 1 else ( 1, 1 )
    m = i2m_vmap( i.ravel( ), F )
    n = F.n
    
    return m.reshape( s + ( n, n ) ) \
            .swapaxes( -2, -3 ) \
            .reshape( s[ : -2 ] + ( s[ -2 ] * n, s[ -1 ] * n ) )

@functools.partial( jax.jit, static_argnums = 1 )
def proj( m, F ):
    
    s = m.shape
    i = m2i_vmap( ravel( m, F ), F )
    
    return unravel( i, F, s )

def testliftproj( key, shape, field ):

    ai = jax.random.randint( key, shape, 0, field.q, dtype = jnp.int64 )
    
    return jnp.nonzero( ai - proj( lift( ai, field ), field ) )

class array:
    def __init__( self, i, dtype = field( 2, 1 ), lifted = False ):
    
        self.field = dtype 
        self.shape = i.shape[ : -2 ] + ( i.shape[ -2 ] // self.field.n, i.shape[ -1 ] // self.field.n ) if lifted else i.shape
        self.lift  = i if lifted else lift( i, self.field )
        
    def __mul__( self, a ):
        
        if self.shape == ( ) or self.shape == ( 1, ) or self.shape == ( 1, 1 ):
            return array( matmulmodp_vmap_im( self.lift, ravel( a.lift, self.field ), self.field.p ).reshape(    a.lift.shape ), dtype = self.field, lifted = True )
        
        if a.shape    == ( ) or    a.shape == ( 1, ) or    a.shape == ( 1, 1 ):               
            return array( matmulmodp_vmap_mi( ravel( self.lift, self.field ), a.lift, self.field.p ).reshape( self.lift.shape ), dtype = self.field, lifted = True )
        
    def __add__( self, a ):
        return array( addmodp( self.lift, a.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def __sub__( self, a ):
        return array( submodp( self.lift, a.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def __neg__( self ):
        return array( negmodp( self.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def __matmul__( self, a ):
        return array( matmulmodp( self.lift, a.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def inv( self ):
        
        assert len( self.shape ) > 1
        assert self.shape[ -1 ] == self.shape[ -2 ]
        
        @functools.partial( jax.jit, static_argnums = 1 )
        def row_reduce_jit( a, j ):    
            
            mask = jnp.where( jnp.arange( self.lift.shape[ 0 ], dtype = jnp.int64 ) < j, 0, 1 )
            i    = jnp.argmax( mask * a[ : , j ] != 0 )
            
            if a.at[ i, j ] == 0:
                return a, a[ j ]
        
            Rj, Ri = a[ j ], a[ i ] * self.field.INV[ a[ i, j ] ] % self.field.p
            a      = a.at[ i ].set( Rj ).at[ j ].set( Ri )
            A      = ( a - jnp.outer(a[ : , j ].at[ j ] .set( 0 ), a[ j ] ) ) % self.field.p
            
            return A, A[ j ]
        
        @jax.jit
        def inv_jit( ):
            
            N     = self.lift.shape[ 0 ]
            I     = jnp.eye( N, dtype = jnp.int64 )
            MI    = jnp.hstack( [ self.lift, I ] )
            RANGE = jnp.arange( N, dtype = jnp.int64 )
            
            return jax.lax.scan( row_reduce_jit, MI, RANGE )[ 0 ][ : , N : ]
        
        INV = inv_jit( )
        
        return array( INV, dtype = self.field, lifted = True )
    
    def proj( self ):
        return proj( self.lift, self.field )

    def trace( self ):

        T = jnp.trace( block( self.lift, self.field ) ) % self.field.p
        
        return array( T, dtype = self.field, lifted = True )

    def __repr__( self ):
        return f'array shape { self.shape }.\n' + repr( self.field ) 

def random( shape, F, seed = SEED ):
    
    a = jax.random.randint( jax.random.PRNGKey( seed ), shape, 0, F.q, dtype = jnp.int64 )
    
    return array( a, F )
