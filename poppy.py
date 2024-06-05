import math
import conway_polynomials
import jax.numpy  as jnp
import jax.lax    as jlx
from   jax        import jit, random, vmap, config
from   functools  import partial

config.update("jax_enable_x64", True)

CONWAY = conway_polynomials.database()
seed   = 1

@partial( jit, static_argnums = 2 )
def mulmodp( a, b, p ):
    return ( a * b ) % p

@partial( jit, static_argnums = 2 )
def matmulmodp( a, b, p ):
    return ( a @ b ) % p

@partial( jit, static_argnums = 2 )
def addmodp( a, b, p ):
    return ( a + b ) % p

@partial( jit, static_argnums = 2 )
def submodp( a, b, p ):
    return ( a - b ) % p

@partial( jit, static_argnums = 1 )
def negmodp( a, p ):
    return ( -a ) % p

matmulmodp_vmap_im = vmap( matmulmodp, in_axes = ( None, 0, None ) )
matmulmodp_vmap_mi = vmap( matmulmodp, in_axes = ( 0, None, None ) )
matmulmodp_vmap_mm = vmap( matmulmodp, in_axes = ( 0,    0, None ) )

class field:
    def __init__( self, p, n ):
        assert n * math.log( p ) < 50 * math.log( 2 )
        self.p = p
        self.n = n
        self.q = p ** n 
        self.CONWAY  = CONWAY[ p ][ n ]
        self.INDICES = jnp.arange(  n,      dtype = jnp.int64 )
        self.ONE     = jnp.ones(    n,      dtype = jnp.int64 )
        self.ZERO    = jnp.zeros( ( n, n ), dtype = jnp.int64 )
        self.I       = jnp.eye(     n,      dtype = jnp.int64 )
        self.BASIS   = jnp.power( p * self.ONE, self.INDICES  )
        self.X       = self.companion( )
        self.INV     = self.inv( )
        
    def companion( self ):
        
        @jit
        def f( a, b ):
            return matmulmodp( a, b, self.p )
        
        @jit
        def companion_jit( ):
            # X_v is the vector of subleading coefficients of -1 * ( Conway polynomial ).
            X_v  = ( -jnp.array( self.CONWAY[ :-1 ], dtype = jnp.int64 ) ) % self.p

            # X_m is the companion matrix of the Conway polynomial.
            X_m  = self.ZERO.at[ 1:, :-1 ].set( self.I[ 1:, 1: ] ).at[ :, -1 ].set( X_v )
            
            X_l = jnp.array( self.n * [ X_m ] )
            X_l = X_l.at[ 0 ].set( self.I )
            # return an array of powers of the companion matrix.
            return jlx.associative_scan( f, X_l )
        
        return companion_jit( )
    
    def inv( self ):
        
        @jit
        def f( a, b ):
            return mulmodp( a, b, self.p )
        
        @partial( jit, static_argnums = 1 )
        def g( ABCD, i ):
            ABCD = ABCD.at[ i - 1, 2 ].set( f( f( ABCD[ i - 2, 0 ], ABCD[ i - 2, 3 ] ), ABCD[ i - 1, 1 ] ) )
            ABCD = ABCD.at[ i - 1, 3 ].set( f( ABCD[ i - 2, 0 ], ABCD[ i - 2, 3 ] ) )
            return ABCD, ABCD[ i - 1, 2 ]
        
        @jit
        def inv_jit( ):
            A = jnp.arange( 1, self.p, dtype = jnp.int64 )
            AA = jnp.concatenate( [ jnp.ones( 1, dtype = jnp.int64 ), jnp.flip( A[ 1 : ] ) ] )
            B = jnp.flip( jlx.associative_scan( f, AA ) )
            C = jnp.ones( self.p - 1, dtype = jnp.int64 )
            D = jnp.ones( self.p - 1, dtype = jnp.int64 )
            D = D.at[ 0 ].set( self.p - 1 )
            ABCD = jnp.vstack( [ A, B, C, D ] ).transpose( )
            return jnp.concatenate( [ jnp.zeros( 1, dtype = jnp.int64 ), jlx.scan( g, ABCD, A )[ 1 ] ] )
        
        return inv_jit( )
    
    def is_( self, f ):
        return ( self.p is f.p ) and ( self.n is f.n )

@partial( jit, static_argnums = 1 )
def i2v( i, F ):
    return jnp.floor_divide( i * F.ONE, F.BASIS ) % F.p

@partial( jit, static_argnums = 1 )
def v2i( v, F ):
    return jnp.sum( v * F.BASIS, dtype = jnp.int64 )

@partial( jit, static_argnums = 1 )
def v2m( v, F ):
    return jnp.einsum( 'i,ijk -> jk', v, F.X ) % F.p

@partial( jit, static_argnums = 1 )
def m2v( m, F ):
    return m[ : F.n, 0 ]

@partial( jit, static_argnums = 1 )
def i2m( i, F ):
    return v2m( i2v( i, F ), F )

@partial( jit, static_argnums = 1 )
def m2i( m, F ):
    return v2i( m[ : F.n, 0 ], F )

i2v_vmap = vmap( i2v, in_axes = ( 0, None ) )
v2i_vmap = vmap( v2i, in_axes = ( 0, None ) )
v2m_vmap = vmap( v2m, in_axes = ( 0, None ) )
m2v_vmap = vmap( m2v, in_axes = ( 0, None ) )
i2m_vmap = vmap( i2m, in_axes = ( 0, None ) )
m2i_vmap = vmap( m2i, in_axes = ( 0, None ) )

@partial( jit, static_argnums = 1 )
def ravel( m, F ):
    s = m.shape
    return m.reshape( s[ : -1 ] + ( s[ -1 ] // F.n, F.n ) ) \
                   .swapaxes( -2, -3 ) \
                   .reshape( s[ : -2 ] + ( s[ -1 ] // F.n, s[ -2 ] // F.n, F.n, F.n ) ) \
                   .swapaxes( -3, -4 ) \
                   .reshape( ( math.prod( s ) // F.n ** 2 , F.n, F.n ) )

@partial( jit, static_argnums = ( 1, 2 ) )
def unravel( i, F, s ):
    return i.reshape( s[ : -2 ] + ( s[ -2 ] // F.n, s[ -1 ] // F.n ) )  

@partial( jit, static_argnums = 1 )
def lift( i, F ):
    s = i.shape if len( i.shape ) > 1 else ( i.shape[ 0 ], 1 ) if len( i.shape ) == 1 else ( 1, 1 )
    m = i2m_vmap( i.ravel( ), F )
    return m.reshape( s + ( F.n, F.n ) ) \
            .swapaxes( -2, -3 ) \
            .reshape( s[ : -2 ] + ( s[ -2 ] * F.n, s[ -1 ] * F.n ) )

@partial( jit, static_argnums = 1 )
def proj( m, F ):
    s = m.shape
    i = m2i_vmap( ravel( m, F ), F )
    return unravel( i, F, s )

def testliftproj( key, shape, field ):
    ai = random.randint( key, shape, 0, field.q, dtype = jnp.int64 )
    return jnp.nonzero( ai - proj( lift( ai, field ), field ) )

class array:
    def __init__( self, i, dtype = field( 2, 1 ), lifted = False ):
        self.field = dtype 
        self.shape = i.shape[ : -2 ] + ( i.shape[ -2 ] // self.field.n, i.shape[ -1 ] // self.field.n ) if lifted else i.shape
        self.rep   = i if lifted else lift( i, self.field )
        
    def __mul__( self, a ):
        assert a.field is self.field
        
        if self.shape == ( ) or self.shape == ( 1, ) or self.shape == ( 1, 1 ):
            return array( matmulmodp_vmap_im( self.rep, ravel( a.rep, self.field ), self.field.p ).reshape(    a.rep.shape ), dtype = self.field, lifted = True )
        if a.shape    == ( ) or    a.shape == ( 1, ) or    a.shape == ( 1, 1 ):               
            return array( matmulmodp_vmap_mi( ravel( self.rep, self.field ), a.rep, self.field.p ).reshape( self.rep.shape ), dtype = self.field, lifted = True )
        print( 'ERROR in a * b: neither a nor b is scalar.' )
        
    def __add__( self, a ):
        assert a.field is self.field
        assert a.shape == self.shape
        return array( addmodp( self.rep, a.rep, self.field.p ),    dtype = self.field, lifted = True )
    
    def __sub__( self, a ):
        assert a.field is self.field
        assert a.shape == self.shape
        return array( submodp( self.rep, a.rep, self.field.p ),    dtype = self.field, lifted = True )
    
    def __neg__( self ):
        return array( negmodp( self.rep, self.field.p ),           dtype = self.field, lifted = True )
    
    def __matmul__( self, a ):
        assert a.field is self.field
        return array( matmulmodp( self.rep, a.rep, self.field.p ), dtype = self.field, lifted = True )
    
    def inv( self ):
        assert self.rep.shape[ 0 ] == self.rep.shape[ 1 ]
        
        @partial( jit, static_argnums = 1 )
        def row_reduce_jit( a, j ):    
            
            mask = jnp.where( jnp.arange( self.rep.shape[ 0 ], dtype = jnp.int64 ) < j, 0, 1 )
            i    = jnp.argmax( mask * a[ : , j ] != 0 )
            
            if a.at[ i, j ] == 0:
                return a, a[ j ]
        
            rj, ri =      a[ j ],        a[ i ] * self.field.INV[ a[ i, j ] ] % self.field.p
            a      =   a.at[ i ].set( rj ).at[ j ].set( ri )
            A      = ( a - jnp.outer(a[ : , j ].at[ j ] .set( 0 ), a[ j ] ) ) % self.field.p
            
            return A, A[ j ]
        
        @jit
        def inv_jit( ):
            
            N     = self.rep.shape[ 0 ]
            IN    = jnp.eye( N, dtype = jnp.int64 )
            REPI  = jnp.hstack( [ self.rep, IN ] )
            RANGE = jnp.arange( N, dtype = jnp.int64 )
            
            return jlx.scan( row_reduce_jit, REPI, RANGE )[ 0 ][ : , N : ]
        
        INV = inv_jit( )
        return array( INV, dtype = self.field, lifted = True )
    
    def proj( self ):
        return proj( self.rep, self.field )
