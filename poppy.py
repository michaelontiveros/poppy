# BEGIN POPPY
# BEGIN initialize

import jax
import jax.numpy as jnp
import conway_polynomials
import functools

jax.config.update("jax_enable_x64", True)

# 64 bit integer arrays encode numbers in finite fields.
DTYPE = jnp.int64

# Finite fields are polynomial rings modulo Conway polynomials.
CONWAY = conway_polynomials.database()

# The pseudo random number generator has a default seed.
SEED = 0 

# END initialize
# BEGIN modular arithmetic

@functools.partial( jax.jit, static_argnums = 1 )
def pneg( a, p ):
    return ( -a ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def padd( a, b, p ):
    return ( a + b ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def psub( a, b, p ):
    return ( a - b ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def pmul( a, b, p ):
    return ( a * b ) % p

@functools.partial( jax.jit, static_argnums = 2 )
def pmatmul( a, b, p ):
    return ( a @ b ) % p

pmatmul_vmap = jax.vmap( pmatmul, in_axes = ( None, 0, None ) )

# END modular arithmetic
# BEGIN linear algebra

@functools.partial( jax.jit, static_argnums = 2 )
def ptrsm( a, b, p ): # mod p triangular solve.
  
    M = a.shape[ 0 ]
    N = b.shape[ 1 ]
    I = jnp.eye( N, M, dtype = DTYPE )
    C = jnp.arange( N )
    R = jnp.arange( 1, M )

    @jax.jit 
    def ptrsm_vmap( a1, b1 ):

        @jax.jit
        def ptrsm_scan( a2, b2, i ): 

            @jax.jit
            def f( x, j ):
  
                x = x.at[ j ].set( x[ j ] + ( 1 - x[ j ] ) * ( b2[ j ] - jnp.dot( a2[ j ], x ) ) % p )
  
                return x, x[ j ]  
  
            # scan the rows.
            return jax.lax.scan( f, I[ i ], R )[ 0 ]

        # vmap the columns.
        return jax.vmap( ptrsm_scan, in_axes = ( None, 0, 0 ) )( a1, b1.transpose( ), C ).transpose( )

    return ptrsm_vmap( a, b )

# END linear algebra
# BEGIN field

class field:
    def __init__( self, p, n ):
        
        self.p = p
        self.n = n
        self.q = p ** n 
        self.CONWAY = CONWAY[ p ][ n ]
        self.RANGE  = jnp.arange(  n, dtype = DTYPE )
        self.ONE    = jnp.ones(    n, dtype = DTYPE )
        self.I      = jnp.eye(     n, dtype = DTYPE )
        self.BASIS  = jnp.power( p * self.ONE, self.RANGE )
        #self.BO = jnp.array( [ int( b ) for b in reversed( bin( self.q - 1 )[ 2 : ] ) ] ) 
        ## BO is the binary expansion of the order of the multiplicative group.
        self.X      = self.x( )
        self.INV    = self.inv( )
        
    def x( self ):

        @jax.jit
        def id( a, i ):
            return a

        stack = jax.vmap( id, ( None, 0 ) )
        
        @jax.jit
        def neg( a ):
            return pneg( a, self.p )
        
        @jax.jit
        def matmul( a, b ):
            return pmatmul( a, b, self.p )
        
        @jax.jit
        def x_scan( ):
            
            # V is the vector of subleading coefficients of -1 * ( Conway polynomial ).
            V  = neg( jnp.array( self.CONWAY[ :-1 ], dtype = DTYPE ) )
            
            # M is the companion matrix of the Conway polynomial.
            M  = jnp.zeros( ( self.n, self.n ), dtype = DTYPE ) \
                    .at[ : -1, 1 : ].set( self.I[ 1 : , 1 : ] ) \
                    .at[ -1 ].set( V )
            
            # X is the array of powers of the companion matrix.
            X = jax.lax.associative_scan( matmul, stack( M, self.RANGE ).at[ 0 ].set( self.I ) )
            
            return X

        return x_scan( )
    
    def inv( self ):
        
        @jax.jit
        def mul( a, b ):
            return pmul( a, b, self.p )
        
        @functools.partial( jax.jit, static_argnums = 1 )
        def inv_jit( ABC, i ):
            
            C = mul( ABC[ i - 2, 0 ], ABC[ i - 2, 2 ] )
            ABC =  ABC.at[ i - 1, 2 ].set( C )
            
            return ABC, mul( ABC[ i - 1, 1 ], C )
        
        @jax.jit
        def inv_scan( ):
            
            A  = jnp.arange( 1, self.p, dtype = DTYPE )
            AA = jnp.concatenate( [ self.ONE[ : 1 ], jnp.flip( A[ 1 : ] ) ] )
            B  = jnp.flip( jax.lax.associative_scan( mul, AA ) )
            C  = jnp.ones( self.p - 1, dtype = DTYPE ).at[ 0 ].set( self.p - 1 )
            ABC = jnp.vstack( [ A, B, C ] ).transpose( )
            
            return jnp.concatenate( [ jnp.zeros( 1, dtype = DTYPE ), jax.lax.scan( inv_jit, ABC, A )[ 1 ] ] )
        
        return inv_scan( )

    def __repr__( self ):
        return f'field order  { self.q }.'

# END field
# BEGIN reshape operations

@functools.partial( jax.jit, static_argnums = 1 )
def block( m, f ):
    
    s = m.shape
    n = f.n
    
    return m.reshape( s[ : -2 ] + ( s[ -2 ] // n, n, s[ -1 ] // n, n ) ).swapaxes( -2, -3 )

@functools.partial( jax.jit, static_argnums = 1 )
def ravel( m, f ):
    
    n = f.n
    
    return block( m, f ).reshape( ( -1, n, n ) )

@functools.partial( jax.jit, static_argnums = ( 1, 2 ) )
def unravel( i, f, s ):

    n = f.n    

    return i.reshape( s[ : -2 ] + ( s[ -2 ] // n, s[ -1 ] // n, n, n ) ).swapaxes( -2, -3 ).reshape( s )

# END reshape operations
# BEGIN en/de-coding operations

@functools.partial( jax.jit, static_argnums = 1 )
def i2v( i, f ):
    return jnp.floor_divide( i * f.ONE, f.BASIS ) % f.p

@functools.partial( jax.jit, static_argnums = 1 )
def v2i( v, f ):
    return jnp.sum( v * f.BASIS, dtype = DTYPE )

@functools.partial( jax.jit, static_argnums = 1 )
def v2m( v, f ):
    return jnp.dot( v, f.X ) % f.p

@functools.partial( jax.jit, static_argnums = 1 )
def m2v( m, f ):
    return m[ 0 ]

@functools.partial( jax.jit, static_argnums = 1 )
def i2m( i, f ):
    return v2m( i2v( i, f ), f )

@functools.partial( jax.jit, static_argnums = 1 )
def m2i( m, f ):
    return v2i( m[ 0 ], f )

i2v_vmap = jax.vmap( i2v, in_axes = ( 0, None ) )
v2i_vmap = jax.vmap( v2i, in_axes = ( 0, None ) )
v2m_vmap = jax.vmap( v2m, in_axes = ( 0, None ) )
m2v_vmap = jax.vmap( m2v, in_axes = ( 0, None ) )
i2m_vmap = jax.vmap( i2m, in_axes = ( 0, None ) )
m2i_vmap = jax.vmap( m2i, in_axes = ( 0, None ) )

@functools.partial( jax.jit, static_argnums = 1 )
def lift( i, f ):
    
    m = i2m_vmap( i.ravel( ), f )
    s = i.shape
    n = f.n
    
    return m.reshape( s + ( n, n ) ).swapaxes( -2, -3 ).reshape( s[ : -2 ] + ( s[ -2 ] * n, s[ -1 ] * n ) )

@functools.partial( jax.jit, static_argnums = 1 )
def proj( m, f ):
    
    i = m2i_vmap( ravel( m, f ), f )
    s = m.shape[ : -2 ] + ( m.shape[ -2 ] // f.n, m.shape[ -1 ] // f.n )
    
    return i.reshape( s )

# END en/de-coding operations
# BEGIN array

class array:
    def __init__( self, a, dtype = field( 2, 1 ), lifted = False ):
    
        if len( a.shape ) > 3:
            print( 'ERROR: poppy arrays are three dimensional' )
            return

        self.field = dtype 
        self.shape = ( 1, 1, a.shape[ 0 ] ) if len( a.shape ) == 1 else ( 1, a.shape[ 0 ], a.shape[ 1 ] ) if len( a.shape ) == 2 else a.shape
        self.shape = ( self.shape[ -3 ], self.shape[ -2 ] // self.field.n, self.shape[ -1 ] // self.field.n ) if lifted else self.shape
        self.lift  = a if lifted else lift( a, self.field )
    
    def __mul__( self, a ):
        
        if self.shape[ -1 ] * self.shape[ -2 ] == 1:
            
            b = pmatmul_vmap( self.lift, ravel( a.lift, self.field ), self.field.p ).reshape( a.lift.shape )
            
            return array( b, dtype = self.field, lifted = True )
        
        if a.shape[ -1 ] * a.shape[ -2 ] == 1:   
            
            b = pmatmul_vmap( a.lift, ravel( self.lift, self.field ), self.field.p ).reshape( self.lift.shape )            
            
            return array( b, dtype = self.field, lifted = True )
        
    def __add__( self, a ):
        return array( padd( self.lift, a.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def __sub__( self, a ):
        return array( psub( self.lift, a.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def __neg__( self ):
        return array( pneg( self.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def __matmul__( self, a ):
        return array( pmatmul( self.lift, a.lift, self.field.p ), dtype = self.field, lifted = True )
    
    def inv( self ):
        
        assert len( self.shape ) > 1
        assert self.shape[ -1 ] == self.shape[ -2 ]
        
        @functools.partial( jax.jit, static_argnums = 1 )
        def row_reduce_jit( a, j ):    
            
            mask = jnp.where( jnp.arange( self.lift.shape[ 0 ], dtype = DTYPE ) < j, 0, 1 )
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
            I     = jnp.eye( N, dtype = DTYPE )
            MI    = jnp.hstack( [ self.lift, I ] )
            RANGE = jnp.arange( N, dtype = DTYPE )
            
            return jax.lax.scan( row_reduce_jit, MI, RANGE )[ 0 ][ : , N : ]
        
        INV = inv_jit( )
        
        return array( INV, dtype = self.field, lifted = True )
    
    def reciprocal( self ):
        
        @functools.partial( jax.jit, static_argnums = 1 )
        def row_reduce_jit( a, j ):    
            
            mask = jnp.where( jnp.arange( self.field.n, dtype = DTYPE ) < j, 0, 1 )
            i    = jnp.argmax( mask * a[ : , j ] != 0 )
            
            if a.at[ i, j ] == 0:
                return a, a[ j ]
        
            Rj, Ri = a[ j ], a[ i ] * self.field.INV[ a[ i, j ] ] % self.field.p
            a      = a.at[ i ].set( Rj ).at[ j ].set( Ri )
            a      = ( a - jnp.outer(a[ : , j ].at[ j ] .set( 0 ), a[ j ] ) ) % self.field.p
            
            return a, a[ j ]

        @jax.jit
        def inv_jit( A ):
            
            AI = jnp.hstack( [ A, self.field.I ] )

            return jax.lax.scan( row_reduce_jit, AI, self.field.RANGE )[ 0 ][ : , self.field.n : ] 
            

        inv_vmap = jax.vmap( inv_jit )
        
        R   = ravel( self.lift, self.field )
        INV = unravel( inv_vmap( R ), self.field, self.lift.shape )
        
        return array( INV, dtype = self.field, lifted = True )

    def proj( self ):
        return proj( self.lift, self.field )

    def trace( self ):

        T = jnp.trace( block( self.lift, self.field ) ) % self.field.p
        
        return array( T, dtype = self.field, lifted = True )

    def __repr__( self ):
        return f'array shape { self.shape }.\n' + repr( self.field ) 

# END array
# BEGIN random

SEED = 0

def key( s = SEED ):
    return jax.random.PRNGKey( s )

def random( shape, f, s = SEED ):

    shape = ( 1, 1, shape ) if type( shape ) == int else shape
    
    a = jax.random.randint( key( s ), shape, 0, f.q, dtype = DTYPE )
    
    return array( a, f )

# END random
# END POPPY