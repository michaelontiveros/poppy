# Introduction
`poppy` is a python package for linear algebra over finite fields smaller than `2^40`. 
It has two classes: `field` and `array`. It is written in `jax`.



# Notation
`q = p^n` is a prime power.

`F = F_q` is a finite field.

`M_d( F )` is the ring of `d x d` matrices over `F`.

`y = y_q( x )` is the Conway polynomial.

`X = X_y` is the `n x n` companion matrix.


# Construction
The injective homomorphism `h : F_q --> M_n( F_p )` mapping `f mod y` to `f( X )` 
extends linearly to a homomorphism `H : M_d( F_q ) --> M_nd( F_p )`, reducing linear algebra over finite fields
to linear algebra `mod p`. GPU optimized machine learning libraries like `jax` do linear algebra operations
like large matrix multiply very quickly `mod p`.

# Performance

`F = F1916327294461`.

`a`, `b` are random `222 x 222` matrices over `F`.

`c` is a random number in `F`.

| operation  | time ( T4 GPU ) |
| ------------- | ------------- |
| `poppy.array( a, F )`  | 188 us |
| `a.proj( )`  | 130 us  |
| `a + b`  | 170 us  |
| `a.trace( )` | 1.37 ms |
| `a * c`  | 1.39 ms  |
| `a @ b`  | 2.24 ms  |
| `a.inv( )`  | 482 ms  |

`poppy` uses standard Gaussian elmimination to invert matrices `mod p`.


