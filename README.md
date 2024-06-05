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
to linear algebra `mod p`. GPU optimized machine learning libraries like `jax` and `pytorch` do linear algebra operations
like large matrix multiply very quickly `mod p`.

# Performance


