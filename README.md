# Introduction
`poppy` is a python package written in `jax` for doing linear algebra over finite fields smaller than `2^50`. 
It has two classes: `field` and `array`.



# Notation
`q = p^n` is a prime power.

`F = F_q` is a finite field.

`g = g_q` is the Conway polynomial.

`X = X_g` is the `n x n` companion matrix.

`I` is the `n x n` identity matrix.

`M_d( F )` is the ring of `d x d` matrices over `F`.

# Construction
The homomorphism `h : F_q --> M_n( F_p )` sending `a_0 +..+ a_(n-1) * x^(n-1)` to `a_0 * I +..+ a_(n-1) * X^(n-1)` 
extends linearly to a homomorphism `H : M_d( F_q ) --> M_nd( F_p )` and reduces linear algebra over finite fields
to linear algebra `mod p`. GPU optimized machine learning libraries like `jax` and `pytorch` do linear algebra operations
like large matrix multiply very quickly `mod p`.
