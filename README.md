# Introduction
`poppy` is a python package for linear algebra over finite fields on the GPU. It has two classes: `field` and `array`. It is written in `jax`.



# Notation
`q = p^n` is a prime power in the [Conway polynomials database](https://github.com/sagemath/conway-polynomials).

`y = y_q` is the Conway polynomial. 

`F = F_q` is the finite field.

`M_d( F )` is the ring of `d x d` matrices over `F`.

`X = X_y` is the `n x n` companion matrix.


# Construction
The injective homomorphism from `F_q` to `M_n( F_p )` that sends `f mod y` to `f(X)`,
extends linearly to an injective homomorphism from `M_d( F_q )` to `M_nd( F_p )`, reducing linear algebra over finite fields
to linear algebra `mod p`. GPU optimized machine learning libraries like `jax` do linear algebra operations
like large matrix multiply very quickly `mod p`.

# Performance

`q = 12421^3`.

`c` is a random number in `F`.
 
`a`, `b` are random `222 x 222` matrices over `F`.

| operation  | time (T4 GPU) |
| ------------- | ------------- |
| `poppy.array(a, F)`  | `188 us` |
| `a.proj()`  | `130 us`  |
| `a + b`  | `170 us`  |
| `a.trace()` | `329 us` |
| `a * c`  | `1.4 ms`  |
| `a @ b`  | `2.2 ms`  |
| `a.lu()`  | `18  ms`  |
| `a.det()`  | `18  ms`  |
| `a.inv()`  | `38  ms`  |


