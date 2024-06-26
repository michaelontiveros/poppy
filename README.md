# Introduction
`poppy` is a python package for linear algebra over finite fields on the GPU. It has two classes: `field` and `array`. It is written in `jax`.



# Notation
`q = p^n` is a prime power in the [Conway polynomials database](https://github.com/sagemath/conway-polynomials).

`y = y_q` is the Conway polynomial. 

`F = F_q` is the finite field.

`M_d( F )` is the ring of `d x d` matrices over `F`.

`X = X_y` is the `n x n` companion matrix.


# Construction
The homomorphism from `F` to `M_n( F_p )`, sending `f mod y` to `f(X)`, is injective and extends linearly to a faithful representation of `M_d( F )` in `M_nd( F_p )`, reducing linear algebra over finite fields
to linear algebra `mod p`. GPU optimized machine learning libraries like `jax` do linear algebra operations
like large matrix multiply very quickly `mod p`.

# Performance

`q = 12421^3`.

`c` is a random number in `F`.
 
`a`, `b` are random `222 x 222` matrices over `F`.

| operation  | time (T4 GPU) |
| ------------- | ------------- |
| `a + b`  | `170 us`  |
| `a.trace()` | `329 us` |
| `a * c`  | `1.4 ms`  |
| `a @ b`  | `2.2 ms`  |
| `a.lu()`  | `18  ms`  |
| `a.inv()`  | `38  ms`  |
| `a.det()`  | `42  ms`  |

