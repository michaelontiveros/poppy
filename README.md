# Introduction
POPPY is a python package for linear algebra over finite fields on the GPU. It has two classes: `field` and `array`. It is written in [JAX](https://github.com/google/jax).



# Notation
`q = p^n` is a prime power in the [Conway polynomials database](https://github.com/sagemath/conway-polynomials).

`y = y_q` is the Conway polynomial. 

`F = F_q` is the finite field.

`M_k( F )` is the associative algebra of `k x k` matrices over `F`.

`X = X_y` is the `n x n` companion matrix.


# Representation
POPPY represents a finite field element `f mod y` by the matrix `f(X) mod p`. The representation is `n` dimensional and faithful. It extends linearly to a faithful `mod p` representation 
of the matrix algebra `M_k( F )`. A matrix `mod p` is a `jax.numpy.int64` array of nonnegative integers less than `p`. The `jax.numpy.mod()` function reduces integer matrices `mod p`.

# Performance

`q = 12421^3`.
 
`a`, `b` are random `222 x 222` matrices over `F`.

`c` is a random number in `F`.

| operation  | time (T4 GPU) |
| ------------- | ------------- |
| `a+b`  | `170 us`  |
| `a.trace()` | `329 us` |
| `a*c`  | `1.4 ms`  |
| `a@b`  | `2.2 ms`  |
| `a.lu()`  | ` 18 ms`  |
| `a.inv()`  | ` 38 ms`  |
| `a.det()`  | ` 42 ms`  |
| `lps(139,103)` | `898 ms` |

