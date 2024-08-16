# Introduction
POPPY is a [JAX](https://github.com/google/jax) library for linear algebra over finite fields. There is a `field` class, and an `array` class. 

An `array` is a batch of matrices over a `field`; it has `3` dimensions. It is stored as a `4`-dimensional `jax.numpy.int64` array. It is lifted to a `5`-dimensional linear representation before multiplicative operations. The length of the `4th` and `5th` dimension is the degree of the field.



# Notation
`q = p^n` is a prime power in the [Conway polynomials database](https://github.com/sagemath/conway-polynomials).

`y = y_q` is the Conway polynomial. 

`F = F_q` is the finite field.

`M_k( F )` is the associative algebra of `k x k` matrices over `F`.

`X = X_y` is an `n x n` matrix root of the polynomial `y`.


# Linear Representation
POPPY represents the finite field element `f mod y` by the matrix `f(X) mod p`. The representation is `n` dimensional and faithful. It extends linearly to a faithful `mod p` representation 
of the matrix algebra `M_k( F )`. A matrix `mod p` is a `2`-dimensional `jax.numpy.int64` array of nonnegative integers less than `p`. The `jax.numpy.mod()` function reduces integer arrays `mod p`.

# Performance

`q = 12421^3`.
 
`a,b` are random `222 x 222` matrices over `F`.

`c` is a random number in `F`.

| operation  | time (T4 GPU) |
| ------------- | ------------- |
| `a+b`  | `200 us`  |
| `a.trace()` | `300 us` |
| `a*c`  | `1.4 ms`  |
| `a@b`  | `2.8 ms`  |
| `a.lu()`  | ` 20 ms`  |
| `a.inv()`  | ` 40 ms`  |
| `a.det()`  | ` 50 ms`  |
