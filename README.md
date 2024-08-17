## Introduction
POPPY is a [JAX](https://github.com/google/jax) library for linear algebra over finite fields.

POPPY has a `field` class, and an `array` class. 

POPPY `arrays` are `3`-dimensional; they are batches of matrices over `fields`.

## Motivation

Arithmetization compiles a computation to a set of combinatorial operations on arithmetic objects (polynomial rings, finitely generated groups, ...). 
Emerging technologies like zero knowledge proofs, error correcting codes and derandomization, depend on arithmetization. 

Linearization approximates a computation with a piecewise linear circuit. It is the basis of modern AI. 
Computer hardware is optimized for linear operations.

Modular representation theory linearizes arithmetic programs automatically, in a variety of constructible and interesting approximations. 
The resulting program is piecewise linear, over a finite field.
Linear operations over finite fields have become slower than linear operations over floating point numbers. POPPY closes the performance gap.

## Notation
`q = p^n` is a prime power in the [Conway polynomials database](https://github.com/sagemath/conway-polynomials).

`y = y_q` is the Conway polynomial. 

`F = F_q` is the finite field.

`M_k( F )` is the associative algebra of `k x k` matrices over `F`.

`X = X_y` is an `n x n` matrix root of the polynomial `y`.

## Linear Representation
POPPY represents the finite field element `f mod y` by the matrix `f(X) mod p`. The representation is `n` dimensional and faithful. 
It extends linearly to a faithful `mod p` representation 
of the matrix algebra `M_k( F )`. A matrix `mod p` is a `2`-dimensional `jax.numpy.int64` array of nonnegative integers less than `p`. 
The `jax.numpy.mod()` function reduces integer arrays `mod p`.

## Performance

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
