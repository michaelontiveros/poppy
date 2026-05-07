## Introduction
POPPY is a [JAX](https://github.com/google/jax) library for linear algebra over finite fields.

## Motivation

Arithmetization compiles computation to combinatorial operations on arithmetic objects like polynomial rings and finitely generated groups. 
Error correcting codes, zero knowledge proofs, and derandomization, all depend on arithmetization. 

Linearization compiles computation to piecewise linear circuits. Hardware is optimized for linear operations.

Modular representation theory linearizes arithmetic programs in a variety of constructible and interesting approximations. Lattice cryptography is linear arithmetic. 

POPPY is designed to run linear arithmetic programs quickly. 

## Notation
- `q = p^n` is a prime power in the [Conway polynomials database](https://github.com/sagemath/conway-polynomials).
- `h = h_q` is the Conway polynomial. 
- `F = F_q` is the finite field.
- `M_k( F )` is the associative algebra of `k x k` matrices over `F`.
- `X = X_h` is an `n x n` matrix root of the polynomial `h`.

## Linear Representation
POPPY represents the finite field element `f mod h` by the matrix `f(X) mod p`. The representation is `n` dimensional and faithful and extends linearly to a faithful `mod p` representation of the matrix algebra `M_k( F )`. A matrix `mod p` is a two-dimensional array of nonnegative integers less than `p`. The `jax.numpy.mod()` function reduces arrays `mod p`.
