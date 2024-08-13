import jax
import conway_polynomials

# 64 bit integer arrays encode numbers in finite fields.
DTYPE = jax.numpy.int64
# A finite field is a polynomial ring modulo an irreducible polynomial.
POLYNOMIAL = conway_polynomials.database()
# Linear algebra subroutines are blocked.
BLOCKSIZE = 32
# The pseudo random number generator has a default seed.
SEED = 0 

p12 = 3329
p16 = 65521
p22 = 4194191
p30 = 999999733

POLYNOMIAL[p22] = {}
POLYNOMIAL[p30] = {}