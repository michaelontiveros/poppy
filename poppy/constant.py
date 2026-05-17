import jax
import conway_polynomials

# 32 bit integers index arrays.
INT = jax.numpy.int32
# A finite field is a polynomial ring modulo an irreducible polynomial.
POLYNOMIAL = conway_polynomials.database()
# Linear algebra subroutines are blocked.
BLOCKSIZE = 32
# The pseudo random number generator has a default seed.
SEED = 0 
# Plot has a default colormap.
CMAP = 'twilight_shifted'

p12 = 3329
p16 = 65521
p22 = 4194191
p30 = 999999733

POLYNOMIAL[p22] = {}
POLYNOMIAL[p30] = {}
