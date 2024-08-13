import jax
jax.config.update("jax_enable_x64", True)

from poppy.constant import DTYPE, POLYNOMIAL, BLOCKSIZE, SEED
from poppy.field import field
from poppy.array import array, zeros, ones, eye, random
from poppy.linear import mtrsm, trsm, mgetrf, getrf, gje2, transpose, tracemod, invmod, kerim
from poppy.rep import block, unblock, int2vec, vec2int, int2mat, mat2int, vec2mat, mat2vec
from poppy.ring import Z2, M2
from poppy.group import gl2, sl2, pgl2, pgl2mod, psl2, psl2mod, decode
from poppy.expander import lps
from poppy.topology import polygon, graph, is_boundary, homology, betti, euler_characteristic
from poppy.plot import plot