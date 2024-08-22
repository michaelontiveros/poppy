import jax
jax.config.update("jax_enable_x64", True)
from poppy.constant import DTYPE, POLYNOMIAL, BLOCKSIZE, SEED
from poppy.field import field
from poppy.array import array, zeros, ones, arange, eye, random
from poppy.modular import addmod, submod, mulmod
from poppy.linear import trace, outer33, outer44, matmul55, mtrsm, trsm, mgetrf, getrf, gje2, inv, kerim
from poppy.rep import transpose, block, unblock, int2vec, vec2int, int2mat, mat2int, vec2mat, mat2vec, rep
from poppy.ring import Z2, M2
from poppy.group import gl2, sl2, pgl2, pgl2mod, psl2, psl2mod, decode
from poppy.expander import lps
from poppy.topology import polygon, graph, boundary, homology, betti, euler
from poppy.plot import plot