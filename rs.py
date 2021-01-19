from re import S
import numpy as np
from numpy.core import atleast_1d
import numpy.core.numeric as NX
import pickle
import os.path

""" 
Irreducible polynomials order 8 for UTF-8 support
https://codyplanteen.com/assets/rs/gf256_prim.pdf
https://codyplanteen.com/assets/rs/gf256_log_antilog.pdf

With the log and anti-log tables we can obviously also derive the binary and polynomial forms.

With this information we are ready to perform the calculative functions required for RS.
"""

# Import the look-up tables for field operations of each of the 256 elements in GF(256)
# Provide lookup data for polynomial/binary form alongside index and decimal, as this will
# enable easier calculations in long devision

# Table 2: 0x11D

if os.path.isfile('.GF256_ANTILOG'):
    with open('.GF256_ANTILOG', 'rb') as fp:
        ANTILOG_TABLE = pickle.load(fp)
else:
    raise FileNotFoundError('Could not load GF256 AntiLog table file')

# x^4 + x + 1 :: [1,0,0,1,1] :: field gen poly

ANTILOG_TABLE = {
    0: 1,
    1: 2,
    2: 4,
    3: 8,
    4: 3,
    5: 6,
    6: 12,
    7: 11,
    8: 5,
    9: 10,
    10: 7,
    11: 14,
    12: 15,
    13: 13,
    14: 9,
}

LOG_TABLE = {a_j: j for j, a_j in ANTILOG_TABLE.items()}

# Code-defining constants
# As they are now, 4 errors can be corrected. This can be adjusted by changing these values.
M = 8
ALPHA = 2
#N = 255
#K = 239
N = 15
K = 11
T = (N - K) // 2
B = 0

# Set the Generator Polynomial and Minimum Primitive Element
# Changing these will mean the rest of the code will function incorrectly, if at all.
# They are the 'unique key' of this particular implementation of Reed-Solomon

GEN_POLY = '100011101'              # polynomial: x^8 + x^4 + x^3 + x^2 + 1 :: 285 :: 0x11D
CODE_GEN_POLY = [1, 59, 13, 104, 189, 68, 209, 30, 8, 163, 65, 41, 229, 98, 50, 36, 59]
                                    # polynomial: x^16 + 59x^15 + 13x^14 + 104x^13 + 189x^12
                                    #             68x^11 + 209x^10 + 30x^9 + 8x^8 + 163x^7
                                    #             65x^6 + 41x^5 + 229x^4 + 98x^3 + 50x^2 + 36x + 59
MIN_PRIM_ELEM = '000000010'         # primitive element (alpha): x :: 1

# Perform addition in the Galois field through bitwise XOR
# num1, num2 must be in decimal form.
# returns decimal form sum.
# since subtraction is identical to addition in GF(256), we don't need a subtract function.
def add(num1, num2):
    if isinstance(num1, int) and isinstance(num2, int):
        return abs(num1) ^ abs(num2)
    else:
        raise TypeError("Numbers must be integers")

# can you make these the same function?

# Perform multiplication within the Galois field using log and anti-log tables mod 255.
# num1, num2 must be in decimal form.
# returns decimal form product.
def multiply(num1, num2):
    if isinstance(num1, int) and isinstance(num2, int):
        j_one, j_two = LOG_TABLE.get(abs(num1), 0), LOG_TABLE.get(abs(num2), 0)
        j = (j_one + j_two) % N
        product = ANTILOG_TABLE.get(j, 0)
        return product
    else:
        raise TypeError("Numbers must be integers")

# Perform division in the Galois field through the log tables.
# num1, num2 must be in the correct order of divison desired: num1 / num2.
# takes two decimal integers num1, num2 and returns a decimal integer product
def divide(num1, num2):
    if isinstance(num1, int) and isinstance(num2, int):
        j_one, j_two = LOG_TABLE.get(abs(num1), 0), LOG_TABLE.get(abs(num2), 0)
        j = (j_one - j_two) % N
        product = ANTILOG_TABLE.get(j, 0)
        return product
    else:
        raise TypeError("Numbers must be integers")

# Visciously stolen from numpy sourcecode and modified to work in GF(256)
def longDivide(u, v):
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = divide(1, int(v[0]))
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m-n+1):
        d = multiply(scale, int(r[k]))
        q[k] = d
        for i, x in enumerate(v):
            v[i] = multiply(d, int(x))
        for j in range(k, k+n+1):
            r[j] = add(int(r[j]), int(v[j-k]))
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    return q, r

# Message will be a two-dimensional array containing k-1 decimal (from 8-bit) symbols.
# returns message*n^(N-K)+remainder=T(x)
def encode(message):
    shift = N-K
    shifted_poly = np.zeros(N+1)
    for index, symbol in enumerate(message): # mumtiply through by x^(n-k) == shift indexes by n-k
        new_index = (index+shift)%N
        shifted_poly[new_index] = symbol
    # now need to divide by code generator polynomial
    quotient, remainder = longDivide(shifted_poly, CODE_GEN_POLY)
    return quotient, np.append(shifted_poly, remainder)

def euclid(f, g):
    # Euclid's algorithm for GCM of polynomials
    q_list, r_list = [], []
    while not np.array_equal(g, [0.]):
        q, r = longDivide(f, g)
        q_list.append(q)
        r_list.append(r)
        f = g
        g = r
    # return the second last remainder = return the last non-zero remainder (gcm)
    return r_list[-2]


def genSyndromes(R_x):
    quotients, syndromes = [], []
    for i in range(B, B+(2*T)):
        Q_i, S_i = longDivide(R_x, [1,ANTILOG_TABLE.get(i, 0)])
        quotients.append(Q_i)
        syndromes.append(S_i)
    quotients, syndromes = np.array(quotients), np.array(syndromes)
    # ensure syndrome equation is written in the correct direction
    # syndromes = np.flip(syndromes)
    if np.count_nonzero(syndromes) != 0:
        f = np.zeros((2*T)+1)
        f[2*T] = 1
        gcd = euclid(f, syndromes)
    else:
        return 0

q, r = longDivide([7,7,9],[2,13])
print(q, r)

#gcd = euclid([1,7,6], [1, 5, 6])
#print(gcd)

#result = add(10, 13)
#print(result)