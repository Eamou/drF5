from re import S
import numpy as np
from numpy.core import atleast_1d
import numpy.core.numeric as NX

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
GF256_ANTILOG = {
    0: 1,
    1: 2,
    2: 4,
    3: 8,
    4: 16,
    5: 32,
    6: 64,
    7: 128,
    8: 29,
    9: 58,
    10: 116,
    11: 232,
    12: 205,
    13: 135,
    14: 19,
    15: 38,
    16: 76,
    17: 152,
    18: 45,
    19: 90,
    20: 180,
    21: 117,
    22: 234,
    23: 201,
    24: 143,
    25: 3,
    26: 6,
    27: 12,
    28: 24,
    29: 48,
    30: 96,
    31: 192,
    32: 157,
    33: 39,
    34: 78,
    35: 156,
    36: 37,
    37: 74,
    38: 148,
    39: 53,
    40: 106,
    41: 212,
    42: 181,
    43: 119,
    44: 238,
    45: 193,
    46: 159,
    47: 35,
    48: 70,
    49: 140,
    50: 5,
    51: 10,
    52: 20,
    53: 40,
    54: 80,
    55: 160,
    56: 93,
    57: 186,
    58: 105,
    59: 210,
    60: 185,
    61: 111,
    62: 222,
    63: 161,
    64: 95,
    65: 190,
    66: 97,
    67: 194,
    68: 153,
    69: 47,
    70: 94,
    71: 188,
    72: 101,
    73: 202,
    74: 137,
    75: 15,
    76: 30,
    77: 60,
    78: 120,
    79: 240,
    80: 253,
    81: 231,
    82: 211,
    83: 187,
    84: 107,
    85: 214,
    86: 177,
    87: 127,
    88: 254,
    89: 225,
    90: 223,
    91: 163,
    92: 91,
    93: 182,
    94: 113,
    95: 226,
    96: 217,
    97: 175,
    98: 67,
    99: 134,
    100: 17,
    101: 34,
    102: 68,
    103: 136,
    104: 13,
    105: 26,
    106: 52,
    107: 104,
    108: 208,
    109: 189,
    110: 103,
    111: 206,
    112: 129,
    113: 31,
    114: 62,
    115: 124,
    116: 248,
    117: 237,
    118: 199,
    119: 147,
    120: 59,
    121: 118,
    122: 236,
    123: 197,
    124: 151,
    125: 51,
    126: 102,
    127: 204,
    128: 133,
    129: 23,
    130: 46,
    131: 92,
    132: 184,
    133: 109,
    134: 218,
    135: 169,
    136: 79,
    137: 158,
    138: 33,
    139: 66,
    140: 132,
    141: 21,
    142: 42,
    143: 84,
    144: 168,
    145: 77,
    146: 154,
    147: 41,
    148: 82,
    149: 164,
    150: 85,
    151: 170,
    152: 73,
    153: 146,
    154: 57,
    155: 114,
    156: 228,
    157: 213,
    158: 183,
    159: 115,
    160: 230,
    161: 209,
    162: 191,
    163: 99,
    164: 198,
    165: 145,
    166: 63,
    167: 126,
    168: 252,
    169: 229,
    170: 215,
    171: 179,
    172: 123,
    173: 246,
    174: 241,
    175: 255,
    176: 227,
    177: 219,
    178: 171,
    179: 75,
    180: 150,
    181: 49,
    182: 98,
    183: 196,
    184: 149,
    185: 55,
    186: 110,
    187: 220,
    188: 165,
    189: 87,
    190: 174,
    191: 65,
    192: 130,
    193: 25,
    194: 50,
    195: 100,
    196: 200,
    197: 141,
    198: 7,
    199: 14,
    200: 28,
    201: 56,
    202: 112,
    203: 224,
    204: 221,
    205: 167,
    206: 83,
    207: 166,
    208: 81,
    209: 162,
    210: 89,
    211: 178,
    212: 121,
    213: 242,
    214: 249,
    215: 239,
    216: 195,
    217: 155,
    218: 43,
    219: 86,
    220: 172,
    221: 69,
    222: 138,
    223: 9,
    224: 18,
    225: 36,
    226: 72,
    227: 144,
    228: 61,
    229: 122,
    230: 244,
    231: 245,
    232: 247,
    233: 243,
    234: 251,
    235: 235,
    236: 203,
    237: 139,
    238: 11,
    239: 22,
    240: 44,
    241: 88,
    242: 176,
    243: 125,
    244: 250,
    245: 233,
    246: 207,
    247: 131,
    248: 27,
    249: 54,
    250: 108,
    251: 216,
    252: 173,
    253: 71,
    254: 142,
    255: 0
}

GF256_LOG = {a_j: j for j, a_j in GF256_ANTILOG.items()}

# Code-defining constants
# As they are now, 4 errors can be corrected. This can be adjusted by changing these values.
M = 8
ALPHA = 2
N = 255
K = 239
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

# Perform multiplication within the Galois field using log and anti-log tables mod 255.
# num1, num2 must be in decimal form.
# returns decimal form product.
def multiply(num1, num2):
    if isinstance(num1, int) and isinstance(num2, int):
        j_one, j_two = GF256_LOG[abs(num1)], GF256_LOG[abs(num2)]
        j = (j_one + j_two) % N
        product = GF256_ANTILOG[j]
        return product
    else:
        raise TypeError("Numbers must be integers")

# Perform addition in the Galois field through bitwise XOR
# num1, num2 must be in decimal form.
# returns decimal form sum.
# since subtraction is identical to addition in GF(256), we don't need a subtract function.
def add(num1, num2):
    if isinstance(num1, int) and isinstance(num2, int):
        return num1 ^ num2
    else:
        raise TypeError("Numbers must be integers")

# Perform division in the Galois field through the log tables.
# num1, num2 must be in the correct order of divison desired: num1 / num2.
# takes two decimal integers num1, num2 and returns a decimal integer product
def divide(num1, num2):
    if isinstance(num1, int) and isinstance(num2, int):
        j_one, j_two = GF256_LOG[abs(num1)], GF256_LOG[abs(num2)]
        j = (j_one - j_two) % N
        product = GF256_ANTILOG[j]
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
    quotient, remainder = longDivide(f, g)
    quotient_list = [quotient]
    remainder_list = [remainder]
    while remainder != [0.]:
        quotient, remainder = longDivide(g, remainder)
        quotient_list.append(quotient)
        remainder_list.append(remainder)
        g = remainder
        remainder = remainder_list[-2]
    if remainder_list[-1] == [0.]:
        return remainder_list[-2]
    else:
        return remainder_list[-1]

def genSyndromes(R_x):
    quotients, syndromes = [], []
    for i in range(B, B+(2*T)):
        Q_i, S_i = longDivide(R_x, [1,GF256_ANTILOG[i]])
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

q, r = longDivide([1, 2, 1], [1, 2, 1])
print(q, r)