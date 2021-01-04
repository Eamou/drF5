""" 
Irreducible polynomials order 8 for UTF-8 support
https://codyplanteen.com/assets/rs/gf256_prim.pdf
https://codyplanteen.com/assets/rs/gf256_log_antilog.pdf

With the log and anti-log tables we can obviously also derive the binary and polynomial forms.

With this information we are ready to perform the calculative functions required for RS.
"""

# Set the Generator Polynomial and Minimum Primitive Element
# Changing these will mean the rest of the code will function incorrectly, if at all.
# They are the 'unique key' of this particular implementation of Reed-Solomon

GEN_POLY = '100011101'              # polynomial: x^8 + x^4 + x^3 + x^2 + 1 :: 285 :: 0x11D
MIN_PRIM_ELEM = '000000010'         # primitive element (alpha): x :: 1

# Import the look-up tables for field operations of each of the 256 elements in GF(256)
# Provide lookup data for polynomial/binary form alongside index and decimal, as this will
# enable easier calculations in long devision

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
    128: 133
}