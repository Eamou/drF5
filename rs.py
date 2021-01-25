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

#GEN_POLY = [1,0,0,0,1,1,1,0,1]      # polynomial: x^8 + x^4 + x^3 + x^2 + 1 :: 285 :: 0x11D
#CODE_GEN_POLY = [1, 59, 13, 104, 189, 68, 209, 30, 8, 163, 65, 41, 229, 98, 50, 36, 59]
                                    # polynomial: x^16 + 59x^15 + 13x^14 + 104x^13 + 189x^12
                                    #             68x^11 + 209x^10 + 30x^9 + 8x^8 + 163x^7
                                    #             65x^6 + 41x^5 + 229x^4 + 98x^3 + 50x^2 + 36x + 59
#MIN_PRIM_ELEM = [1, 0]              # primitive element (alpha): x :: 1 :: '000000010'

GEN_POLY = [1,0,0,1,1]
CODE_GEN_POLY = [1,15,3,1,12]

# Perform addition in the Galois field through bitwise XOR
# num1, num2 must be in decimal form.
# returns decimal form sum.
# since subtraction is identical to addition in GF(256), we don't need a subtract function.
def add(num1, num2):
    try:
        num1, num2 = int(num1), int(num2)
    except:
        raise TypeError("Numbers must be integers:", num1, num2)
    return abs(num1) ^ abs(num2)

# Perform multiplication within the Galois field using log and anti-log tables mod 255.
# num1, num2 must be in decimal form.
# returns decimal form product.
def multiply(num1, num2):
    try:
        num1, num2 = int(num1), int(num2)
    except:
        raise TypeError("Numbers must be integers:", num1, num2)
    if num1 == 0 or num2 == 0:
        return 0
    j1, j2 = LOG_TABLE.get(abs(num1), 0), LOG_TABLE.get(abs(num2), 0)
    j = (j1 + j2) % N
    product = ANTILOG_TABLE.get(j, 0)
    return product

# Perform division in the Galois field through the log tables.
# num1, num2 must be in the correct order of divison desired: num1 / num2.
# takes two decimal integers num1, num2 and returns a decimal integer product
def divide(num1, num2):
    try:
        num1, num2 = int(num1), int(num2)
    except:
        raise TypeError("Numbers must be integers:", num1, num2)
    if num1 == 0:
        return 0
    elif num2 == 0:
        raise ZeroDivisionError('Cannot divide by zero in finite field')
    j2_inv = (-1*LOG_TABLE[abs(num2)])%N # division is the same as multiplying by the inverse
    num2_inv = ANTILOG_TABLE[j2_inv]
    return multiply(num1, num2_inv)

# Adds two polynomials of arbitrary lengths together under the GF
def polyAdd(poly1, poly2):
    len1, len2 = len(poly1), len(poly2)
    len_diff = len1 - len2
    if len_diff < 0:
        poly1 = np.pad(poly1, (abs(len_diff), 0), mode='constant')
    elif len_diff > 0:
        poly2 = np.pad(poly2, (len_diff, 0), mode='constant')
    final_len = len(poly1)
    poly_sum = np.zeros(final_len)
    for i in range(final_len):
        poly_sum[i] = add(int(poly1[i]), int(poly2[i]))
    return poly_sum

# Visciously stolen from numpy sourcecode and modified to work in GFs
def polyDiv(u, v):
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    q_scale = divide(1, int(v[0]))
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m-n+1):
        scale = divide(1, int(v[0]))
        d = multiply(scale, int(r[k]))
        d_q = multiply(q_scale, int(r[k]))
        q[k] = d_q
        for i, x in enumerate(v):
            v[i] = multiply(d, int(x))
        for j in range(k, k+n+1):
            r[j] = add(int(r[j]), int(v[j-k]))
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    return q, r

# Perform polynomial multiplication within the finite field.
# The length of two multiplied polynomial can be taken as the sum of the
# largest degrees of each. In list form, this is the length-1. We then add 1
# to include a constant at the end. so len-1 + len-1 + 1 = len + len -1
def polyMult(poly1, poly2):
    prod_len = len(poly1) + len(poly2) - 1
    prod = np.zeros(prod_len)
    for i, val1 in enumerate(poly1):
        for j, val2 in enumerate(poly2):
            prod[i+j] = add(int(prod[i+j]), multiply(int(val1), int(val2)))
    prod = np.trim_zeros(prod, 'f')
    return prod.tolist()

# Raises number 'val' to power 'exp' within the finite field.
# Receives two integers, returns one integer.
def exponent(val, exp):
    result = val
    for i in range(exp-1):
        result = multiply(result, val)
    return result

# Evaluates the polynomial at value 'val' within the finite field.
# Receives an array and an integer, returns an integer.
def polyVal(poly, val):
    result = 0
    deg = len(poly)-1
    for i in range(len(poly)-1):
        coef = poly[i]
        result = add(result, multiply(coef, exponent(val, deg-i)))
    result = add(result, poly[-1])
    return result

# Message will be a two-dimensional array containing k-1 decimal (from 8-bit) symbols.
# returns message*n^(N-K)+remainder=T(x)
def encode(message):
    shift = N-K
    shifted_poly = np.zeros(N+1)
    for index, symbol in enumerate(message): # mumtiply through by x^(n-k) == shift indexes by n-k
        new_index = (index+shift)%N
        shifted_poly[new_index] = symbol
    # now need to divide by code generator polynomial
    quotient, remainder = polyDiv(shifted_poly, CODE_GEN_POLY)
    return quotient, np.append(shifted_poly, remainder)

# Euclid's algorithm for finding the GCM of two polynomials
# Takes two arrays, returns one array (GCM).
def euclid(f, g):
    q_list, r_list = [], []
    # Stop if remainder is 0
    # while not np.array_equal(g, [0.]):
    while not np.array_equal(g, [0.]):
        q, r = polyDiv(f, g)
        q_list.append(q)
        r_list.append(r)
        f = g
        g = r
    # return the second last remainder = return the last non-zero remainder (gcm)
    return r_list[-2]

# Finds the magnitude polynomial from dividing x^(2t) by the Syndrome equation
# Takes two arrays as input and returns two arrays (polynomial, list of quotients)
def polyEuclid(f, g):
    # Euclid's algorithm for GCM of polynomials
    q_list, r_list = [], []
    # Stop if remainder is 0
    # while not np.array_equal(g, [0.]):
    while len(g)-1 >= T:
        q, r = polyDiv(f, g)
        q_list.append(q)
        r_list.append(r)
        f = g
        g = r
    # return the second last remainder = return the last non-zero remainder (gcm)
    return r_list[-1], q_list

# Finds the location and magnitude polynomials given the Syndrome equation.
# Takes one array as input and returns two arrays as output.
def solveSyndromes(S_x):
    # f = x^(2t)
    f = np.zeros((2*T)+1)
    f[0] = 1
    mag_poly, q_list = polyEuclid(f, S_x)
    # Initial sum value must be 0, initial value must be 1
    isv, iv = [0], [1]
    for poly in q_list:
        loc_poly = polyAdd(isv, polyMult(iv, poly))
        isv = iv
        iv = poly
    return loc_poly, mag_poly

# Finds the errors given the location and magnitude polynomials 
# derived from the Syndrome equation.
# Takes two arrays as input and returns an array.
def findErrors(loc_poly, mag_poly):
    errors = []
    for j in range(N):
        alpha_j = ANTILOG_TABLE[j]
        alpha_neg_j = ANTILOG_TABLE[-j%15]
        loc_val = polyVal(loc_poly, alpha_neg_j)
        if loc_val == 0:
            err_mag = multiply(alpha_j, divide(polyVal(mag_poly, alpha_neg_j), loc_poly[0]))
            errors.append([j,err_mag])
    return errors

# Finds the locations and magnitudes of errors in the received message R_x
# if they exist.
# Takes an array as input returns an array containing errors or 0 if none.
def detectErrors(R_x):
    syndromes = []
    for i in range(B, B+(2*T)):
        alpha_i = ANTILOG_TABLE[i]
        S_i = polyVal(R_x, alpha_i)
        syndromes.insert(0, S_i)
    # ensure syndrome equation is written in the correct direction
    # syndromes = np.flip(syndromes)
    if np.count_nonzero(syndromes) != 0:
        loc_poly, mag_poly = solveSyndromes(syndromes)
        errors = findErrors(loc_poly, mag_poly)
        return errors
    else:
        return 0

print(detectErrors([1,2,3,4,5,11,7,8,9,10,11,3,1,12,12]))

#print(polyDiv([3, 14], [9]))
#print(polyDiv([7,7,9], [9]))
#print(polyMult([3,14],[7]))
#print(polyAdd([7,7,8], [0,7,0]))
#print(euclid([7,7,9], [3,14]))
#print(solveSyndromes([11,11,5]))
#print(findErrors([6, 14],[10]))
#print(multiply(12, divide(10, 6)))