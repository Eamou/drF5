from re import S
import numpy as np
from numpy.core import atleast_1d
import numpy.core.numeric as NX
import pickle
import os.path
from numpy.lib.polynomial import poly
from numpy.testing._private.utils import KnownFailureException

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

# Code-defining constants
# As they are now, 4 errors can be corrected. This can be adjusted by changing these values.
# M = 8
# ALPHA = 2

# Set the Generator Polynomial and Minimum Primitive Element
# Changing these will mean the rest of the code will function incorrectly, if at all.
# They are the 'unique key' of this particular implementation of Reed-Solomon

                                    # polynomial: x^16 + 59x^15 + 13x^14 + 104x^13 + 189x^12
                                    #             68x^11 + 209x^10 + 30x^9 + 8x^8 + 163x^7
                                    #             65x^6 + 41x^5 + 229x^4 + 98x^3 + 50x^2 + 36x + 59
#MIN_PRIM_ELEM = [1, 0]              # primitive element (alpha): x :: 1 :: '000000010'


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
        poly_sum[i] = add(poly1[i], poly2[i])
    return poly_sum

# Visciously stolen from numpy sourcecode and modified to work in GFs
def polyDiv(u, v):
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    v_original = v.copy()
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    q_scale = divide(1, v[0])
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m-n+1):
        #print(f"line {k}:", v, r, v[0], r[k])
        if int(v[0]) == 0:
            scale = 1
        else:
            scale = divide(1, v[0])
        d = multiply(scale, r[k])
        d_q = multiply(q_scale, r[k])
        q[k] = d_q
        if not np.any(v):
            v = v_original.copy()
        for i, x in enumerate(v):
            v[i] = multiply(d, x)
        for j in range(k, k+n+1):
            r[j] = add(r[j], v[j-k])
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
            prod[i+j] = add(prod[i+j], multiply(val1, val2))
    prod = np.trim_zeros(prod, 'f')
    return prod.tolist()

# Raises number 'val' to power 'exp' within the finite field.
# Receives two integers, returns one integer.
def exponent(val, exp):
    result = val
    if exp == 0:
        return 1
    elif exp == 1:
        return val
    else:
        for _ in range(exp-1):
            result = multiply(result, val)
        return result

# Evaluates the polynomial at value 'val' within the finite field.
# Receives an array and an integer, returns an integer.
def polyVal(poly, val):
    # strip leading 0s
    poly = np.trim_zeros(poly, trim='f')
    result = 0
    deg = len(poly)-1
    for i, coef in enumerate(poly):
        result = add(result, multiply(coef, exponent(val, deg-i)))
    return result


#print("TEST:", polyVal([14,14,1],ANTILOG_TABLE[(-j)%N]))

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
    g = np.trim_zeros(g, 'f')
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

# Finds derivative of polynomial within finite field.
# This is the same as setting even powers to 0.
# Reveives one array, returns one array
def derviative(poly):
    poly_len = len(poly)
    if poly_len == 0:
        return []
    deg = poly_len-1
    for i in range(len(poly)):
        power = deg - i
        if (power % 2) == 0:
            poly[i] = 0
    poly = poly[:-1] # remove constant term
    return np.trim_zeros(poly, 'f')

# Finds the errors given the location and magnitude polynomials 
# derived from the Syndrome equation.
# Takes two arrays as input and returns an array.
def findErrors(loc_poly, mag_poly):
    errors = []
    for j in range(N):
        alpha_j = ANTILOG_TABLE[j]
        alpha_neg_j = ANTILOG_TABLE[-j%N]
        loc_val = polyVal(loc_poly, alpha_neg_j)
        if loc_val == 0:
            loc_poly_prime = derviative(loc_poly.copy())
            err_mag = multiply(alpha_j, divide(polyVal(mag_poly, alpha_neg_j), polyVal(loc_poly_prime, alpha_neg_j)))
            errors.append([j,err_mag])
    return errors

# Amends errors in received message
# Takes two arrays as input, returns one array
def fixErrors(R_x, errors):
    l = len(R_x)-1
    for i, mag in errors:
        R_x[l-i] = add(R_x[l-i], mag)
    return R_x

# Finds the locations and magnitudes of errors in the received message R_x
# if they exist.
# Takes an array as input returns an array containing errors or 0 if none.
def detectErrors(R_x):
    syndromes = []
    for i in range(B, B+(2*T)):
        alpha_i = ANTILOG_TABLE[i]
        S_i = polyVal(R_x, alpha_i)
        syndromes.insert(0, S_i)
    syndromes = np.trim_zeros(syndromes, 'f')
    # ensure syndrome equation is written in the correct direction
    # syndromes = np.flip(syndromes)
    if np.count_nonzero(syndromes) != 0:
        loc_poly, mag_poly = solveSyndromes(syndromes)
        loc_roots = []
        for j in range(N):
            alpha_j = ANTILOG_TABLE[j]
            val = polyVal(loc_poly, alpha_j)
            if val == 0:
                loc_roots.append(alpha_j)
        if len(loc_roots) != len(loc_poly)-1:
            # if the locaction polynomial has num of roots unequal to its degree, too many errors.
            raise Exception('Codeword contains too many errors')
        errors = findErrors(loc_poly, mag_poly)
        R_x = fixErrors(R_x, errors)
    return R_x

# Message will be a two-dimensional array containing k-1 decimal (from 8-bit) symbols.
# returns message*n^(N-K)+remainder=T(x)
def encode(message):
    if len(message) == 0:
        raise Exception('Message length zero')
    if len(message) > K:
        raise ValueError('Message too long, length:', len(message))
    # multiply by x^(2t) same as appending 2t 0s
    for _ in range(N-K):
        message.append(0)
    # now need to divide by code generator polynomial
    _, remainder = polyDiv(message, CODE_GEN_POLY)
    message = np.trim_zeros(message, 'b')
    message = np.append(message, remainder)
    return message

# Converts message bitstring into polynomial with decimal coefficients
# Takes string as input, returns array
def prepareBitString(bitstring):
    blocks, block_size = len(bitstring), 8
    message = [bitstring[i:i+block_size] for i in range(0, blocks, block_size)]
    message = [int(i, 2) for i in message]
    return message

###############################################################################################################

GF_PARAM = 256

# Table 2: 0x11D

if GF_PARAM == 16:
    table_name = '.GF16_ANTILOG'
    GEN_POLY = [1,0,0,1,1]             # x^4 + x + 1 :: [1,0,0,1,1] :: field gen poly
    CODE_GEN_POLY = [1,15,3,1,12]
    N = 15
    K = 11

elif GF_PARAM == 256:
    table_name = '.GF256_ANTILOG'
    GEN_POLY = [1,0,0,0,1,1,1,0,1]      # polynomial: x^8 + x^4 + x^3 + x^2 + 1 :: 285 :: 0x11D
    CODE_GEN_POLY = [1, 59, 13, 104, 189, 68, 209, 30, 8, 163, 65, 41, 229, 98, 50, 36, 59]
    N = 255
    K = 239

T = (N - K) // 2
B = 0

if os.path.isfile(table_name):
    with open(table_name, 'rb') as fp:
        ANTILOG_TABLE = pickle.load(fp)
else:
    raise FileNotFoundError(f'Could not load {table_name} table file')

LOG_TABLE = {a_j: j for j, a_j in ANTILOG_TABLE.items()}


#bitstring = '011100100110010101100101011001000010000001110011011011110110110001101111011011010110111101101110'
#message_poly = prepareBitString(bitstring)
encoded_poly = encode([1,2,3,4,5,6,7,8,9,10,11])
print("encoded:", encoded_poly)
error_poly = encoded_poly.copy()
# two consecutive same numbers [115, 115, 110, 120...] breaks it for some reason?
# doesnt break in gf(16)...
error_poly[0] = 2.
error_poly[2] = 2.
#error_poly[3] = 2.
#error_poly[1] = error_poly[1]+1.
print("encoded+error:", error_poly)
corrected_poly = detectErrors(error_poly)
print("corrected:",corrected_poly)
assert np.array_equal(encoded_poly, corrected_poly)
print("***pass***")

"""
to do:
when we know the locations of the errors, the error correction capacity is doubled.
with jpeg cropping we will always know where the errors are as they will be unable to be read
when attempting to read the message embedding path so add the functionality to action on this!

bugfix 256 //done???
"""