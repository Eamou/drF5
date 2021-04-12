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


# Set the Generator Polynomial and Minimum Primitive Element
# Changing these will mean the rest of the code will function incorrectly, if at all.
# They are the 'unique key' of this particular implementation of Reed-Solomon

# polynomial: x^16 + 59x^15 + 13x^14 + 104x^13 + 189x^12
#             68x^11 + 209x^10 + 30x^9 + 8x^8 + 163x^7
#             65x^6 + 41x^5 + 229x^4 + 98x^3 + 50x^2 + 36x + 59
#MIN_PRIM_ELEM = [1, 0]              # primitive element (alpha): x :: 1 :: '000000010'

class gf:
    def __init__(self, z):
        if z == 16:
            table_name = '.GF16_ANTILOG'

        elif z == 256:
            table_name = '.GF256_ANTILOG'
        
        else:
            raise ValueError("Sorry, RS does not support that finite field yet")
    
        if os.path.isfile(table_name):
            with open(table_name, 'rb') as fp:
                self.ANTILOG_TABLE = pickle.load(fp)
        else:
            raise FileNotFoundError(f'Could not load {table_name} table file')

        self.LOG_TABLE = {a_j: j for j, a_j in self.ANTILOG_TABLE.items()}
        self.N = z-1

    # Perform addition in the Galois field through bitwise XOR
    # num1, num2 must be in decimal form.
    # returns decimal form sum.
    # since subtraction is identical to addition in GF(256), we don't need a subtract function.
    def add(self, num1, num2):
        try:
            num1, num2 = int(num1), int(num2)
        except:
            raise TypeError("Numbers must be integers:", num1, num2)
        return abs(num1) ^ abs(num2)

    # Perform multiplication within the Galois field using log and anti-log tables mod 255.
    # num1, num2 must be in decimal form.
    # returns decimal form product.
    def multiply(self, num1, num2):
        try:
            num1, num2 = int(num1), int(num2)
        except:
            raise TypeError("Numbers must be integers:", num1, num2)
        if num1 == 0 or num2 == 0:
            return 0
        j1, j2 = self.LOG_TABLE.get(abs(num1), 0), self.LOG_TABLE.get(abs(num2), 0)
        j = (j1 + j2) % self.N
        product = self.ANTILOG_TABLE.get(j, 0)
        return product

    # Perform division in the Galois field through the log tables.
    # num1, num2 must be in the correct order of divison desired: num1 / num2.
    # takes two decimal integers num1, num2 and returns a decimal integer product
    def divide(self, num1, num2):
        try:
            num1, num2 = int(num1), int(num2)
        except:
            raise TypeError("Numbers must be integers:", num1, num2)
        if num1 == 0:
            return 0
        elif num2 == 0:
            raise ZeroDivisionError('Cannot divide by zero in finite field')
        j2_inv = (-1*self.LOG_TABLE[abs(num2)])%self.N # division is the same as multiplying by the inverse
        num2_inv = self.ANTILOG_TABLE[j2_inv]
        return self.multiply(num1, num2_inv)
    
    # Raises number 'val' to power 'exp' within the finite field.
    # Receives two integers, returns one integer.
    def exponent(self, val, exp):
        result = val
        if exp == 0:
            return 1
        elif exp == 1:
            return val
        else:
            for _ in range(exp-1):
                result = self.multiply(result, val)
            return result

class gf_poly(gf):
    def __init__(self, z):
        super().__init__(z)
    
    # Adds two polynomials of arbitrary lengths together under the GF
    # Takes two np arrays and returns one np array
    def polyAdd(self, poly1, poly2):
        len1, len2 = len(poly1), len(poly2)
        len_diff = len1 - len2
        if len_diff < 0:
            poly1 = np.pad(poly1, (abs(len_diff), 0), mode='constant')
        elif len_diff > 0:
            poly2 = np.pad(poly2, (len_diff, 0), mode='constant')
        final_len = len(poly1)
        poly_sum = np.zeros(final_len)
        for i in range(final_len):
            poly_sum[i] = self.add(poly1[i], poly2[i])
        return poly_sum

    # Visciously stolen from numpy sourcecode and modified to work in GFs
    # Takes two numpy arrays and returns two numpy arrays, quotient and remainder
    def polyDiv(self, u, v):
        u = atleast_1d(u) + 0.0
        v = atleast_1d(v) + 0.0
        v_original = v.copy()
        # w has the common type
        w = u[0] + v[0]
        m = len(u) - 1
        n = len(v) - 1
        q_scale = self.divide(1, v[0])
        q = NX.zeros((max(m - n + 1, 1),), w.dtype)
        r = u.astype(w.dtype)
        for k in range(0, m-n+1):
            if int(v[0]) == 0:
                scale = 1
            else:
                scale = self.divide(1, v[0])
            d = self.multiply(scale, r[k])
            d_q = self.multiply(q_scale, r[k])
            q[k] = d_q
            if not np.any(v):
                v = v_original.copy()
            for i, x in enumerate(v):
                v[i] = self.multiply(d, x)
            for j in range(k, k+n+1):
                r[j] = self.add(r[j], v[j-k])
        while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
            r = r[1:]
        return q, r

    # Perform polynomial multiplication within the finite field.
    # The length of two multiplied polynomial can be taken as the sum of the
    # largest degrees of each. In list form, this is the length-1. We then add 1
    # to include a constant at the end. so len-1 + len-1 + 1 = len + len -1
    def polyMult(self, poly1, poly2):
        prod_len = len(poly1) + len(poly2) - 1
        prod = np.zeros(prod_len)
        for i, val1 in enumerate(poly1):
            for j, val2 in enumerate(poly2):
                prod[i+j] = self.add(prod[i+j], self.multiply(val1, val2))
        prod = np.trim_zeros(prod, 'f')
        return prod.tolist()

    # Evaluates the polynomial at value 'val' within the finite field.
    # Receives an array and an integer, returns an integer.
    def polyVal(self, poly, val):
        # strip leading 0s
        poly = np.trim_zeros(poly, trim='f')
        result = 0
        deg = len(poly)-1
        for i, coef in enumerate(poly):
            result = self.add(result, self.multiply(coef, self.exponent(val, deg-i)))
        return result


    # Euclid's algorithm for finding the GCM of two polynomials
    # Takes two arrays, returns one array (GCM).
    def euclid(self, f, g):
        q_list, r_list = list(), list()
        # Stop if remainder is 0
        while not np.array_equal(g, [0.]):
            q, r = self.polyDiv(f, g)
            q_list.append(q)
            r_list.append(r)
            f = g
            g = r
        # return the second last remainder = return the last non-zero remainder (gcm)
        return r_list[-2]
    
    # Finds derivative of polynomial within finite field.
    # This is the same as setting even powers to 0.
    # Reveives one array, returns one array
    def derviative(self, poly):
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


class rs:
    def __init__(self, z):
        self.gf = gf(z)
        self.gf_poly = gf_poly(z)
        self.N, self.K, self.GEN_POLY, self.CODE_GEN_POLY = self.__def_params(z)
        self.T = (self.N - self.K) // 2
        self.B = 0
    
    def __gen_generator_poly(self, N, K):
        f = [1]
        for i in range(N-K):
            alpha_i = self.gf.ANTILOG_TABLE[i]
            f = self.gf_poly.polyMult(f, [alpha_i, 1])
        return f[::-1]

    def __def_params(self, GF_PARAM):
        if GF_PARAM == 16:
            GEN_POLY = [1,0,0,1,1]             # x^4 + x + 1 :: [1,0,0,1,1] :: field gen poly
            CODE_GEN_POLY = [1,15,3,1,12]
            N = 15
            K = 11

        elif GF_PARAM == 256:
            GEN_POLY = [1,0,0,0,1,1,1,0,1]      # polynomial: x^8 + x^4 + x^3 + x^2 + 1 :: 285 :: 0x11D
            # need correct code gen poly
            N = 255
            K = 223
            CODE_GEN_POLY = self.__gen_generator_poly(N, K) #[1, 59, 13, 104, 189, 68, 209, 30, 8, 163, 65, 41, 229, 98, 50, 36, 59]
        return N, K, GEN_POLY, CODE_GEN_POLY

    # Message will be a two-dimensional array containing k-1 decimal (from 8-bit) symbols.
    # returns message*n^(N-K)+remainder=T(x)
    def encodeMsg(self, message):
        if len(message) == 0:
            raise Exception('Message length zero')
        if len(message) > self.K:
            raise ValueError('Message too long, length:', len(message))
        # multiply by x^(2t) same as appending 2t zeros
        for _ in range(self.N-self.K):
            message.append(0)
        # now need to divide by code generator polynomial
        _, remainder = self.gf_poly.polyDiv(message, self.CODE_GEN_POLY)
        message = np.trim_zeros(message, 'b')
        message = np.append(message, remainder)
        return message
    
    # Finds the magnitude polynomial from dividing x^(2t) by the Syndrome equation
    # Takes two arrays as input and returns two arrays (polynomial, list of quotients)
    def polyEuclid(self, f, g):
        g = np.trim_zeros(g, 'f')
        # Euclid's algorithm for GCM of polynomials
        q_list, r_list = list(), list()
        # Stop if remainder is 0
        while len(g)-1 >= self.T:
            q, r = self.gf_poly.polyDiv(f, g)
            q_list.append(q)
            r_list.append(r)
            f = g
            g = r
        # return the second last remainder = return the last non-zero remainder (gcm)
        return r_list[-1], q_list

    # Finds the location and magnitude polynomials given the Syndrome equation.
    # Takes one array as input and returns two arrays as output.
    def solveSyndromes(self, S_x):
        # f = x^(2t)
        f = np.zeros((2*self.T)+1)
        f[0] = 1
        mag_poly, q_list = self.polyEuclid(f, S_x)
        # Initial sum value must be 0, initial value must be 1
        isv, iv = [0], [1]
        for poly in q_list:
            loc_poly = self.gf_poly.polyAdd(isv, self.gf_poly.polyMult(iv, poly))
            isv = iv
            iv = poly
        return loc_poly, mag_poly

    # Finds the errors given the location and magnitude polynomials 
    # derived from the Syndrome equation.
    # Takes two arrays as input and returns an array.
    def findErrors(self, loc_poly, mag_poly):
        errors = list()
        for j in range(self.N):
            alpha_j = self.gf.ANTILOG_TABLE[j]
            alpha_neg_j = self.gf.ANTILOG_TABLE[-j%self.N]
            loc_val = self.gf_poly.polyVal(loc_poly, alpha_neg_j)
            if loc_val == 0:
                loc_poly_prime = self.gf_poly.derviative(loc_poly.copy())
                err_mag = self.gf.multiply(alpha_j, self.gf.divide(self.gf_poly.polyVal(mag_poly, alpha_neg_j), self.gf_poly.polyVal(loc_poly_prime, alpha_neg_j)))
                errors.append([j,err_mag])
        return errors

    # Amends errors in received message
    # Takes two arrays as input, returns one array
    def fixErrors(self, R_x, errors):
        l = len(R_x)-1
        for i, mag in errors:
            R_x[l-i] = self.gf.add(R_x[l-i], mag)
        return R_x

    # Finds the locations and magnitudes of errors in the received message R_x
    # if they exist.
    # Takes an array as input returns an array containing errors or 0 if none.
    def detectErrors(self, R_x):
        syndromes = list()
        for i in range(self.B, self.B+(2*self.T)):
            alpha_i = self.gf.ANTILOG_TABLE[i]
            S_i = self.gf_poly.polyVal(R_x, alpha_i)
            syndromes.insert(0, S_i)
        syndromes = np.trim_zeros(syndromes, 'f')
        if np.count_nonzero(syndromes) != 0:
            loc_poly, mag_poly = self.solveSyndromes(syndromes)
            loc_roots = list()
            for j in range(self.N):
                alpha_j = self.gf.ANTILOG_TABLE[j]
                val = self.gf_poly.polyVal(loc_poly, alpha_j)
                if val == 0:
                    loc_roots.append(alpha_j)
            if len(loc_roots) != len(loc_poly)-1:
                # if the locaction polynomial has num of roots unequal to its degree, too many errors.
                raise Exception('Codeword contains too many errors')
                #cont = str(input('ERROR: Codeword contains too many errors to correct. Continue anyway? y/n: '))
                #if cont != 'y':
                #    exit(0)
            errors = self.findErrors(loc_poly, mag_poly)
            try:
                R_x = self.fixErrors(R_x, errors)
            except:
                return R_x
        return R_x

    # Converts message bitstring into polynomial with decimal coefficients
    # Takes string as input, returns array
    def prepareMessage(self, bitstring):
        blocks, block_size = len(bitstring), 8
        message = [bitstring[i:i+block_size] for i in range(0, blocks, block_size)]
        message = [int(i, 2) for i in message]
        messages = [message[i:i+self.K] for i in range(0, len(message), self.K)]
        return np.array([self.encodeMsg(message) for message in messages])

    def getLocPoly(self, R_x, err_locs):
        #convert error locations to right-to-left indices
        err_locs = [len(R_x)-1-x for x in err_locs]
        loc_poly = [1]
        for i in err_locs:
            alpha_i = self.gf.ANTILOG_TABLE[i]
            loc_poly = self.gf_poly.polyMult(loc_poly, [alpha_i, 1])
        return loc_poly

    def getErrPoly(self, S_x, L_x):
        f = np.zeros((2*self.T)+1)
        f[0] = 1
        _, r = self.gf_poly.polyDiv(self.gf_poly.polyMult(S_x, L_x), f)
        return r

    def detectErasures(self, R_x, err_locs):
        syndromes = list()
        for i in range(self.B, self.B+(2*self.T)):
            alpha_i = self.gf.ANTILOG_TABLE[i]
            S_i = self.gf_poly.polyVal(R_x, alpha_i)
            syndromes.insert(0, S_i)
        syndromes = np.trim_zeros(syndromes, 'f')
        if np.count_nonzero(syndromes) != 0:
            loc_poly = self.getLocPoly(R_x, err_locs) #capital Gamma
            err_poly = self.getErrPoly(syndromes, loc_poly)
            errors = self.findErrors(loc_poly, err_poly)
            return self.fixErrors(R_x, errors)

#rs_obj = rs(256)
#m = [1,2,3,4,5,6,7,8,9,10,11]
#e = rs_obj.encodeMsg(m)
#print(e)
#e[1] = 0
#e[2] = 0
#print(e)
#print(rs_obj.detectErasures(e, [1, 2]))


#rs_obj = rs(256)
#mp = [1,2,3,4,5,6,7,8,9,10,12]
#ep = rs_obj.encodeMsg(mp)
#print(ep)
#ep[2] = 0
#ep[4] = 5
#ep[6] = 7
#print(ep)
#print(rs_obj.detectErrors(ep))

# encode the coefficients themselves.
# reed solomon will restore coefficients that went to 0 but loses the sign.
# then dmcss restores the sign and the message can be extracted.



#print(rs_obj.detectErrors(np.array([114., 101., 101., 100.,  32., 115., 111., 108., 111., 109., 111.,
#       110.,  32., 116., 101., 115., 116.,  32.,  52., 171., 238., 229.,
#       206., 113.,  68., 207.,  42.,  73., 122., 207.,  87., 248.,  54.,
#        14., 178.])))

"""
to do:
when we know the locations of the errors, the error correction capacity is doubled.
with jpeg cropping we will always know where the errors are as they will be unable to be read
when attempting to read the message embedding path so add the functionality to action on this!
"""