import numpy as np
import cv2
import math
import pickle
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import simplejpeg
from random import randrange, choice
from timeit import default_timer as timer

from sdcs import sdcs
from stc import stc
from rs import rs

# to-do:
# 1. enable program to work with any image dimension //done?
# 2. enable chroma subsampling - not required but might be nice
# 3. jpeg quality options?
# 4. command line options for the 2 and 3

# use a unique hamming/reed-solomon code to encode each letter, ensuring each letter
# uses any particular block no more than once. this way, if there are multiple bits in a block
# that is corrupted by cropping, the error correcting codes still need only correct one error.
# hence, hamming codes can be used. reed-solomon will allow for higher payloads
# as more letters will be included in an error-correcting chunk, reducing redundancy
# overheads and increasing overall payload size.

# Huffman tables for DC and AC values
# yes they are massive, I think this is probably
# the fastest and most sensible way to implement them
# considering I have to pull the inverse too.

class encoder:
    def __init__(self, block_size, rs_param):
        self.TIME_LIMIT = 10
        self.BLOCK_SIZE = block_size
        self.RS_PARAM = rs_param
        self.img_width, self.img_height = None, None
        self.hor_block_count, self.ver_block_count = None, None
        self.Y_quant_table, self.C_quant_table = self.__getQuantTables()
        self.dc_codeword_dict, self.dc_codeword_dict_inv = self.__getDCCodewordDicts()
        self.ac_codeword_dict, self.ac_codeword_dict_inv = self.__getACCodewordDicts()

    def defineBlockCount(self, v, h):
        # helper function for when calling from main.py
        self.hor_block_count = h
        self.ver_block_count = v

    def defineImgDim(self, h, w):
        self.img_height = h
        self.img_width = w

    def __readImage(self, image_name):
        # return image object img
        if image_name[-3:] == "pgm":
            img, greyscale = cv2.imread(image_name, -1), True
        else:
            img, greyscale = cv2.imread(image_name,cv2.IMREAD_COLOR), False
        self.img_height, self.img_width = self.getImageDimensions(img)
        return img, greyscale

    def __getDCCodewordDicts(self):
        dc_codeword_dict = {
            0: '00',
            1: '010',
            2: '011',
            3: '100',
            4: '101',
            5: '110',
            6: '1110',
            7: '11110',
            8: '111110',
            9: '1111110',
            10: '11111110',
            11: '111111110'
        }

        dc_codeword_dict_inv = {codeword: cat for cat, codeword in dc_codeword_dict.items()}
        return dc_codeword_dict, dc_codeword_dict_inv

    def __getACCodewordDicts(self):
        ac_codeword_dict = {
            (0,0): '1010',
            (0,1): '00',
            (0,2): '01',
            (0,3): '100',
            (0,4): '1011',
            (0,5): '11010',
            (0,6): '1111000',
            (0,7): '11111000',
            (0,8): '1111110110',
            (0,9): '1111111110000010',
            (0,10): '1111111110000011',
            (1,1): '1100',
            (1,2): '11011',
            (1,3): '1111001',
            (1,4): '111110110',
            (1,5): '11111110110',
            (1,6): '1111111110000100',
            (1,7): '1111111110000101',
            (1,8): '1111111110000110',
            (1,9): '1111111110000111',
            (1,10): '1111111110001000',
            (2,1): '11100',
            (2,2): '11111001',
            (2,3): '1111110111',
            (2,4): '111111110100',
            (2,5): '1111111110001001',
            (2,6): '1111111110001010',
            (2,7): '1111111110001011',
            (2,8): '1111111110001100',
            (2,9): '1111111110001101',
            (2,10): '1111111110001110',
            (3,1): '111010',
            (3,2): '111110111',
            (3,3): '111111110101',
            (3,4): '1111111110001111',
            (3,5): '1111111110010000',
            (3,6): '1111111110010001',
            (3,7): '1111111110010010',
            (3,8): '1111111110010011',
            (3,9): '1111111110010100',
            (3,10): '1111111110010101',
            (4,1): '111011',
            (4,2): '1111111000',
            (4,3): '1111111110010110',
            (4,4): '1111111110010111',
            (4,5): '1111111110011000',
            (4,6): '1111111110011001',
            (4,7): '1111111110011010',
            (4,8): '1111111110011011',
            (4,9): '1111111110011100',
            (4,10): '1111111110011101',
            (5,1): '1111010',
            (5,2): '11111110111',
            (5,3): '1111111110011110',
            (5,4): '1111111110011111',
            (5,5): '1111111110100000',
            (5,6): '1111111110100001',
            (5,7): '1111111110100010',
            (5,8): '1111111110100011',
            (5,9): '1111111110100100',
            (5,10): '1111111110100101',
            (6,1): '1111011',
            (6,2): '111111110110',
            (6,3): '1111111110100110',
            (6,4): '1111111110100111',
            (6,5): '1111111110101000',
            (6,6): '1111111110101001',
            (6,7): '1111111110101010',
            (6,8): '1111111110101011',
            (6,9): '1111111110101100',
            (6,10): '1111111110101101',
            (7,1): '11111010',
            (7,2): '111111110111',
            (7,3): '1111111110101110',
            (7,4): '1111111110101111',
            (7,5): '1111111110110000',
            (7,6): '1111111110110001',
            (7,7): '1111111110110010',
            (7,8): '1111111110110011',
            (7,9): '1111111110110100',
            (7,10): '1111111110110101',
            (8,1): '111111000',
            (8,2): '111111111000000',
            (8,3): '1111111110110110',
            (8,4): '1111111110110111',
            (8,5): '1111111110111000',
            (8,6): '1111111110111001',
            (8,7): '1111111110111010',
            (8,8): '1111111110111011',
            (8,9): '1111111110111100',
            (8,10): '1111111110111101',
            (9,1): '111111001',
            (9,2): '1111111110111110',
            (9,3): '1111111110111111',
            (9,4): '1111111111000000',
            (9,5): '1111111111000001',
            (9,6): '1111111111000010',
            (9,7): '1111111111000011',
            (9,8): '1111111111000100',
            (9,9): '1111111111000101',
            (9,10): '1111111111000110',
            (10,1): '111111010',
            (10,2): '1111111111000111',
            (10,3): '1111111111001000',
            (10,4): '1111111111001001',
            (10,5): '1111111111001010',
            (10,6): '1111111111001011',
            (10,7): '1111111111001100',
            (10,8): '1111111111001101',
            (10,9): '1111111111001110',
            (10,10): '1111111111001111',
            (11,1): '1111111001',
            (11,2): '1111111111010000',
            (11,3): '1111111111010001',
            (11,4): '1111111111010010',
            (11,5): '1111111111010011',
            (11,6): '1111111111010100',
            (11,7): '1111111111010101',
            (11,8): '1111111111010110',
            (11,9): '1111111111010111',
            (11,10): '1111111111011000',
            (12,1): '1111111010',
            (12,2): '1111111111011001',
            (12,3): '1111111111011010',
            (12,4): '1111111111011011',
            (12,5): '1111111111011100',
            (12,6): '1111111111011101',
            (12,7): '1111111111011110',
            (12,8): '1111111111011111',
            (12,9): '1111111111100000',
            (12,10): '1111111111100001',
            (13,1): '11111111000',
            (13,2): '1111111111100010',
            (13,3): '1111111111100011',
            (13,4): '1111111111100100',
            (13,5): '1111111111100101',
            (13,6): '1111111111100110',
            (13,7): '1111111111100111',
            (13,8): '1111111111101000',
            (13,9): '1111111111101001',
            (13,10): '1111111111101010',
            (14,1): '1111111111101011',
            (14,2): '1111111111101100',
            (14,3): '1111111111101101',
            (14,4): '1111111111101110',
            (14,5): '1111111111101111',
            (14,6): '1111111111110000',
            (14,7): '1111111111110001',
            (14,8): '1111111111110010',
            (14,9): '1111111111110011',
            (14,10): '1111111111110100',
            (15,1): '1111111111110101',
            (15,2): '1111111111110110',
            (15,3): '1111111111110111',
            (15,4): '1111111111111000',
            (15,5): '1111111111111001',
            (15,6): '1111111111111010',
            (15,7): '1111111111111011',
            (15,8): '1111111111111100',
            (15,9): '1111111111111101',
            (15,10): '1111111111111110',
            (15,0): '11111111001'
        }

        ac_codeword_dict_inv = {codeword: cat for cat, codeword in ac_codeword_dict.items()}

        return ac_codeword_dict, ac_codeword_dict_inv

    def __getQuantTables(self):
        Y_quant_table = np.array([[
            16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99
        ]])

        C_quant_table = np.array([[
            17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99
        ]])

        return Y_quant_table, C_quant_table

    def getImageDimensions(self, img):
        # return image height, width as integers
        return img.shape[0], img.shape[1]

    def padImageHeight(self, img):
        # repeat last row of pixels until dimension is multiple of 8
        while len(img) % self.BLOCK_SIZE != 0:
            img = np.append(img, [img[len(img)-1]], axis=0)
        return img

    def padImageWidth(self, img):
        # repeat last column of pixels until dimension is multiple of 8
        img_list = list(img)
        width = self.getImageDimensions(img)[1]
        while width % self.BLOCK_SIZE != 0:
            for row_index in range(len(img_list)):
                row_list = list(img_list[row_index])
                try:
                    pixel_list = list(row_list[-1])
                    row_list.append(pixel_list)
                except:
                    pixel_list = [row_list[-1]]
                    row_list += pixel_list
                img_list[row_index] = row_list
            width += 1
        return np.array(img_list)

    def displayImage(self, img):
        # show image in new window until key pressed
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def blockify(self, img):
        final_img = list()
        for channel in img:
            img_tiles = list() # transform image into series of 8x8 blocks
            for row in range(0, self.img_height, self.BLOCK_SIZE):
                img_tiles_row = list() # fill in row by row
                for column in range(0, self.img_width, self.BLOCK_SIZE):
                    column_end = column + self.BLOCK_SIZE
                    row_end = row + self.BLOCK_SIZE
                    # select the 8x8 tile
                    tile = channel[row:row_end, column:column_end]
                    # add it to the row array
                    img_tiles_row.append(tile)
                # append rows individually to the image matrix
                # this ensure the dimensions are consistent
                img_tiles.append(img_tiles_row)
            final_img.append(np.array(img_tiles, dtype=np.float32))
        return np.array(final_img)

    def w(self, k_num):
        # for use in DCT transformation
        if k_num == 0:
            return 1/math.sqrt(2)
        else:
            return 1

    def DCT_2(self, img):
        # transform img into DCT coefficients
        return np.array([np.array([np.array([cv2.dct(block) for block in row]) for row in channel]) for channel in img])

    def quantizeAndRound(self, img):
        # quantizes DCT coefs in-place using quant_table_2 atm (add quality options later)
        # then rounds to nearest integer
        final_img = list()
        for i, channel in enumerate(img):
            table = self.Y_quant_table if i == 0 else self.C_quant_table
            final_img.append(np.array([np.rint(np.divide(block, table)) for block in np.array([row for row in channel])]))
        return np.array(final_img)

    def zigZagEncode(self, img):
        # https://stackoverflow.com/questions/39440633/matrix-to-vector-with-python-numpy
        # convert 8x8 block of dct coef's into a 64-len array via zig zag arrangement
        return np.array([np.array([np.array([np.hstack([np.diagonal(block[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-block.shape[0], block.shape[0])]) for block in row]) for row in channel]) for channel in img])

    def RLEandDPCM(self, zz_img):
        # create array of all DC values, encoded using DPCM - each value is the difference
        # from the previous value rather than the actual value
        # create array of RLE-encoded AC values - [skip, value]
        # where skip is the number of zeroes preceeding value.
        # [0,0] indicates the end of the block and is appended to the end
        final_img = list()
        for channel in zz_img:
            dc_array, ac_arrays = list(), list()
            zz_img_len = len(channel)
            for row_block_i in range(zz_img_len):
                row_len = len(channel[row_block_i])
                for block_i in range(row_len):
                    ac_rle = list()
                    if block_i == 0 and row_block_i == 0:
                        # encode the first DC value as-is
                        dc_array.append(channel[row_block_i][block_i][0])
                    else:
                        # for the rest, encode the difference
                        if block_i != 0:
                            dc_array.append(channel[row_block_i][block_i][0] - channel[row_block_i][block_i-1][0])
                        else:
                            dc_array.append(channel[row_block_i][block_i][0] - channel[row_block_i-1][block_i-1][0])
                    # start at 1 to skip the DC value
                    ac_i = 1
                    zero_count = 0
                    # max zero count is to keep track of how many 16-0's there are
                    # these should only be added to the ac_rle if there is a non-zero
                    # coefficient after them. otherwise, [0,0] should follow the final
                    # non-zero coefficient.
                    max_zero_count = 0
                    while ac_i < len(channel[row_block_i][block_i]):
                        cur_num = channel[row_block_i][block_i][ac_i]
                        if cur_num == 0:
                            if zero_count == 15:
                                max_zero_count += 1
                                zero_count = 0
                                ac_i += 1
                                continue
                            zero_count += 1
                            ac_i += 1
                            continue
                        else:
                            if max_zero_count > 0:
                                for _ in range(max_zero_count):
                                    ac_rle.append([15,0])
                            ac_rle.append([zero_count, cur_num])
                            zero_count = 0
                            ac_i += 1
                            continue
                    # append end of block marker
                    ac_rle.append([0,0])
                    ac_arrays.append(ac_rle)
            final_img += [np.array(dc_array), ac_arrays]
        return np.array(final_img)

    def categorize(self, coef):
        # return category of coefficient (DC or AC) based on the table
        # specified in the JPEG standard
        if coef == 0:
            return 0
        elif coef == -1 or coef == 1:
            return 1
        elif coef in range(-3,-1) or coef in range(2,4):
            return 2
        elif coef in range(-7,-3) or coef in range(4,8):
            return 3
        elif coef in range(-15,-7) or coef in range(8,16):
            return 4
        elif coef in range(-31,-15) or coef in range(16,32):
            return 5
        elif coef in range(-63,-31) or coef in range(32,64):
            return 6
        elif coef in range(-127,-63) or coef in range(64,128):
            return 7
        elif coef in range(-255,-127) or coef in range(128,256):
            return 8
        elif coef in range(-511,-255) or coef in range(256,512):
            return 9
        elif coef in range(-1023,-511) or coef in range(512,1024):
            return 10
        elif coef in range(-2047,-1023) or coef in range(1024,2048):
            return 11
        elif coef in range(-4095,-2047) or coef in range(2048, 4096):
            return 12
        elif coef in range(-8191,-4095) or coef in range(4096, 8192):
            return 13
        elif coef in range(-16383,-8191) or coef in range(8192,16384):
            return 14
        elif coef in range(-32767,-16383) or coef in range(16384,32768):
            return 15
        elif coef == 32768:
            return 16
        else:
            return -1

    def onesComp(self, bitstring):
        oc_bitstring = ''
        for char in bitstring:
            if char == '1':
                oc_bitstring += '0'
            elif char == '0':
                oc_bitstring += '1'
        return oc_bitstring

    def huffman(self, img):
        # compute and create final bitstring of data
        # dc and ac arrays should have same length, so can just use one
        Y_dc_arr, Y_ac_arr, Cb_dc_arr, Cb_ac_arr, Cr_dc_arr, Cr_ac_arr = img
        bitstring = ''
        length = len(Y_dc_arr)
        dc_arr, ac_arr = Y_dc_arr, Y_ac_arr # set default value to remove warnings
        for index in range(length):
            YCbCr_num = 0
            while YCbCr_num < 3:
                if YCbCr_num == 0:
                    dc_arr, ac_arr = Y_dc_arr, Y_ac_arr
                elif YCbCr_num == 1:
                    dc_arr, ac_arr = Cb_dc_arr, Cb_ac_arr
                elif YCbCr_num == 2:
                    dc_arr, ac_arr = Cr_dc_arr, Cr_ac_arr
                # find dc bit representation first
                #print(dc_arr[index], YCbCr_num)
                dc_category = self.categorize(dc_arr[index])
                dc_codeword = self.dc_codeword_dict[dc_category]
                # the [2:] is to remove the '0b' from the front of the binary string
                dc_magnitude = bin(int(dc_arr[index]))[2:]
                if int(dc_arr[index]) < 0:
                    dc_magnitude = self.onesComp(dc_magnitude)
                if dc_codeword != '00':
                    dc_bitstring = dc_codeword + dc_magnitude
                else:
                    dc_bitstring = dc_codeword
                bitstring += dc_bitstring
                # now find all the ac bit reps
                for ac_coef in ac_arr[index]:
                    # ac values are stored in pairs (skip, value)
                    # with skip being the number of zeroes, value being the
                    # value of the next non-zero coefficient
                    ac_skip = int(ac_coef[0])
                    ac_value = int(ac_coef[1])
                    ac_category = self.categorize(ac_value)
                    ac_magnitude = bin(ac_value)[2:]
                    if ac_value < 0:
                        ac_magnitude = self.onesComp(ac_magnitude)
                    ac_codeword = self.ac_codeword_dict[(ac_skip, ac_category)]
                    if ac_codeword != '1010' and ac_codeword != '11111111001':
                        ac_bitstring = ac_codeword + ac_magnitude
                    else:
                        ac_bitstring = ac_codeword
                    bitstring += ac_bitstring
                YCbCr_num += 1
        return bitstring

    def messageConv(self, message):
        return ''.join([format(ord(x), '08b') for x in message])

    def findMaxPayload(self, img_height, img_width):
        return (img_height // self.BLOCK_SIZE * img_width // self.BLOCK_SIZE)

    def lsbF5(self, x):
        if x < 0:
            return int((1 - x) % 2)
        else:
            return int(x % 2)

    def sdcsF5(self, msg, img):
        hash_path = ''
        num_channels = img.shape[0]
        # set up sdcs
        n, k, m, a = 3, 2, 17, [1,2,6]
        f5_sdcs = sdcs((n,k,m), a)
        # convert message to correct format for sdcs - blocks of n z_m integers
        num_bits_per_int = math.floor(math.log(m, 2))
        b_arr = list()
        for i in range(0, len(msg), num_bits_per_int):
            bits = msg[i:i+num_bits_per_int]
            b_arr.append(int(bits,2))
        # what to do with left-over vals if it doesnt divide equally into n coefs?
        # begin embedding
        path = list()
        b_i = 0
        block_perms = np.arange(num_channels * self.ver_block_count * self.hor_block_count)
        for block_num in block_perms:
            channel_i = block_num // (self.ver_block_count * self.hor_block_count)
            row_i = (block_num % (self.ver_block_count * self.hor_block_count)) // self.hor_block_count
            block_i = (block_num % (self.ver_block_count * self.hor_block_count)) % self.hor_block_count
            channel = img[channel_i]
            suitable_coefs_boolmask = np.array([0<coef<(m-1) for coef in channel[row_i][block_i]]) # true or false based on value
            suitable_coefs_boolmask[0] = False # avoid DC values
            suitable_coefs = np.extract(suitable_coefs_boolmask, channel[row_i][block_i]) # filter array by value
            suitable_coefs_index = np.where(suitable_coefs_boolmask==True)[0] # get indexes of those filtered
            if len(suitable_coefs) < n: # if there arent enough suitable coefficients
                continue
            for j in range(0, len(suitable_coefs), n):
                coefs_i = suitable_coefs_index[j:j+n]
                coefs = suitable_coefs[j:j+n] # what we have,
                if len(coefs) < n:
                    continue
                b = b_arr[b_i] # what we want,
                delta = f5_sdcs.embed(coefs, b) # how we change what we have to get what we want
                block_path = list()
                for i, coef_index in enumerate(coefs_i): # make the changes:
                    channel[row_i][block_i][coef_index] += delta[i]
                    block_path.append(coef_index)
                if len(block_path) != 0:
                    global_block = (row_i * self.hor_block_count) + block_i
                    hash_path += ''.join(['0', str(channel_i), str(global_block).zfill(len(str(self.ver_block_count*self.hor_block_count)))] + [str(x).zfill(2) for x in block_path] + ['0','0'])
                    path.append([channel_i, row_i, block_i, block_path])
                    b_i += 1
                    if b_i >= len(b_arr):
                        return hash_path, img
        raise Exception('Message is too long!')

    def compress(self, block, qm, t):
        return np.rint(np.divide(cv2.dct(np.rint(cv2.idct(np.multiply(block, qm[t-1])))), qm[t]))

    def genQFactor(self, q, m):
        s = 5000/q if q < 50 else 200-2*q
        op = lambda x: np.floor((s * x + 50)/100)
        return np.array([op(x) for x in m])

    def ditherAdjust(self, block, Y_flag, k=1, T=1, n=30):
        # do not dither dc coefs
        # https://www.sciencedirect.com/science/article/pii/S0165168420300013
        qm = list()
        if Y_flag:
            for q in range(90, 90-n-1, -1):
                qm.append(self.genQFactor(q, self.Y_quant_table))
        else:
            for q in range(90, 90-n-1, -1):
                qm.append(self.genQFactor(q, self.C_quant_table))
        qm = np.array(qm)
        block_original = block.copy()

        for t in range(1, n):
            block_tilde = block.copy()
            while True:
                if k > T:
                    break
                block = self.compress(block, qm, t)
                mod_ind = np.where(block % 2 != block_tilde % 2)
                if len(mod_ind[0]) and len(mod_ind[1]) == 0:
                    break
                for row_ind, col_ind in np.dstack(mod_ind)[0]:
                    if row_ind == 0 and col_ind == 0:
                        continue
                    if block[row_ind][col_ind] == block_original[row_ind][col_ind] + 1:
                        block[row_ind][col_ind] -= 2*k
                    elif block[row_ind][col_ind] == block_original[row_ind][col_ind] - 1:
                        block[row_ind][col_ind] += 2*k
                k += 1
            #block = np.divide(block, qm[t])
        return block

    def diffMancEnc(self, a):
        map_sign = lambda x: 1 if math.copysign(1, x) == 1 else 0
        x = np.zeros(len(a), dtype=np.uint8)
        x[0] = map_sign(a[0])
        i = 1
        while i < len(a):
            x[i] = map_sign(a[i-1]) ^ map_sign(a[i])
            i += 1
        return x

    def optimDMCSS(self, msg, img):
        rs_obj = rs(256)
        TAU = 3
        num_channels = img.shape[0]
        hash_path = ''
        path = list()
        avail_coefs = list()
        poly_coefs = list()
        H_hat = np.array([71,109], dtype=np.uint8)
        stc_obj = stc(H_hat)
        map_sign = lambda x: 1 if math.copysign(1, x) == 1 else 0
        block_perms = np.random.permutation(np.arange(num_channels * self.ver_block_count * self.hor_block_count))
        int_format = len(str(self.ver_block_count*self.hor_block_count))
        if int_format % 2 != 0: int_format += 1
        for block_num in block_perms:
            if len(avail_coefs) >= 2 * len(msg):
                avail_coefs = avail_coefs[:2*len(msg)]
                poly_coefs = poly_coefs[:2*len(msg)]
                m = np.array(list(msg), dtype=np.int_) # change if break np.uint8
                y, _ = stc_obj.generate(avail_coefs,m)
                y_polys = [rs_obj.encodeMsg(poly_coefs[j:j+rs_obj.K]).astype(np.int_) for j in range(0, len(poly_coefs), rs_obj.K)]
                parity_nums = list()
                bin_msg = ''
                for poly in y_polys:
                    parity_nums += list(poly[len(poly)-(rs_obj.N-rs_obj.K):])
                    poly = poly[:-(rs_obj.N-rs_obj.K)]
                    bin_poly = [format(num, '08b') for num in np.array(poly, dtype=np.uint8)]
                    bin_msg += ''.join([bit for bit in bin_poly])
                final_x = list()
                actual_i, effective_i = 0, 0
                while actual_i < len(path):
                    channel_i, row_i, block_i, coefs_ind = path[actual_i]
                    for coef_ind in coefs_ind:
                        if effective_i >= len(y):
                            path = path[:actual_i+1]
                            path[actual_i][3] = path[actual_i][3][:effective_i]
                            break
                        coef = img[channel_i][row_i][block_i][coef_ind]
                        img[channel_i][row_i][block_i][coef_ind] *= (-1)**(map_sign(coef) - y[effective_i])
                        final_x.append(img[channel_i][row_i][block_i][coef_ind])
                        effective_i += 1
                    actual_i += 1
                diff_manc = self.diffMancEnc(final_x)+1
                actual_i, effective_i = 0, 0
                while actual_i < len(path):
                    channel_i, row_i, block_i, coefs_ind = path[actual_i]
                    block_path = list()
                    for j, x in enumerate(coefs_ind):
                        if j+effective_i >= len(diff_manc):
                            break
                        block_path.append([x, diff_manc[j+effective_i]])
                    global_block = (row_i * self.hor_block_count) + block_i
                    hash_path += ''.join(['0', str(channel_i), str(global_block).zfill(int_format)] + [str(x).zfill(2) + str(y).zfill(2) for x, y in block_path] + ['0','0'])            
                    effective_i += len(coefs_ind)
                    actual_i += 1
                parity_nums = ''.join([str(x).zfill(4) for x in parity_nums])
                hash_path += 'PB'+parity_nums
                return hash_path, img
            channel_i = block_num // (self.ver_block_count * self.hor_block_count)
            row_i = (block_num % (self.ver_block_count * self.hor_block_count)) // self.hor_block_count
            block_i = (block_num % (self.ver_block_count * self.hor_block_count)) % self.hor_block_count
            channel = img[channel_i]
            qcomp = self.genQFactor(60, self.Y_quant_table if channel_i == 0 else self.C_quant_table)
            block = channel[row_i][block_i]
            # compress block
            comp_block = np.rint(np.divide(np.multiply(block.copy().reshape((self.BLOCK_SIZE,self.BLOCK_SIZE)), self.Y_quant_table if channel_i == 0 else self.C_quant_table), qcomp)).reshape((self.BLOCK_SIZE*self.BLOCK_SIZE))
            coef_mask = np.array([0 < abs(coef) < TAU for coef in comp_block])
            coef_mask[0] = False # ignore dc coef
            coefs = np.extract(coef_mask, block)
            coefs_ind = np.where(coef_mask == True)[0]
            poly_coefs += list(coefs)
            avail_coefs += [map_sign(x) for x in coefs]
            path.append([channel_i, row_i, block_i, list(coefs_ind)])
        raise Exception('Message too long!')

    def dmcss(self, msg, img):
        # do stc at end after gathering all coefs + locations?
        TAU = 3
        num_channels = img.shape[0]
        msg_i = 0
        hash_path = ''
        path = list()
        H_hat = np.array([71,109], dtype=np.uint8)
        stc_obj = stc(H_hat)
        map_sign = lambda x: 1 if math.copysign(1, x) == 1 else 0
        block_perms = np.random.permutation(np.arange(num_channels * self.ver_block_count * self.hor_block_count))
        for block_num in block_perms:
            if msg_i >= len(msg):
                return hash_path, img
            channel_i = block_num // (self.ver_block_count * self.hor_block_count)
            row_i = (block_num % (self.ver_block_count * self.hor_block_count)) // self.hor_block_count
            block_i = (block_num % (self.ver_block_count * self.hor_block_count)) % self.hor_block_count
            channel = img[channel_i]
            qcomp = self.genQFactor(60, self.Y_quant_table if channel_i == 0 else self.C_quant_table)
            block = channel[row_i][block_i]
            # compress block
            comp_block = np.rint(np.divide(np.multiply(block.copy().reshape((self.BLOCK_SIZE,self.BLOCK_SIZE)), self.Y_quant_table if channel_i == 0 else self.C_quant_table), qcomp)).reshape((self.BLOCK_SIZE*self.BLOCK_SIZE))
            coef_mask = np.array([0 < abs(coef) < TAU for coef in comp_block])
            coef_mask[0] = False # ignore dc coef
            coefs = np.extract(coef_mask, block)
            if len(coefs) < 8:
                continue
            coefs_ind = np.where(coef_mask == True)[0]
            x = [map_sign(x) for x in coefs]
            if msg_i + len(x) // 2 < len(msg):
                m = msg[msg_i:(msg_i + len(x)//2)]
            else:
                m = msg[msg_i:]
            m = np.array(list(m), dtype=np.uint8)
            msg_i += len(x) // 2
            y, _ = stc_obj.generate(x,m)
            block_path = list()
            final_coefs = list()
            for y_i, coef_i in enumerate(coefs_ind):
                block[coef_i] *= (-1)**(map_sign(block[coef_i]) - y[y_i])
                final_coefs.append(block[coef_i])
            diff_manc = self.diffMancEnc(final_coefs)+1
            block_path = [[x, diff_manc[i]] for i, x in enumerate(coefs_ind)]
            # format path properly
            path.append([channel_i, row_i, block_i, block_path])
            int_format = len(str(self.ver_block_count*self.hor_block_count))
            if int_format % 2 != 0: int_format += 1
            global_block = (row_i * self.hor_block_count) + block_i
            hash_path += ''.join(['0', str(channel_i), str(global_block).zfill(len(str(self.ver_block_count*self.hor_block_count)))] + [str(x).zfill(2) + str(y).zfill(2) for x, y in block_path] + ['0','0'])
        raise Exception('Message too long!')

    def drF5(self, msg, img):
        hash_path = ''
        path = list()
        msg_i = 0
        num_channels = img.shape[0]
        block_perms = np.random.permutation(np.arange(num_channels * self.ver_block_count * self.hor_block_count))
        H_hat = np.array([71,109], dtype=np.uint8)
        stc_obj = stc(H_hat)
        for block_num in block_perms:
            if msg_i >= len(msg):
                return hash_path, img
            channel_i = block_num // (self.ver_block_count * self.hor_block_count)
            row_i = (block_num % (self.ver_block_count * self.hor_block_count)) // self.hor_block_count
            block_i = (block_num % (self.ver_block_count * self.hor_block_count)) % self.hor_block_count
            channel = img[channel_i]
            block = channel[row_i][block_i]
            coef_mask = np.array([abs(coef) > 0 for coef in block])
            coef_mask[0] = False # ignore dc coef
            coefs = np.extract(coef_mask, block)
            if len(coefs) < 8:
                continue
            coefs_ind = np.where(coef_mask == True)[0]
            x = [int(x%2) for x in coefs]
            if msg_i + len(x) // 2 < len(msg):
                m = msg[msg_i:(msg_i + len(x)//2)]
            else:
                m = msg[msg_i:]
            m = np.array(list(m), dtype=np.uint8)
            msg_i += len(x) // 2
            y, _ = stc_obj.generate(x,m)
            block_path = list()
            for y_i, coef_i in enumerate(coefs_ind):
                block[coef_i] += y[y_i] - block[coef_i]%2
                block_path.append(coef_i) #row, coef
            # we now have an stc encoded block, so we must now perform the dither adjustment
            block = self.ditherAdjust(block.reshape((8,8)), True if channel_i == 0 else False)
            path.append([channel_i, row_i, block_i, block_path])
            int_format = len(str(self.ver_block_count*self.hor_block_count))
            if int_format % 2 != 0: int_format += 1
            global_block = (row_i * self.hor_block_count) + block_i
            hash_path += ''.join(['0', str(channel_i), str(global_block).zfill(len(str(self.ver_block_count*self.hor_block_count)))] + [str(x).zfill(2) for x in block_path] + ['0','0'])
        raise Exception('Message too long!')

    def F5(self, msg, img):
        num_channels = img.shape[0]
        # c1, c2, c3 = y,cb,cr
        path = list()
        i, j = 0, 1
        # random permutation of row indices
        # generate a new permutation of blocks for each row change
        block_perms = np.random.permutation(np.arange(num_channels * self.ver_block_count * self.hor_block_count))
        for block_num in block_perms:
            if i >= len(msg):
                path = self.formatPath(np.array(path))
                return path, img
            channel_i = block_num // (self.ver_block_count * self.hor_block_count)
            row_i = (block_num % (self.ver_block_count * self.hor_block_count)) // self.hor_block_count
            block_i = (block_num % (self.ver_block_count * self.hor_block_count)) % self.hor_block_count
            channel = img[channel_i]
            j = 1
            while j < 64:
                host = channel[row_i][block_i][j]
                if host != 0.:
                    try:
                        msg_bit = int(msg[i])
                    except:
                        break
                    #print(f"host:{host} lsb:{lsbF5(host)} bit: {msg_bit}")
                    if self.lsbF5(host) == msg_bit:
                        #print(msg_bit, i, "MATCH:", host)
                        path.append(np.array([channel_i, row_i, block_i, j]))
                        i+=1
                    else:
                        #print(msg_bit, i, ": NO MATCH:", host, "->", host, "-", math.copysign(1,host))
                        host = host - math.copysign(1, host)
                        if host != 0:
                            channel[row_i][block_i][j] = host
                            path.append(np.array([channel_i, row_i, block_i, j]))
                            i+=1
                j+=1
        raise Exception('Message too long!')

    def LSB(self, msg, img):
        msg_i = 0
        path = list()
        # choosing to store in the last 10 ac coefficients to reduce artefacts
        START_COEF = 1
        END_COEF = 64
        num_channels = img.shape[0]
        valid_indices = np.arange(START_COEF,END_COEF)
        block_perms = np.arange(num_channels * self.ver_block_count * self.hor_block_count)
        for block_num in block_perms:
            channel_i = block_num // (self.ver_block_count * self.hor_block_count)
            row_i = (block_num % (self.ver_block_count * self.hor_block_count)) // self.hor_block_count
            block_i = (block_num % (self.ver_block_count * self.hor_block_count)) % self.hor_block_count
            channel = img[channel_i]
            for coef_i in np.random.permutation(valid_indices):
                if msg_i >= len(msg):
                    path = self.formatPath(np.array(path))
                    return path, img
                bit = msg[msg_i]
                chosen_coef = int(channel[row_i][block_i][coef_i])
                if chosen_coef == 0:
                    continue
                if str(chosen_coef % 2) != bit:
                    img[channel_i][row_i][block_i][coef_i] += 1
                path.append(np.array([channel_i, row_i, block_i, coef_i]))
                msg_i += 1
        raise Exception('Message is too long!')

    def formatPath(self, path):
        path = path+1 # so we can use 00 as an end-of-block marker
        row_format = len(str(self.ver_block_count))
        block_format = len(str(self.hor_block_count))
        if row_format % 2 != 0:
            row_format += 1
        if block_format % 2 != 0:
            block_format += 1
        path_string = ''
        for bit_loc in path:
            path_string += str(bit_loc[0]).zfill(2) + str(bit_loc[1]).zfill(row_format) + str(bit_loc[2]).zfill(block_format) \
                + str(bit_loc[3]).zfill(2) + '00'
        return path_string

    def hashPath(self, path):
        byte_path = str.encode(path)
        key = b'Sixteen byte key'
        #print("Your key is: ", key.decode())
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(byte_path)

        file_out = open("path_key.bin", "wb")
        [ file_out.write(x) for x in (cipher.nonce, tag, ciphertext) ]
        file_out.close()

        return 0

    def encode(self, img_name, message_path, func=2, verbose=True, use_rs=True, output_name="stego"):
        img, greyscale = self.__readImage(img_name)
        if not greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        with open('.v_imgdim', 'wb') as fp:
            pickle.dump((self.img_height, self.img_width), fp)
        if self.img_width % self.BLOCK_SIZE != 0:
            img = self.__padImageWidth(img)
        if self.img_height % self.BLOCK_SIZE != 0:
            img = self.__padImageHeight(img)
        
        new_img_height, new_img_width = self.getImageDimensions(img)
        self.hor_block_count, self.ver_block_count = new_img_width // self.BLOCK_SIZE, new_img_height // self.BLOCK_SIZE
        total_blocks = self.ver_block_count * self.hor_block_count
        with open('.imgdim', 'wb') as fp:
            pickle.dump((new_img_height, new_img_width), fp)

        if not greyscale:
            Y_img, Cr_img, Cb_img = cv2.split(img)
            img = self.blockify([Y_img, Cb_img, Cr_img])
            #print("Separated successfully")
        else:
            img = self.blockify([img])

        #print("beginning dct...")
        img = self.DCT_2(img)
        #print("finished dct")

        img = self.quantizeAndRound(img)
        #print("finished quantization and round")

        img = self.zigZagEncode(img)
        #print("finished zigzag")

        #print("encoding message...")
        try:
            with open(message_path, 'r') as f:
                message = f.read()
        except:
            raise FileNotFoundError('Could not find message file, is the path correct?')
        bin_msg = self.messageConv(message)
        if use_rs:
            rs_obj = rs(self.RS_PARAM)
            message_polys = rs_obj.prepareMessage(bin_msg)
            bin_msg = ''
            for message_poly in message_polys:
                bin_poly = [format(num, '08b') for num in np.array(message_poly, dtype=np.uint8)]
                bin_msg += ''.join([bit for bit in bin_poly])
        if func == 0:
            hash_path, img = self.F5(bin_msg, img)

        elif func == 1:
            hash_path, img = self.sdcsF5(bin_msg, img)

        elif func == 2:
            hash_path, img = self.optimDMCSS(bin_msg, img)
        
        elif func == 3:
            hash_path, img = self.LSB(bin_msg, img)
        
        else:
            raise ValueError('Algorithm must be:\n0: F5\n1: SDCS F5\n2: drF5')
        self.hashPath(hash_path)
        #print("encoded and written path to file")

        if verbose:
            # verbose mode outputs jpeg as txt and completes all encoding steps
            img = self.RLEandDPCM(img)
            print("finished rle")

            bitstring = self.huffman(img)
            final_file = open(output_name+".txt", "w")
            final_file.write(bitstring)
            final_file.close()
            print("done!")
        
        else:
            from decoder import decoder
            decoder_obj = decoder(self.BLOCK_SIZE, self.RS_PARAM)
            decoder_obj.defineBlockCount(self.ver_block_count, self.hor_block_count)
            img = [channel.reshape((total_blocks, self.BLOCK_SIZE*self.BLOCK_SIZE)) for channel in img]
            img = decoder_obj.unZigZag(img)
            img = decoder_obj.deQuantize(img)
            img = decoder_obj.DCT_3(img)
            if not greyscale:
                img = np.clip(decoder_obj.YCbCr2BGR(img), 0,255)
            else:
                img = np.clip(img, 0, 255)[0]
            img = decoder_obj.assembleImage(img)
            if self.img_height != new_img_height:
                img = decoder_obj.removeVPadding(img, new_img_height)
            if self.img_width != new_img_width:
                img = decoder_obj.removeHPadding(img, new_img_width)
            #cv2.imwrite(output_name+".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if not greyscale:
                jpeg_bytes = simplejpeg.encode_jpeg(img.astype(np.uint8), 100, 'BGR', '444', False)
                with open(output_name+".jpg", "wb") as f:
                    f.write(jpeg_bytes)
            else:
                cv2.imwrite(output_name+".jpg", img)
            #print("done!")

########################################
########PROGRAM BEGINS HERE#############
########################################
#"./bossbase/1.pgm"
#encoder_obj = encoder(8, 256)
#encoder_obj.encode("./bossbase/1.pgm", "message.txt", func=2, verbose=False, use_rs=False)

key = 'Sixteen byte key'
from decoder import decoder
decoder_obj = decoder(8, 256)
decoder_obj.decode('stego', bytes(key, "utf8"), func=2, verbose=False, use_rs=False, greyscale=True)

#none type poly in optimaldmcssdecode HUH?? //FIXED was not returning if no errors!

#img = cv2.imread("images/fagen.png", cv2.IMREAD_COLOR)
#jpeg_bytes = simplejpeg.encode_jpeg(img, 100, 'BGR', '444', False)
#with open("test.jpg", "wb") as f:
#    f.write(jpeg_bytes)

"""
test_block=np.array([[10., -2., -4., -0.,  0.,  0., -0., 1.],
 [ 2.,  1.,  0., -1., -1.,  0.,  0., -0.],
 [-0., -0.,  0., -0., -0., -0.,  0.,  0.],
 [ 0., -0., -0.,  0.,  0.,  0.,  0., -0.],
 [-0., -0., -0., -0.,  0.,  0.,  0., 0.],
 [ 0.,  0., -0.,  0., -0., -0.,  0.,  0.],
 [-0.,  0., -0., -0.,  0.,  0., -0., -0.],
 [-0., -0., -0., -0.,  0.,  0., -0., -0.]])
 
new_block = encoder_obj.ditherAdjust(test_block, True)
new_block = encoder_obj.compress(new_block, [encoder_obj.Y_quant_table], 0)
y = np.array([new_block[0][1], new_block[0][2], new_block[0][4], new_block[0][7], new_block[1][0], new_block[1][1], new_block[1][3], new_block[1][4]]) % 2
print(y)
new_block = encoder_obj.compress(test_block, [encoder_obj.Y_quant_table], 0)
y2 = np.array([new_block[0][1], new_block[0][2], new_block[0][4], new_block[0][7], new_block[1][0], new_block[1][1], new_block[1][3], new_block[1][4]]) % 2
print(y2)
"""

# can we embed the path into the header? only 2kb...

# ensure message isnt embedded in padded bits?

# perform DCT transform.....


# divide the dct-transformed 8x8 block by the selected quantization table
# and round to nearest integer


# zig zag encoding



# generate pseudo-random path for encoding message along
# and encode message along path


# encode DC coefficients ([0][0]) using DPCM
# encode AC coefficients using RLE


# Huffman coding

# this is done then! you can simply write bitstring to a file.


###########################################################
############# extra shit in case i need it ################
###########################################################

"""
#Y_img = DCT_2(Y_img)
block = Y_img[0][0]
block = np.array([[
    144,139,149,155,153,155,155,155],	
[151, 151, 151,	159, 156, 156, 156, 158],
[151, 156, 160, 162, 159, 151, 151, 151],
[158, 163, 161, 160, 160, 160, 160, 161],
[158, 160, 161, 162, 160, 155, 155, 156],
[161, 161, 161, 161, 160, 157, 157, 157],
[162, 162, 161, 160, 161, 157, 157, 157],
[162, 162, 161, 160, 163, 157, 158, 154
]])
block = np.array([  [16,  11,  10,  16,  24,  40,  51,  61],
  [12,  12,  14,  19,  26,  58,  60,  55],
  [14,  13,  16,  24,  40,  57,  69,  56],
  [14,  17,  22,  29,  51,  87,  80,  62],
  [18,  22,  37,  56,  68, 109, 103,  77],
  [24,  35,  55,  64,  81, 104, 113,  92],
  [49,  64,  78,  87, 103, 121, 120, 101],
  [72,  92,  95,  98, 112, 100, 103,  99] ])
out_block = np.zeros((8,8))
for k in range(block_size):
    for l in range(block_size):
        sigma_sum = 0 
        for i in range(block_size):
            for j in range(block_size):
                Bij = block[i][j]
                sigma_sum += ((w(k)*w(l))/4)*math.cos((math.pi/16)*k*((2*i)+1))*math.cos((math.pi/16)*l*((2*j)+1))*Bij
        out_block[k][l] = sigma_sum
"""

"""
try:
                    if zz_img[row_block_i][block_i][ac_i+1] != cur_num:
                        ac_rle.append([1,cur_num])
                        ac_i += 1
                        continue
                    else:
                        cur_num_count = 0
                        while zz_img[row_block_i][block_i][ac_i] == cur_num:
                            cur_num_count += 1
                            if ac_i < len(zz_img[row_block_i][block_i])-1:
                                ac_i += 1
                            else:
                                ac_i = len(zz_img[row_block_i][block_i])+1
                                break
                        ac_i -= 1
                        ac_rle.append([cur_num_count,cur_num])
                except:
                    ac_rle.append([1,cur_num])
                    ac_i += 1
                    continue
                ac_i += 1
"""

"""
test_block = np.array([[96., -1.,  2.,  1., -0., -0., -0., 0.],
 [-2., -1.,  2.,  0., -0.,  0., -0., -0.],
 [ 1., -0., -1., -0., -0.,  0.,  0., -0.],
 [ 0., -0., -0., -0., -0.,  0.,  0., -0.],
 [ 0., -0., -0.,  0.,  0.,  0., -0.,  0.],
 [ 0.,  0., -0.,  0.,  0., -0.,  0., 0.],
 [ 0.,  0.,  0., -0., -0.,  0., -0., 0.],
 [ 0.,  0., -0., -0.,  0., -0.,  0., -0.]])

print(np.rint(np.divide(cv2.dct(np.rint(cv2.idct(np.multiply(test_block, Y_quant_table)))), Y_quant_table)))

coef_mask = np.array([abs(coef) > 0 for coef in test_block])
coef_mask[0][0] = False # ignore dc coef
coefs = np.extract(coef_mask, test_block)
coefs_ind = np.dstack(np.where(coef_mask == True))[0]
x = np.array([int(x%2) for x in coefs])
m = np.array([0,1,1,1])
stc_obj = stc(np.array([71,109], dtype=np.uint8))
y, _ = stc_obj.generate(x,m)
H = stc_obj.gen_H(x,m)
assert np.array_equal((H @ y) % 2, m)
for y_i, index in enumerate(coefs_ind):
    test_block[index[0]][index[1]] += y[y_i] - test_block[index[0]][index[1]]%2
test_block = ditherAdjust(test_block, True)
print(test_block)
ext_y = [int(test_block[x[0]][x[1]]%2) for x in coefs_ind]
print(ext_y)
print((H @ ext_y) % 2, m)
exit(0)
"""

"""
def genRandomPath(bin_msg, Y_zz_img, Cb_zz_img, Cr_zz_img):
    bit_locations = []
    # choosing to store in the last 10 ac coefficients to reduce artefacts
    START_COEF = 53
    END_COEF = START_COEF + MAX_COEF_NUM
    valid_indices = range(START_COEF,END_COEF)
    for bit in bin_msg:
        component = random.randrange(0,3)
        cur_comp = []
        if component == 0:
            cur_comp = Y_zz_img
        elif component == 1:
            cur_comp = Cb_zz_img
        elif component == 2:
            cur_comp = Cr_zz_img
        valid_index = False
        while not valid_index:
            rand_row, rand_col = random.randrange(hor_block_count), random.randrange(ver_block_count)
            index = random.choice(valid_indices)
            two_d_location = (rand_row * hor_block_count) + rand_col
            if [component, two_d_location, index] not in bit_locations:
                chosen_coef = cur_comp[rand_row][rand_col][index]
                bin_string = bin(int(chosen_coef))
                if bin_string[-1] != bit:
                    if int(chosen_coef) >= 0:
                        cur_comp[rand_row][rand_col][index] = float(int(chosen_coef) + 1)
                    elif int(chosen_coef) < 0:
                        cur_comp[rand_row][rand_col][index] = float(int(chosen_coef) - 1)
                bit_locations.append([component, two_d_location, index])
                valid_index = True
    return bit_locations, Y_zz_img, Cb_zz_img, Cr_zz_img
"""