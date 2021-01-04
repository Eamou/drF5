import numpy as np
import cv2
import math
import random
import pickle

# to-do:
# 1. enable program to work with any image dimension //done?
# 2. enable chroma subsampling - not required but might be nice
# 3. jpeg quality options?
# 4. command line options for the 2 and 3

# quantization tables - these will be changed later
# possibly stored in a file?
# option to change encoding quality added later

# use a unique hamming/reed-solomon code to encode each letter, ensuring each letter
# uses any particular block no more than once. this way, if there are multiple bits in a block
# that is corrupted by cropping, the error correcting codes still need only correct one error.
# hence, hamming codes can be used. reed-solomon will allow for higher payloads
# as more letters will be included in an error-correcting chunk, reducing redundancy
# overheads and increasing overall payload size.

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

# Huffman tables for DC and AC values
# yes they are massive, I think this is probably
# the fastest and most sensible way to implement them
# considering I have to pull the inverse too.

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

def readImage(image_name):
    # return image object img
    return cv2.imread('./images/'+image_name,cv2.IMREAD_COLOR)

def getImageDimensions(img):
    # return image height, width as integers
    return img.shape[0], img.shape[1]

def displayImage(img):
    # show image in new window until key pressed
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blockify(img):
    img_tiles = [] # transform image into series of 8x8 blocks
    for row in range(0, img_height, BLOCK_SIZE):
        img_tiles_row = [] # fill in row by row
        for column in range(0, img_width, BLOCK_SIZE):
            column_end = column + BLOCK_SIZE
            row_end = row + BLOCK_SIZE
            # select the 8x8 tile
            tile = img[row:row_end, column:column_end]
            # add it to the row array
            img_tiles_row.append(tile)
        # append rows individually to the image matrix
        # this ensure the dimensions are consistent
        img_tiles.append(img_tiles_row)
    return img_tiles

def YCbCr_convert(bgr):
    # values from https://wikipedia.org/wiki/YCbCr#JPEG_conversion
    B, G, R = bgr
    Y = (0.299*R)+(0.587*G)+(0.114*B)
    Cb = 128 - (0.168736*R) - (0.331264*G) + (0.5 * B)
    Cr = 128 + (0.5 * R) - (0.418688 * G) - (0.081312 * B)
    return Y, Cb, Cr

def BGR2YCbCr(img_tiles):
    # perform conversion for each pixel
    Y_img, Cb_img, Cr_img = [], [], []
    # I know this looks bad but it's only O(n^2)!
    for row in range(ver_block_count):
        Y_tiles, Cb_tiles, Cr_tiles = [], [], []
        for column in range(hor_block_count):
            Y_block, Cb_block, Cr_block = np.zeros((8,8)), np.zeros((8,8)), np.zeros((8,8))
            for block in range(BLOCK_SIZE):
                for pixel_i in range(BLOCK_SIZE):
                    pixel = np.array(YCbCr_convert(img_tiles[row][column][block][pixel_i]))
                    img_tiles[row][column][block][pixel_i] = pixel
                    # shift by -128 when using own dct!
                    Y_block[block][pixel_i], Cb_block[block][pixel_i], Cr_block[block][pixel_i] = pixel
            Y_tiles.append(Y_block)
            Cb_tiles.append(Cb_block)
            Cr_tiles.append(Cr_block)
        Y_img.append(np.array(Y_tiles))
        Cb_img.append(np.array(Cb_tiles))
        Cr_img.append(np.array(Cr_tiles))     
    return img_tiles, np.array(Y_img), np.array(Cb_img), np.array(Cr_img)

def w(k_num):
    # for use in DCT transformation
    if k_num == 0:
        return 1/math.sqrt(2)
    else:
        return 1

def DCT_2(Y_img):
    # transform Y values into DCT coefficients
    # i think this is O(N^2) as it goes through the whole block
    # for each value in the block. definitely the slowest part of the process either way
    dct_img = []
    for row_block in Y_img:
        dct_img_row = []
        for block in row_block:
            out_block = cv2.dct(block)
            dct_img_row.append(out_block)
        dct_img.append(np.array(dct_img_row))
    return np.array(dct_img)

def quantizeAndRound(Y_img, Y_flag):
    # quantizes DCT coefs in-place using quant_table_2 atm (add quality options later)
    # then rounds to nearest integer
    Y_img_len = len(Y_img)
    for row_block_i in range(Y_img_len):
        row_len = len(Y_img[row_block_i])
        for block_i in range(row_len):
            # divide by relevant quantization table
            if Y_flag:
                np.divide(Y_img[row_block_i][block_i], Y_quant_table, Y_img[row_block_i][block_i])
            else:
                np.divide(Y_img[row_block_i][block_i], C_quant_table, Y_img[row_block_i][block_i])
            # round to nearest int
            np.rint(Y_img[row_block_i][block_i], Y_img[row_block_i][block_i])
    return Y_img


def zigZagEncode(Y_img):
    # i know this looks horrible but it is honestly the fastest way to do it!
    # convert 8x8 block of dct coef's into a 64-len array via zig zag arrangement
    zz_image = []
    for row_block in Y_img:
        zz_row = []
        for out_block in row_block:
            zz_array = np.zeros(64)
            zz_array[0] = out_block[0][0]
            zz_array[1] = out_block[0][1]
            zz_array[2] = out_block[1][0]
            zz_array[3] = out_block[2][0]
            zz_array[4] = out_block[1][1]
            zz_array[5] = out_block[0][2]
            zz_array[6] = out_block[0][3]
            zz_array[7] = out_block[1][2]
            zz_array[8] = out_block[2][1]
            zz_array[9] = out_block[3][0]
            zz_array[10] = out_block[4][0]
            zz_array[11] = out_block[3][1]
            zz_array[12] = out_block[2][2]
            zz_array[13] = out_block[1][3]
            zz_array[14] = out_block[0][4]
            zz_array[15] = out_block[0][5]
            zz_array[16] = out_block[1][4]
            zz_array[17] = out_block[2][3]
            zz_array[18] = out_block[3][2]
            zz_array[19] = out_block[4][1]
            zz_array[20] = out_block[5][0]
            zz_array[21] = out_block[6][0]
            zz_array[22] = out_block[5][1]
            zz_array[23] = out_block[4][2]
            zz_array[24] = out_block[3][3]
            zz_array[25] = out_block[2][4]
            zz_array[26] = out_block[1][5]
            zz_array[27] = out_block[0][6]
            zz_array[28] = out_block[0][7]
            zz_array[29] = out_block[1][6]
            zz_array[30] = out_block[2][5]
            zz_array[31] = out_block[3][4]
            zz_array[32] = out_block[4][3]
            zz_array[33] = out_block[5][2]
            zz_array[34] = out_block[6][1]
            zz_array[35] = out_block[7][0]
            zz_array[36] = out_block[7][1]
            zz_array[37] = out_block[6][2]
            zz_array[38] = out_block[5][3]
            zz_array[39] = out_block[4][4]
            zz_array[40] = out_block[3][5]
            zz_array[41] = out_block[2][6]
            zz_array[42] = out_block[1][7]
            zz_array[43] = out_block[2][7]
            zz_array[44] = out_block[3][6]
            zz_array[45] = out_block[4][5]
            zz_array[46] = out_block[5][4]
            zz_array[47] = out_block[6][3]
            zz_array[48] = out_block[7][2]
            zz_array[49] = out_block[7][3]
            zz_array[50] = out_block[6][4]
            zz_array[51] = out_block[5][5]
            zz_array[52] = out_block[4][6]
            zz_array[53] = out_block[3][7]
            zz_array[54] = out_block[4][7]
            zz_array[55] = out_block[5][6]
            zz_array[56] = out_block[6][5]
            zz_array[57] = out_block[7][4]
            zz_array[58] = out_block[7][5]
            zz_array[59] = out_block[6][6]
            zz_array[60] = out_block[5][7]
            zz_array[61] = out_block[6][7]
            zz_array[62] = out_block[7][6]
            zz_array[63] = out_block[7][7]
            zz_row.append(zz_array)
        zz_image.append(np.array(zz_row))
    return np.array(zz_image)

def RLEandDPCM(zz_img):
    # create array of all DC values, encoded using DPCM - each value is the difference
    # from the previous value rather than the actual value
    # create array of RLE-encoded AC values - [skip, value]
    # where skip is the number of zeroes preceeding value.
    # [0,0] indicates the end of the block and is appended to the end
    dc_array, ac_arrays = [], []
    zz_img_len = len(zz_img)
    for row_block_i in range(zz_img_len):
        row_len = len(zz_img[row_block_i])
        for block_i in range(row_len):
            ac_rle = []
            if block_i == 0 and row_block_i == 0:
                # encode the first DC value as-is
                dc_array.append(zz_img[row_block_i][block_i][0])
            else:
                # for the rest, encode the difference
                if block_i != 0:
                    dc_array.append(zz_img[row_block_i][block_i][0] - zz_img[row_block_i][block_i-1][0])
                else:
                    dc_array.append(zz_img[row_block_i][block_i][0] - zz_img[row_block_i-1][block_i-1][0])
            # start at 1 to skip the DC value
            ac_i = 1
            zero_count = 0
            # max zero count is to keep track of how many 16-0's there are
            # these should only be added to the ac_rle if there is a non-zero
            # coefficient after them. otherwise, [0,0] should follow the final
            # non-zero coefficient.
            max_zero_count = 0
            while ac_i < len(zz_img[row_block_i][block_i]):
                cur_num = zz_img[row_block_i][block_i][ac_i]
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
                        for i in range(max_zero_count):
                            ac_rle.append([15,0])
                    ac_rle.append([zero_count, cur_num])
                    zero_count = 0
                    ac_i += 1
                    continue
            # append end of block marker
            ac_rle.append([0,0])
            ac_arrays.append(ac_rle)
    return np.array(dc_array), ac_arrays

def categorize(coef):
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

def onesComp(bitstring):
    oc_bitstring = ''
    for char in bitstring:
        if char == '1':
            oc_bitstring += '0'
        elif char == '0':
            oc_bitstring += '1'
    return oc_bitstring


def huffman(Y_dc_arr, Y_ac_arr, Cb_dc_arr, Cb_ac_arr, Cr_dc_arr, Cr_ac_arr):
    # compute and create final bitstring of data
    # dc and ac arrays should have same length, so can just use one
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
            dc_category = categorize(dc_arr[index])
            dc_codeword = dc_codeword_dict[dc_category]
            # the [2:] is to remove the '0b' from the front of the binary string
            dc_magnitude = bin(int(dc_arr[index]))[2:]
            if int(dc_arr[index]) < 0:
                dc_magnitude = onesComp(dc_magnitude)
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
                ac_category = categorize(ac_value)
                ac_magnitude = bin(ac_value)[2:]
                if ac_value < 0:
                    ac_magnitude = onesComp(ac_magnitude)
                ac_codeword = ac_codeword_dict[(ac_skip, ac_category)]
                if ac_codeword != '1010' and ac_codeword != '11111111001':
                    ac_bitstring = ac_codeword + ac_magnitude
                else:
                    ac_bitstring = ac_codeword
                bitstring += ac_bitstring
            YCbCr_num += 1
    return bitstring

def messageConv(message):
    bin_string = ""
    for char in message:
        bin_val = bin(ord(char))[2:]
        while len(bin_val) < 7:
            bin_val = '0' + bin_val
        bin_string += bin_val
    return bin_string

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

def padImageHeight(img):
    # repeat last row of pixels until dimension is multiple of 8
    while len(img) % BLOCK_SIZE != 0:
        img = np.append(img, [img[len(img)-1]], axis=0)
    return img

def padImageWidth(img):
    # repeat last column of pixels until dimension is multiple of 8
    img_list = list(img)
    width = getImageDimensions(img)[1]
    while width % BLOCK_SIZE != 0:
        for row_index in range(len(img)):
            row_list = list(img_list[row_index])
            pixel_list = list(row_list[-1])
            row_list.append(pixel_list)
            img_list[row_index] = row_list
        width += 1
    return np.array(img_list)

def findMaxPayload(img_height, img_width):
    return (img_height // BLOCK_SIZE * img_width // BLOCK_SIZE) * MAX_COEF_NUM

def writeHeader(bitstring):
    # byte is 16 bits
    ####### UNDER CONSTRUCTION #######
    
    # SOI is always the same
    SOI = 'FFD8'
    
    # JFIF takes some work
    JFIF_APP0 = 'FFE0'
    JFIF_LEN = '0010' # calculated based on the rest
    JFIF_ID = '4A46494600' # doesn't change
    JFIF_VER = '0101' # version 1.02
    JFIF_UNITS = '00' # specified via Xdensity and Ydensity as an aspect ratio
    JFIF_Xden = '0001' # set as default values, change if needed
    JFIF_Yden = '0001'
    aspect_ratio = img_width_copy // img_height_copy
    if aspect_ratio == 0:
        JFIF_Xden = '0001'
        JFIF_Yden = '0001'
    JFIF_XThumb = '00'
    JFIF_YThumb = '00'
    JFIF = JFIF_APP0 + JFIF_LEN + JFIF_ID + JFIF_VER + JFIF_UNITS + JFIF_Xden + JFIF_Yden + JFIF_XThumb + JFIF_YThumb
    
    # quantization table(s)
    DQT = 'FFDB'

    EOI = 'FFD9'

########################################
########PROGRAM BEGINS HERE#############
########################################

# constants
MAX_COEF_NUM = 10
BLOCK_SIZE = 8

# read image (ability to input image name to be added later)
# get image dimensions
image_name = 'fagen.png'
img = readImage(image_name)
img_height, img_width = getImageDimensions(img)
img_height_copy, img_width_copy = img_height, img_width
MAX_PAYLOAD = findMaxPayload(img_height, img_width)

# store original image dimensions
with open('.v_imgdim', 'wb') as fp:
    pickle.dump((img_height, img_width), fp)

# adjust image with padding to enable 8x8 blocks
if img_width % BLOCK_SIZE != 0:
    img = padImageWidth(img)
elif img_height % BLOCK_SIZE != 0:
    img = padImageHeight(img)

# new dimensions
img_height, img_width = getImageDimensions(img)
with open('.imgdim', 'wb') as fp:
    pickle.dump((img_height, img_width), fp)

# ensure message isnt embedded in padded bits?

"""
try:
    message = str(input("Enter message: "))
except:
    print("Please enter a string")
    quit(1)
"""
message = "lttstore.com"
bin_msg = messageConv(message)
if len(bin_msg) > MAX_PAYLOAD:
    raise ValueError('Message too long')

# split image into 8x8 blocks and store in img_tiles
# note that this is a downsampling ratio of 4:4:4 (no downsampling), others added later?
hor_block_count = img_width // BLOCK_SIZE
ver_block_count = img_height // BLOCK_SIZE

img_tiles = blockify(img)
# convert from list to numpy array
img_tiles = np.array(img_tiles)

# values to floats for DCT
img_tiles = [np.float32(tile) for tile in img_tiles]

# convert BGR to YCbCr
# the image is in img_tiles with YCbCr pixels
# Y_img contains the 8x8 blocks of just the Y values for use in DCT

img_tiles, Y_img, Cb_img, Cr_img = BGR2YCbCr(img_tiles)
print("Separated successfully")

# perform DCT transform.....
print("beginning dct...")
Y_img_dct = DCT_2(Y_img)
Cb_img_dct = DCT_2(Cb_img)
Cr_img_dct = DCT_2(Cr_img)
print("finished dct")

# divide the dct-transformed 8x8 block by the selected quantization table
# and round to nearest integer

Y_img_quant = quantizeAndRound(Y_img_dct, True)
Cb_img_quant = quantizeAndRound(Cb_img_dct, False)
Cr_img_quant = quantizeAndRound(Cr_img_dct, False)
print("finished quantization and round")

# zig zag encoding

Y_zz_img = zigZagEncode(Y_img_quant)
Cb_zz_img = zigZagEncode(Cb_img_quant)
Cr_zz_img = zigZagEncode(Cr_img_quant)
print("finished zigzag")
#print("zz", Y_zz_img)
# generate pseudo-random path for encoding message along
# and encode message along path

print("encoding message...")
encode_path, Y_zz_img, Cb_zz_img, Cr_zz_img = genRandomPath(bin_msg, Y_zz_img, Cb_zz_img, Cr_zz_img)
#print(len(bin_msg), bin_msg, encode_path)
with open('.msgpath', 'wb') as fp:
    pickle.dump(encode_path, fp)
print("encoded and written path to file")

# encode DC coefficients ([0][0]) using DPCM
# encode AC coefficients using RLE

Y_dc_arr, Y_ac_arr = RLEandDPCM(Y_zz_img)
Cb_dc_arr, Cb_ac_arr = RLEandDPCM(Cb_zz_img)
Cr_dc_arr, Cr_ac_arr = RLEandDPCM(Cr_zz_img)
print("finished rle")
#print("rle", Y_ac_arr)

# Huffman coding

bitstring = huffman(Y_dc_arr, Y_ac_arr, Cb_dc_arr, Cb_ac_arr, Cr_dc_arr, Cr_ac_arr)
final_file = open("jpeg.txt", "w")
final_file.write(bitstring)
final_file.close()
print("done!")
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