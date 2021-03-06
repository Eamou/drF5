import numpy as np
import cv2
import math
import pickle
from sdcs import sdcs

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

#############################################

def onesComp(bitstring):
    oc_bitstring = ''
    for char in bitstring:
        if char == '1':
            oc_bitstring += '0'
        elif char == '0':
            oc_bitstring += '1'
    return oc_bitstring

def huffmanDecode(bitstring):
    cur_bitstream = ''
    cur_bit_i, YCbCr_num = 0, 0
    DC_flag = True
    Y_decoded_img, Y_decoded_block = [], []
    Cb_decoded_img, Cb_decoded_block = [], []
    Cr_decoded_img, Cr_decoded_block = [], []
    bitstring_length = len(bitstring)
    decoded_img = Y_decoded_img # set default value to remove warnings
    decoded_block = Y_decoded_block # ^^
    while cur_bit_i < bitstring_length:
        cur_bitstream += bitstring[cur_bit_i]
        if YCbCr_num == 0:
            decoded_img = Y_decoded_img
            decoded_block = Y_decoded_block
        elif YCbCr_num == 1:
            decoded_img = Cb_decoded_img
            decoded_block = Cb_decoded_block
        elif YCbCr_num == 2:
            decoded_img = Cr_decoded_img
            decoded_block = Cr_decoded_block
        try:
            if DC_flag:
                category = dc_codeword_dict_inv[cur_bitstream]
                DC_flag = False
                cur_bit_i += 1
                if category != 0:
                    dc_magnitude = ''
                    dc_magnitude_len = cur_bit_i + category
                    while cur_bit_i < dc_magnitude_len:
                        dc_magnitude += bitstring[cur_bit_i]
                        cur_bit_i += 1
                    if dc_magnitude[0] == '0':
                        dc_magnitude = onesComp(dc_magnitude)
                        dc_magnitude = -1 * int(dc_magnitude, 2)
                    else:
                        dc_magnitude = int(dc_magnitude, 2)
                    decoded_block.append(dc_magnitude)
                elif category == 0:
                    decoded_block.append(0)
                cur_bitstream = ''
                continue
            else:
                category = ac_codeword_dict_inv[cur_bitstream]
                if category == (0,0):
                    YCbCr_num = (YCbCr_num + 1) % 3
                    if YCbCr_num == 0:
                        Cr_decoded_block = []
                    elif YCbCr_num == 1:
                        Y_decoded_block = []
                    elif YCbCr_num == 2:
                        Cb_decoded_block = []
                    DC_flag = True
                    decoded_img.append(decoded_block)
                    decoded_block = []
                    cur_bit_i += 1
                    cur_bitstream = ''
                    continue
                elif category == (15,0):
                    decoded_block.append([category[0], 0])
                    cur_bitstream = ''
                    cur_bit_i += 1
                    continue
                else:
                    cur_bit_i += 1
                    ac_magnitude = ''
                    ac_magnitude_len = cur_bit_i + category[1]
                    while cur_bit_i < ac_magnitude_len:
                        ac_magnitude += bitstring[cur_bit_i]
                        cur_bit_i += 1
                    if ac_magnitude[0] == '0':
                        ac_magnitude = onesComp(ac_magnitude)
                        ac_magnitude = -1 * int(ac_magnitude, 2)
                    else:
                        ac_magnitude = int(ac_magnitude, 2)
                    decoded_block.append([category[0], ac_magnitude])
                    cur_bitstream = ''
                    continue
        except:
            cur_bit_i += 1
            continue
    return Y_decoded_img, Cb_decoded_img, Cr_decoded_img

def unRLE(decoded_img):
    zz_img = []
    for block in decoded_img:
        zz_block = np.zeros(64)
        zz_block[0] = block[0]
        block_len = len(block)
        ac_val_i = 0
        for ac_arr_i in range(1, block_len):
            ac_val_i += block[ac_arr_i][0] + 1
            zz_block[ac_val_i] = block[ac_arr_i][1]
        zz_img.append(zz_block)
    return np.array(zz_img)

def unDPCM(zz_img):
    cur_dc_val = zz_img[0][0]
    img_len = len(zz_img)
    for dc_count in range(1, img_len):
        cur_dc_val += zz_img[dc_count][0]
        zz_img[dc_count][0] = cur_dc_val
    return zz_img

def unZigZag(zz_img):
    img_tiles, block_row = [], []
    for zz_array in zz_img:
        # this needs to be updated to change 50 to the actual dimensions of the image
        out_block = np.zeros((8,8))
        out_block[0][0] = zz_array[0]
        out_block[0][1] = zz_array[1]
        out_block[1][0] = zz_array[2]
        out_block[2][0] = zz_array[3]
        out_block[1][1] = zz_array[4]
        out_block[0][2] = zz_array[5]
        out_block[0][3] = zz_array[6]
        out_block[1][2] = zz_array[7]
        out_block[2][1] = zz_array[8]
        out_block[3][0] = zz_array[9]
        out_block[4][0] = zz_array[10]
        out_block[3][1] = zz_array[11]
        out_block[2][2] = zz_array[12]
        out_block[1][3] = zz_array[13]
        out_block[0][4] = zz_array[14]
        out_block[0][5] = zz_array[15]
        out_block[1][4] = zz_array[16]
        out_block[2][3] = zz_array[17]
        out_block[3][2] = zz_array[18]
        out_block[4][1] = zz_array[19]
        out_block[5][0] = zz_array[20]
        out_block[6][0] = zz_array[21]
        out_block[5][1] = zz_array[22]
        out_block[4][2] = zz_array[23]
        out_block[3][3] = zz_array[24]
        out_block[2][4] = zz_array[25]
        out_block[1][5] = zz_array[26]
        out_block[0][6] = zz_array[27]
        out_block[0][7] = zz_array[28]
        out_block[1][6] = zz_array[29]
        out_block[2][5] = zz_array[30]
        out_block[3][4] = zz_array[31]
        out_block[4][3] = zz_array[32]
        out_block[5][2] = zz_array[33]
        out_block[6][1] = zz_array[34]
        out_block[7][0] = zz_array[35]
        out_block[7][1] = zz_array[36]
        out_block[6][2] = zz_array[37]
        out_block[5][3] = zz_array[38]
        out_block[4][4] = zz_array[39]
        out_block[3][5] = zz_array[40]
        out_block[2][6] = zz_array[41]
        out_block[1][7] = zz_array[42]
        out_block[2][7] = zz_array[43]
        out_block[3][6] = zz_array[44]
        out_block[4][5] = zz_array[45]
        out_block[5][4] = zz_array[46]
        out_block[6][3] = zz_array[47]
        out_block[7][2] = zz_array[48]
        out_block[7][3] = zz_array[49]
        out_block[6][4] = zz_array[50]
        out_block[5][5] = zz_array[51]
        out_block[4][6] = zz_array[52]
        out_block[3][7] = zz_array[53]
        out_block[4][7] = zz_array[54]
        out_block[5][6] = zz_array[55]
        out_block[6][5] = zz_array[56]
        out_block[7][4] = zz_array[57]
        out_block[7][5] = zz_array[58]
        out_block[6][6] = zz_array[59]
        out_block[5][7] = zz_array[60]
        out_block[6][7] = zz_array[61]
        out_block[7][6] = zz_array[62]
        out_block[7][7] = zz_array[63]
        block_row.append(out_block)
        if len(block_row) == hor_block_count:
            img_tiles.append(np.array(block_row))
            block_row = []
    return np.array(img_tiles)

def deQuantize(Y_img, Y_flag):
    Y_img_len = len(Y_img)
    for row_block_i in range(Y_img_len):
        row_len = len(Y_img[row_block_i])
        for block_i in range(row_len):
            # divide by quantization table
            if Y_flag:
                np.multiply(Y_quant_table, Y_img[row_block_i][block_i], Y_img[row_block_i][block_i])
            else:
                np.multiply(C_quant_table, Y_img[row_block_i][block_i], Y_img[row_block_i][block_i])
    return Y_img

def w(k_num):
    # for use in DCT transformation
    if k_num == 0:
        return 1/math.sqrt(2)
    else:
        return 1

def DCT_3(img_comp):
    # basically the same as DCT2, but returns Y values from DCT coefs!
    dct_img = []
    for row_block in img_comp:
        dct_row = []
        for block in row_block:
            dct_block = cv2.idct(block)
            dct_row.append(dct_block)
        dct_img.append(np.array(dct_row))
    return np.array(dct_img)

def BGR_convert(YCbCr):
    # values from https://wikipedia.org/wiki/YCbCr#JPEG_conversion
    Y, Cb, Cr = YCbCr
    B = Y + 1.772*(Cb-128)
    G = Y - 0.344136*(Cb-128)-0.714136*(Cr-128)
    R = Y + 1.402*(Cr-128)
    return B, G, R

def YCbCr2BGR(Y_img, Cb_img, Cr_img):
    img = []
    # I know this looks bad but it's only O(n^2)!
    for row in range(ver_block_count):
        img_tiles = []
        for column in range(hor_block_count):
            BGR_block = []
            for block in range(BLOCK_SIZE):
                pixel_row = []
                for pixel_i in range(BLOCK_SIZE):
                    Y_val = Y_img[row][column][block][pixel_i]
                    Cb_val = Cb_img[row][column][block][pixel_i]
                    Cr_val = Cr_img[row][column][block][pixel_i]
                    pixel_row.append(np.array(BGR_convert([Y_val, Cb_val, Cr_val])))
                BGR_block.append(np.array(pixel_row))
            img_tiles.append(np.array(BGR_block))
        img.append(np.array(img_tiles))
    return np.array(img)

def assembleImage(img_tiles):
    img, row = [], []
    num_rows = len(img_tiles)
    for row_i in range(num_rows):
        num_cols = len(img_tiles[row_i])
        for pixel in range(BLOCK_SIZE):
            for col_i in range(num_cols):
                block_len = len(img_tiles[row_i][col_i])
                for block in range(block_len):
                    row.append(img_tiles[row_i][col_i][pixel][block])
            img.append(np.array(row))
            row = []
    return np.array(img)

"""
def extractMessage(msg_path, Y_zz_img, Cb_zz_img, Cr_zz_img):
    cur_img = Y_zz_img
    bit_msg = ''
    ac_val_arr = []
    for bit_location in msg_path:
        component, two_d_location, index = bit_location
        if component == 0:
            cur_img = Y_zz_img
        elif component == 1:
            cur_img = Cb_zz_img
        elif component == 2:
            cur_img = Cr_zz_img
        ac_val = int(cur_img[two_d_location][index])
        ac_val_arr.append(ac_val)
        ac_val_lsb = bin(ac_val)[-1]
        bit_msg += ac_val_lsb
    #print(bit_msg)
    #print(ac_val_arr)
    char, message = '', ''
    for bit in bit_msg:
        char += bit
        if len(char) == 8:
            letter = chr(int(char, 2))
            message += letter
            char = ''
    return message
"""

def removeHPadding(img):
    img_list = list(img)
    width = img_width
    while width != v_img_width:
        for row_index in range(len(img)):
            row_list = list(img_list[row_index])
            row_list.pop()
            img_list[row_index] = row_list
        width -= 1
    return np.array(img_list)

def removeVPadding(img):
    img_list = list(img)
    while len(img_list) != v_img_height:
        img_list.pop()
    return np.array(img_list)

def lsbF5(x):
    if x < 0:
        return int((1 - x) % 2)
    else:
        return int(x % 2)

def extractmeF5(msg_path, img):
    n,k,m,a = 3,2,17,[1,2,6]
    f5_sdcs = sdcs((n,k,m), a)
    channel = img[0] #2500 blocks in a single row, no actual rows
    bit_msg = ''
    for bloc in msg_path:
        # row, block, coef
        sdcs_block = list()
        for location in bloc:
            row_i, block_i, coef_i = location
            block = (row_i * hor_block_count) + block_i
            sdcs_block.append(img[0][block][coef_i])
        b = int(f5_sdcs.extract(sdcs_block))
        b_bit = bin(b)[2:]
        while len(b_bit) < math.floor(math.log(m, 2)):
            b_bit = '0' + b_bit
        bit_msg += b_bit
    char, message = '', ''
    for bit in bit_msg:
        char += bit
        if len(char) == 8:
            letter = chr(int(char, 2))
            message += letter
            char = ''
    return message
        


def extractF5(msg_path, img):
    bit_msg = ""
    for bit_loc in msg_path:
        channel = bit_loc[0]
        row_i = bit_loc[1][0]
        block_i = bit_loc[1][1]
        coef_i = bit_loc[1][2]
        block = (row_i * hor_block_count) + block_i
        coef = img[channel][block][coef_i]
        #print(bit_loc, "=", block, coef_i, "coef:", coef)
        bit_msg += str(int(lsbF5(coef)))
    #print(bit_msg)
    char, message = '', ''
    for bit in bit_msg:
        char += bit
        if len(char) == 8:
            letter = chr(int(char, 2))
            message += letter
            char = ''
    return message

########################################
########PROGRAM BEGINS HERE#############
########################################

# constants
###############
BLOCK_SIZE = 8

###############

with open('jpeg.txt', 'r') as f:
    bitstring = f.read()

with open ('.msgpath', 'rb') as fp:
    msg_path = pickle.load(fp)

with open ('.imgdim', 'rb') as fp:
    img_height, img_width = pickle.load(fp)

with open ('.v_imgdim', 'rb') as fp:
    v_img_height, v_img_width = pickle.load(fp)

hor_block_count = img_width // BLOCK_SIZE
ver_block_count = img_height // BLOCK_SIZE

# extract data from Huffman encoding
Y_decoded_img, Cb_decoded_img, Cr_decoded_img = huffmanDecode(bitstring)
print("finished decode")

# restore Huffman data to 64-len zigzag arrays
Y_zz_img = unRLE(Y_decoded_img)
Cb_zz_img = unRLE(Cb_decoded_img)
Cr_zz_img = unRLE(Cr_decoded_img)
print("extracted zigzags")

# extract message
message = extractmeF5(msg_path, [Y_zz_img, Cb_zz_img, Cr_zz_img])
print("extracted message:", message)

# zig zags are stored differently - 2d array here, but a 3d array when encoding positions.
# e.g, rather than {[[],[],[],[]],[[],[],[],[]]}, it is {[],[],[],[],...,[],[],[],[]}

# restore DC values from DPCM
Y_zz_img = unDPCM(Y_zz_img)
Cb_zz_img = unDPCM(Cb_zz_img)
Cr_zz_img = unDPCM(Cr_zz_img)
print("extracted DC values from DPCM")

# transform 64-len zigzag array to 8x8 tile
Y_img_tiles = unZigZag(Y_zz_img)
Cb_img_tiles = unZigZag(Cb_zz_img)
Cr_img_tiles = unZigZag(Cr_zz_img)
print("restored 8x8 tiles")

# de-quantize
Y_dct_img = deQuantize(Y_img_tiles, True)
Cb_dct_img = deQuantize(Cb_img_tiles, False)
Cr_dct_img = deQuantize(Cr_img_tiles, False)
print("reversed quantization")

# inverse DCT and shift +128
print("beginning dct...")
Y_img = DCT_3(Y_dct_img)
Cb_img = DCT_3(Cb_dct_img)
Cr_img = DCT_3(Cr_dct_img)
print("performed inverse DCT")

# transform YCbCr to BGR
img_tiles = YCbCr2BGR(Y_img, Cb_img, Cr_img)
print("converted YCbCr to BGR")

# collate tiles into 2d image array
#print(img_height, v_img_height)
#print(img_width, v_img_width)
img = assembleImage(img_tiles)
if img_height != v_img_height:
    img = removeVPadding(img)
if img_width != v_img_width:
    img = removeHPadding(img)
cv2.imwrite('images/color_img.png', img)
print("done!")