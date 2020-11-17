import numpy as np
import cv2
import math


# quantization tables - these will be changed later
# possibly stored in a file?
# option to change encoding quality added later
quant_table = np.array([[
    16, 11, 10, 16, 24, 40, 51, 61],
[12, 12, 14, 19, 26, 58, 60, 55],
[14, 13, 16, 24, 40, 57, 69, 56],
[14, 17, 22, 29, 51, 87, 80, 62],
[18, 22, 37, 56, 68, 109, 103, 77],
[24, 36, 55, 64, 81, 104, 113, 92],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 99
]])

quant_table_2 = np.array([[
5, 3, 4, 4, 4, 3, 5, 4],
[4, 4, 5, 5, 5, 6, 7, 12],
[8, 7, 7, 7, 7, 15, 11, 11],
[9, 12, 13, 15, 18, 18, 17, 15],
[20, 20, 20, 20, 20, 20, 20, 20],
[20, 20, 20, 20, 20, 20, 20, 20],
[20, 20, 20, 20, 20, 20, 20, 20],
[20, 20, 20, 20, 20, 20, 20, 20
]])

def readImage(image_name):
    # return image object img
    img = cv2.imread('./images/'+image_name,cv2.IMREAD_COLOR)
    return img

def getImageDimensions(img):
    # return image height, width as integers
    img_height = img.shape[0]
    img_width = img.shape[1]
    return img_height, img_width

def displayImage(img):
    # show image in new window until key pressed
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blockify(img):
    # transform image into series of 8x8 blocks
    img_tiles = []
    for row in range(0, img_height, block_size):
        # fill in row by row
        img_tiles_row = []
        for column in range(0, img_width, block_size):
            column_end = column + block_size
            row_end = row + block_size
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
    B = bgr[0]
    G = bgr[1]
    R = bgr[2]
    Y = (0.299*R)+(0.587*G)+(0.114*B)
    Cb = 128 - (0.168736*R) - (0.331264*G) + (0.5 * B)
    Cr = 128 + (0.5 * R) - (0.418688 * G) - (0.081312 * B)
    return [Y, Cb, Cr]

def BGR2YCbCr(img_tiles):
    # perform conversion for each pixel
    Y_img = []
    # I know this looks bad but it's only O(n^2)!
    for row in range(ver_block_count):
        Y_tiles = []
        for column in range(hor_block_count):
            Y_block = np.zeros((8,8))
            for block in range(block_size):
                for pixel_i in range(block_size):
                    pixel = np.array([YCbCr_convert(img_tiles[row][column][block][pixel_i])])
                    img_tiles[row][column][block][pixel_i] = pixel
                    Y_block[block][pixel_i] = pixel[0][0]-128
            Y_tiles.append(Y_block)
        Y_tiles = np.array(Y_tiles)
        Y_img.append(Y_tiles)
    Y_img = np.array(Y_img)      

    return img_tiles, Y_img

def w(k_num):
    # for use in DCT transformation
    if k_num == 0:
        return 1/math.sqrt(2)
    else:
        return 1

def DCT_2(Y_img):
    # transform Y values into DCT coefficients
    # i think this is O(N^4) as it goes through the whole block
    # for each value in the block. definitely the slowest part of the process either way
    dct_img = []
    for row_block in Y_img:
        dct_img_row = []
        for block in row_block:
            out_block = np.zeros((8,8))
            for k in range(block_size):
                for l in range(block_size):
                    sigma_sum = 0 
                    for i in range(block_size):
                        for j in range(block_size):
                            Bij = block[i][j]
                            sigma_sum += ((w(k)*w(l))/4)*math.cos((math.pi/16)*k*((2*i)+1))*math.cos((math.pi/16)*l*((2*j)+1))*Bij
                    out_block[k][l] = sigma_sum
            dct_img_row.append(out_block)
        dct_img_row = np.array(dct_img_row)
        dct_img.append(dct_img_row)
    dct_img = np.array(dct_img)
    return dct_img

def DCT_3(Y_img):
    # basically the same as DCT2, but returns Y values from DCT coefs!
    dct_Y = []
    for row_block in Y_img:
        dct_row = []
        for block in row_block:
            dct_block = np.zeros((8,8))
            for i in range(block_size):
                for j in range(block_size):
                    sigma_sum = 0 
                    for k in range(block_size):
                        for l in range(block_size):
                            dkl = block[k][l]
                            sigma_sum += ((w(k)*w(l))/4)*math.cos((math.pi/16)*k*((2*i)+1))*math.cos((math.pi/16)*l*((2*j)+1))*dkl
                    dct_block[i][j] = sigma_sum
            dct_row.append(dct_block)
        dct_row = np.array(dct_row)
        dct_Y.append(dct_row)
    dct_Y = np.array(dct_Y)
    return dct_Y

def quantizeAndRound(Y_img):
    # quantizes DCT coefs in-place using quant_table_2 atm (add quality options later)
    # then rounds to nearest integer
    Y_img_len = len(Y_img)
    for row_block_i in range(Y_img_len):
        row_len = len(Y_img[row_block_i])
        for block_i in range(row_len):
            # divide by quantization table
            np.divide(Y_img[row_block_i][block_i], quant_table_2, Y_img[row_block_i][block_i])
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
        zz_row = np.array(zz_row)
        zz_image.append(zz_row)
    zz_image = np.array(zz_image)
    return zz_image

def RLEandDPCM(zz_img):
    # create array of all DC values, encoded using DPCM - each value is the difference
    # from the previous value rather than the actual value
    # create array of RLE-encoded AC values - [skip, value]
    # where skip is the number of zeroes preceeding value.
    # [0,0] indicates the end of the block and is appended to the end
    dc_array = []
    ac_arrays = []
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
                dc_array.append(zz_img[row_block_i][block_i][0] - zz_img[row_block_i][block_i-1][0])
            
            # start at 1 to skip the DC value
            ac_i = 1
            zero_count = 0
            while ac_i < len(zz_img[row_block_i][block_i]):
                cur_num = zz_img[row_block_i][block_i][ac_i]
                if cur_num == 0:
                    if zero_count == 15:
                        ac_rle.append([zero_count, cur_num])
                        zero_count = 0
                        ac_i += 1
                        continue
                    zero_count += 1
                    ac_i += 1
                    continue
                else:
                    ac_rle.append([zero_count, cur_num])
                    zero_count = 0
                    ac_i += 1
                    continue
            # append end of block marker
            ac_rle.append([0,0])
            ac_arrays.append(ac_rle)
    dc_array = np.array(dc_array)
    return dc_array, ac_arrays

########################################
########PROGRAM BEGINS HERE#############
########################################

# read image (ability to input image name to be added later)
# get image dimensions
image_name = 'fagen.png'
img = readImage(image_name)
img_height, img_width = getImageDimensions(img)

# split image into 8x8 blocks and store in img_tiles
# note that this is a downsampling ratio of 4:4:4 (no downsampling), others added later?
block_size = 8
hor_block_count = img_width // block_size
ver_block_count = img_height // block_size
img_tiles = blockify(img)

# convert from list to numpy array
img_tiles = np.array(img_tiles)

# values to floats for DCT
img_tiles = [np.float32(tile) for tile in img_tiles]

# convert BGR to YCbCr
# the image is in img_tiles with YCbCr pixels
# Y_img contains the 8x8 blocks of just the Y values for use in DCT
img_tiles, Y_img = BGR2YCbCr(img_tiles)

# perform DCT transform.....

Y_img_dct = DCT_2(Y_img)

# divide the dct-transformed 8x8 block by the selected quantization table
# and round to nearest integer

Y_img_quant = quantizeAndRound(Y_img_dct)

# zig zag encoding

zz_img = zigZagEncode(Y_img_quant)

# encode DC coefficients ([0][0]) using DPCM
# encode AC coefficients using RLE

dc_arr, ac_arr = RLEandDPCM(zz_img)

print(dc_arr, ac_arr[0])

# Huffman coding







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