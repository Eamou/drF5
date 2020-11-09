import numpy as np
import cv2
import math

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
    if k_num == 0:
        return 1/math.sqrt(2)
    else:
        return 1

def DCT_2(Y_img):
    for row_block in Y_img:
        for block in row_block:
            for k in range(block_size):
                for l in range(block_size):
                    sigma_sum = 0 
                    for i in range(block_size):
                        for j in range(block_size):
                            Bij = block[i][j]
                            sigma_sum += ((w(k)*w(l))/4)*math.cos((math.pi/16)*k*((2*i)+1))*math.cos((math.pi/16)*l*((2*j)+1))*Bij
                    block[k][l] = sigma_sum
    return Y_img

def DCT_3(Y_img):
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

#print(Y_img[0][0])

# perform DCT transform.....
#Y_img = DCT_2(Y_img)
block = Y_img[0][0]
for k in range(block_size):
    for l in range(block_size):
        sigma_sum = 0 
        for i in range(block_size):
            for j in range(block_size):
                Bij = block[i][j]
                sigma_sum += ((w(k)*w(l))/4)*math.cos((math.pi/16)*k*((2*i)+1))*math.cos((math.pi/16)*l*((2*j)+1))*Bij
        block[k][l] = sigma_sum

dct = cv2.dct(Y_img[0][0])

print(block)
print(dct)



#print(img_tiles[0][0][0])