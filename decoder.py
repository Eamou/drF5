#from encoder import encoder
import numpy as np
import cv2
import math
import pickle
import os.path
from Crypto.Cipher import AES
import simplejpeg

from sdcs import sdcs
from stc import stc
from rs import rs

#############################################

class decoder:
    def __init__(self, block_size, rs_param):
        self.BLOCK_SIZE = block_size
        self.RS_PARAM = rs_param
        self.img_height, self.img_width = None, None
        self.hor_block_count, self.ver_block_count = None, None
        self.Y_quant_table, self.C_quant_table = self.__getQuantTables()
        self.dc_codeword_dict, self.dc_codeword_dict_inv = self.__getDCCodewordDicts()
        self.ac_codeword_dict, self.ac_codeword_dict_inv = self.__getACCodewordDicts()

    def getImageDimensions(self, img):
        # return image height, width as integers
        return img.shape[0], img.shape[1]

    def defineBlockCount(self, v, h):
        # helper function for when calling from main.py
        self.hor_block_count = h
        self.ver_block_count = v

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

    def onesComp(self, bitstring):
        oc_bitstring = ''
        for char in bitstring:
            if char == '1':
                oc_bitstring += '0'
            elif char == '0':
                oc_bitstring += '1'
        return oc_bitstring

    def huffmanDecode(self, bitstring):
        cur_bitstream = ''
        cur_bit_i, YCbCr_num = 0, 0
        DC_flag = True
        Y_decoded_img, Y_decoded_block = list(), list()
        Cb_decoded_img, Cb_decoded_block = list(), list()
        Cr_decoded_img, Cr_decoded_block = list(), list()
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
                    category = self.dc_codeword_dict_inv[cur_bitstream]
                    DC_flag = False
                    cur_bit_i += 1
                    if category != 0:
                        dc_magnitude = ''
                        dc_magnitude_len = cur_bit_i + category
                        while cur_bit_i < dc_magnitude_len:
                            dc_magnitude += bitstring[cur_bit_i]
                            cur_bit_i += 1
                        if dc_magnitude[0] == '0':
                            dc_magnitude = self.onesComp(dc_magnitude)
                            dc_magnitude = -1 * int(dc_magnitude, 2)
                        else:
                            dc_magnitude = int(dc_magnitude, 2)
                        decoded_block.append(dc_magnitude)
                    elif category == 0:
                        decoded_block.append(0)
                    cur_bitstream = ''
                    continue
                else:
                    category = self.ac_codeword_dict_inv[cur_bitstream]
                    if category == (0,0):
                        YCbCr_num = (YCbCr_num + 1) % 3
                        if YCbCr_num == 0:
                            Cr_decoded_block = list()
                        elif YCbCr_num == 1:
                            Y_decoded_block = list()
                        elif YCbCr_num == 2:
                            Cb_decoded_block = list()
                        DC_flag = True
                        decoded_img.append(decoded_block)
                        decoded_block = list()
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
                            ac_magnitude = self.onesComp(ac_magnitude)
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

    def unRLE(self, img):
        final_img = list()
        for channel in img:
            zz_img = list()
            for block in channel:
                zz_block = np.zeros(64)
                zz_block[0] = block[0]
                block_len = len(block)
                ac_val_i = 0
                for ac_arr_i in range(1, block_len):
                    ac_val_i += block[ac_arr_i][0] + 1
                    zz_block[ac_val_i] = block[ac_arr_i][1]
                zz_img.append(zz_block)
            final_img.append(np.array(zz_img))
        return np.array(final_img)

    def unDPCM(self, zz_img):
        final_img = list()
        for channel in zz_img:
            cur_dc_val = channel[0][0]
            img_len = len(channel)
            for dc_count in range(1, img_len):
                cur_dc_val += channel[dc_count][0]
                channel[dc_count][0] = cur_dc_val
            final_img.append(channel)
        return final_img

    def unZigZag(self, zz_img):
        id_block = np.reshape(np.arange(self.BLOCK_SIZE*self.BLOCK_SIZE), (self.BLOCK_SIZE, self.BLOCK_SIZE))
        indices = np.hstack([np.diagonal(id_block[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-id_block.shape[0], id_block.shape[0])])
        final_img = list()
        for channel in zz_img:
            img=list()
            for zz_block in channel:
                block = np.zeros(self.BLOCK_SIZE*self.BLOCK_SIZE)
                for i, ind in enumerate(indices):
                    block[ind] = zz_block[i]
                img.append(np.reshape(block, (self.BLOCK_SIZE,self.BLOCK_SIZE)))
            final_img.append(np.reshape(img, (self.hor_block_count,self.ver_block_count,self.BLOCK_SIZE,self.BLOCK_SIZE)))
        return np.array(final_img)

    def deQuantize(self, img):
        final_img = list()
        for i, channel in enumerate(img):
            table = self.Y_quant_table if i == 0 else self.C_quant_table
            final_img.append(np.array([np.multiply(block, table) for block in np.array([row for row in channel])]))
        return np.array(final_img)

    def w(self, k_num):
        # for use in DCT transformation
        if k_num == 0:
            return 1/math.sqrt(2)
        else:
            return 1

    def DCT_3(self, img):
        # basically the same as DCT2, but returns Y values from DCT coefs!
        return np.array([np.array([np.array([cv2.idct(block) for block in row]) for row in channel]) for channel in img])

    def BGR_convert(self, YCbCr):
        # values from https://wikipedia.org/wiki/YCbCr#JPEG_conversion
        Y, Cb, Cr = YCbCr
        B = Y + 1.772*(Cb-128)
        G = Y - 0.344136*(Cb-128)-0.714136*(Cr-128)
        R = Y + 1.402*(Cr-128)
        return B, G, R

    def YCbCr2BGR(self, orig_img):
        img = list()
        # I know this looks bad but it's only O(n^2)!
        for row in range(self.ver_block_count):
            img_tiles = list()
            for column in range(self.hor_block_count):
                BGR_block = list()
                for block in range(self.BLOCK_SIZE):
                    pixel_row = list()
                    for pixel_i in range(self.BLOCK_SIZE):
                        Y_val = orig_img[0][row][column][block][pixel_i]
                        Cb_val = orig_img[1][row][column][block][pixel_i]
                        Cr_val = orig_img[2][row][column][block][pixel_i]
                        pixel_row.append(np.array(self.BGR_convert([Y_val, Cb_val, Cr_val])))
                    BGR_block.append(np.array(pixel_row))
                img_tiles.append(np.array(BGR_block))
            img.append(np.array(img_tiles))
        return np.array(img)

    def assembleImage(self, img_tiles):
        img, row = list(), list()
        num_rows = len(img_tiles)
        for row_i in range(num_rows):
            num_cols = len(img_tiles[row_i])
            for pixel in range(self.BLOCK_SIZE):
                for col_i in range(num_cols):
                    block_len = len(img_tiles[row_i][col_i])
                    for block in range(block_len):
                        row.append(img_tiles[row_i][col_i][pixel][block])
                img.append(np.array(row))
                row = list()
        return np.array(img)

    def removeHPadding(self, img, v_img_width):
        img_list = list(img)
        width = self.img_width
        while width != v_img_width:
            for row_index in range(len(img)):
                row_list = list(img_list[row_index])
                row_list.pop()
                img_list[row_index] = row_list
            width -= 1
        return np.array(img_list)

    def removeVPadding(self, img, v_img_height):
        img_list = list(img)
        while len(img_list) != v_img_height:
            img_list.pop()
        return np.array(img_list)

    def lsbF5(self, x):
        if x < 0:
            return int((1 - x) % 2)
        else:
            return int(x % 2)

    def oldFixMancErrors(self, y, diff_manc):
        sign = lambda x: math.copysign(1, x)
        for coef_i, coef in enumerate(y):
            if coef == 0:
                if coef_i == 0:
                    y[0] = (-1)**int(1-diff_manc[0])
                else:
                    if diff_manc[coef_i]: # if numbers are different
                        y[coef_i] = -1 * sign(y[coef_i-1])
                    else:
                        y[coef_i] = sign(y[coef_i])
        return y
    
    def diffMancEnc(self, a):
        map_sign = lambda x: 1 if math.copysign(1, x) == 1 else 0
        x = np.zeros(len(a), dtype=np.uint8)
        x[0] = map_sign(a[0])
        i = 1
        while i < len(a):
            x[i] = map_sign(a[i-1]) ^ map_sign(a[i])
            i += 1
        return x

    def fixMancErrors(self, y, diff_manc):
        diff_manc = ''.join([str(x) for x in diff_manc])
        r = ''.join([str(x) for x in self.diffMancEnc(y)])
        diff = format(int(r,2) ^ int(diff_manc,2), '0'+str(len(diff_manc))+'b')
        i = 0
        while i < len(diff)-1:
            if diff[i] + diff[i+1] == '11':
                y[i] *= -1 #what if it's zero?
            i += 1
        return y

    def extractOptimaldmcss(self, msg_path, parity, img):
        rs_obj = rs(256)
        H_hat = np.array([71,109], dtype=np.uint8)
        stc_obj = stc(H_hat)
        bit_msg = ''
        map_sign = lambda x: 1 if math.copysign(1, x) == 1 else 0
        y = list()
        diff_manc = list()
        for loc in msg_path:
            channel_i, row_i, block_i, block_path = loc
            block = (row_i * self.hor_block_count) + block_i%self.hor_block_count
            for coef_i, manc_i in block_path:
                diff_manc.append(manc_i)
                try:
                    y.append(img[channel_i][block][coef_i])
                except:
                    # if we do random, we'll get a number the dmc can correct?
                    y.append(0)
        # fix w/ reedsolomon
        #   split into groups of 239, append relevant parity bits
        #   correct, then pass forward without parity#
        y_polys = [y[j:j+rs_obj.K] for j in range(0, len(y), rs_obj.K)]
        corrected_y = list()
        for k, poly in enumerate(y_polys): # only need to correct 0s
            full_poly = poly + parity[k]
            zero_mask = [coef==0 for coef in full_poly]
            err_ind = np.where(np.array(zero_mask) == True)[0]
            if len(err_ind) != 0 and len(err_ind) <= (2*rs_obj.T):
                full_poly = rs_obj.detectErasures(full_poly, err_ind)
            corrected_y += full_poly[:-(rs_obj.N-rs_obj.K)] # wrap coefs instead of signs?
        # now have lossy dct coefs + differentia manchester
        y = self.fixMancErrors(corrected_y, diff_manc)
        y = np.array([map_sign(x) for x in y])
        H = stc_obj.gen_H(y, len(y)//2)
        m = np.array((H @ y) % 2, dtype=np.uint8)
        bit_msg += ''.join([str(bit) for bit in m])
        return bit_msg

    def extractdmcss(self, msg_path, img):
        H_hat = np.array([71,109], dtype=np.uint8)
        stc_obj = stc(H_hat)
        bit_msg = ''
        map_sign = lambda x: 1 if math.copysign(1, x) == 1 else 0
        for loc in msg_path:
            channel_i, row_i, block_i, block_path = loc
            block = (row_i * self.hor_block_count) + block_i%self.hor_block_count
            y = list()
            diff_manc = list()
            #print(img[channel_i][block], block)
            #exit(0)
            for coef_i, manc_i in block_path:
                diff_manc.append(manc_i)
                try:
                    y.append(img[channel_i][block][coef_i])
                except:
                    y.append(0)
            # now have lossy dct coefs + differentia manchester
            y = self.fixMancErrors(y, diff_manc)
            y = np.array([map_sign(x) for x in y])
            H = stc_obj.gen_H(y, len(y)//2)
            m = np.array((H @ y) % 2, dtype=np.uint8)
            bit_msg += ''.join([str(bit) for bit in m])
        return bit_msg

    def extractdrF5(self, msg_path, img):
        H_hat = np.array([71,109], dtype=np.uint8)
        stc_obj = stc(H_hat)
        bit_msg = ''
        for loc in msg_path:
            channel_i, row_i, block_i, block_path = loc
            block = (row_i * self.hor_block_count) + block_i
            y = list()
            for coef_i in block_path:
                try:
                    y.append(img[channel_i][block][coef_i] % 2)
                except:
                    y.append(0)
            H = stc_obj.gen_H(y, len(y)//2)
            m = np.array((H @ y) % 2, dtype=np.uint8)
            bit_msg += ''.join([str(bit) for bit in m])
        return bit_msg

    def extractsdcsF5(self, msg_path, img):
        n,k,m,a = 3,2,17,[1,2,6]
        f5_sdcs = sdcs((n,k,m), a)
        bit_msg = ''
        for bloc in msg_path:
            # channel, global block, coefs
            channel_i, row_i, block_i, coefs = bloc
            block = (row_i * self.hor_block_count) + block_i%self.hor_block_count
            sdcs_block = list()
            for coef in coefs:
                try:
                    sdcs_block.append(img[channel_i][block][coef])
                except:
                    sdcs_block.append(0)
            b = int(f5_sdcs.extract(sdcs_block))
            b_bit = bin(b)[2:]
            while len(b_bit) < math.floor(math.log(m, 2)):
                b_bit = '0' + b_bit
            bit_msg += b_bit
        return bit_msg

    def extractF5(self, msg_path, img, LSB):
        bit_msg = ""
        for bit_loc in msg_path:
            channel = bit_loc[0]
            row_i = bit_loc[1]
            block_i = bit_loc[2]
            coef_i = bit_loc[3]
            block = (row_i * self.hor_block_count) + block_i
            try:
                coef = img[channel][block][coef_i]
            except:
                coef = np.random.choice([0,1])
            if not LSB:
                bit_msg += str(int(self.lsbF5(coef)))
            else:
                bit_msg += str(int(coef%2))
        return bit_msg 

    def extractRSPoly(self, bit_msg):
        char, message = '', list()
        for bit in bit_msg:
            char += bit
            if len(char) == 8:
                letter = int(char, 2)
                message.append(letter)
                char = ''
        return np.array(message)

    def extractMsgTxt(self, bit_msg):
        char, message = '', ''
        for bit in bit_msg:
            char += bit
            if len(char) == 8:
                letter = chr(int(char, 2))
                message += letter
                char = ''
        return message

    def retrievePath(self, key):
        if os.path.isfile("path_key.bin"):
            file_in = open("path_key.bin", "rb")
            nonce, tag, ciphertext = [ file_in.read(x) for x in (16,16,-1) ]

            cipher = AES.new(key, AES.MODE_EAX, nonce)
            data = cipher.decrypt_and_verify(ciphertext, tag)
            path = data.decode()
        else:
            raise FileNotFoundError("Can't find path file - ensure it is named 'path_key.bin'")
        return np.array(list(path))

    def formatPathF5(self, path):
        new_path = list()
        split_path = np.split(path, len(path)//2)
        split_path = [''.join([str(x) for x in a]) for a in split_path]
        partition = list()
        i = 0
        rsize = math.ceil(len(str(self.ver_block_count)) / 2)
        bsize = math.ceil(len(str(self.hor_block_count)) / 2)
        while i < len(split_path):
            partition += [int(split_path[i])-1]
            i += 1

            partition += [int(str(''.join([str(x) for x in split_path[i:i+(rsize)]])))-1]
            i += rsize
            
            partition += [int(str(''.join([str(x) for x in split_path[i:i+(bsize)]])))-1]
            i += bsize

            while split_path[i] != '00':
                partition += [int(split_path[i])-1]
                i += 1
            i += 1
            new_path.append(partition)
            partition = list()
        return new_path

    def formatPath(self, path, mode):
        rs_obj = rs(256)  
        new_path = list()
        split_path = np.split(path, len(path)//2)
        split_path = [''.join([str(x) for x in a]) for a in split_path]
        partition = list()
        i = 0
        bsize = len(str(self.ver_block_count * self.hor_block_count))
        if bsize % 2 != 0: bsize += 1
        while i < len(split_path):
            partition += [int(split_path[i])]
            i += 1

            block = int(str(''.join([str(x) for x in split_path[i:i+(bsize//2)]])))
            row_i = block // self.hor_block_count
            block_i = block % self.hor_block_count
            partition += [int(row_i), int(block_i)]
            i += bsize//2

            block_path = list()
            while split_path[i] != '00':
                if mode == 0:
                    block_path.append([int(split_path[i]), int(split_path[i+1])-1]) #minus one for diff code
                    i += 2
                elif mode == 1:
                    block_path.append(int(split_path[i]))
                    i += 1
            i += 1
            partition.append(block_path)
            new_path.append(partition)
            partition = list()
            if split_path[i] == 'PB':
                parity_nums = split_path[i+1:]
                parity_nums = [int(parity_nums[j] + parity_nums[j+1]) for j in range(0, len(parity_nums), 2)]
                parity_polys = [parity_nums[j:j+2*rs_obj.T] for j in range(0, len(parity_nums), 2*rs_obj.T)]
                return new_path, parity_polys
        return new_path
         
    def decode(self, img, key, func=2, verbose=True, use_rs=True, output_file="stego", greyscale=False):
        if verbose:
            with open(img, 'r') as f:
                bitstring = f.read()

            with open ('.imgdim', 'rb') as fp:
                self.img_height, self.img_width = pickle.load(fp)

            with open ('.v_imgdim', 'rb') as fp:
                v_img_height, v_img_width = pickle.load(fp)
            
            self.hor_block_count = self.img_width // self.BLOCK_SIZE
            self.ver_block_count = self.img_height // self.BLOCK_SIZE

            hash_path = self.retrievePath(key)
            # extract data from Huffman encoding
            Y_decoded_img, Cb_decoded_img, Cr_decoded_img = self.huffmanDecode(bitstring)
            print("finished decode")

            img = self.unRLE([Y_decoded_img, Cb_decoded_img, Cr_decoded_img])
            print("extracted zigzags")

            if func == 0:
                msg_path = self.formatPathF5(hash_path)
                message = self.extractF5(msg_path, img, False)
            elif func == 1:
                msg_path = self.formatPath(hash_path, mode=1)
                message = self.extractsdcsF5(msg_path, img)
            elif func == 2:
                msg_path = self.formatPath(hash_path, mode=0)
                message = self.extractdmcss(msg_path, img)
            elif func == 3:
                msg_path = self.formatPathF5(hash_path)
                message = self.extractF5(msg_path, img, True)
            if use_rs:
                rs_obj = rs(self.RS_PARAM)
                polys = self.extractRSPoly(message)
                polys = [polys[i:i+rs_obj.N] for i in range(0, len(polys), rs_obj.N)]
                message = ''
                for poly in polys:
                    corrected_message = rs_obj.detectErrors(poly)
                    message += ''.join([chr(x) for x in corrected_message[:len(corrected_message)-(rs_obj.T*2)]])
                print("extracted message:", message)

            else:
                message = self.extractMsgTxt(message)
                print("non-rs extracted message:", message)

            img = self.unDPCM(img)
            print("extracted DC values from DPCM")

            img = self.unZigZag(img)
            print("restored 8x8 tiles")

            img = self.deQuantize(img)
            print("reversed quantization")

            print("beginning dct...")
            img = self.DCT_3(img)
            print("performed inverse DCT")

            img = self.YCbCr2BGR(img)
            print("converted YCbCr to BGR")

            img = self.assembleImage(img)

            if self.img_height != v_img_height:
                img = self.removeVPadding(img, v_img_height)
            if self.img_width != v_img_width:
                img = self.removeHPadding(img, v_img_width)
            cv2.imwrite(output_file+'.png', img)
            print("done!")
        
        else:
            from encoder import encoder
            encoder_obj = encoder(self.BLOCK_SIZE, self.RS_PARAM)
            encoder_obj.defineBlockCount(self.ver_block_count, self.hor_block_count)
            if not greyscale:
                with open(img+".jpg", "rb") as f:
                    jpg_img = simplejpeg.decode_jpeg(f.read(), 'BGR', False, False)
                    jpg_img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2YCR_CB)
            else:
                jpg_img = cv2.imread(img+".jpg", cv2.IMREAD_GRAYSCALE)
            self.img_height, self.img_width = self.getImageDimensions(jpg_img)
            encoder_obj.defineImgDim(self.img_height, self.img_width)
            if self.img_width % self.BLOCK_SIZE != 0:
                jpg_img = encoder_obj.padImageWidth(jpg_img)
            if self.img_height % self.BLOCK_SIZE != 0:
                jpg_img = encoder_obj.padImageHeight(jpg_img)
            new_img_height, new_img_width = self.getImageDimensions(jpg_img)
            self.hor_block_count, self.ver_block_count = new_img_width // self.BLOCK_SIZE, new_img_height // self.BLOCK_SIZE
            total_blocks = self.ver_block_count * self.hor_block_count

            if not greyscale:
                Y_img, Cr_img, Cb_img = cv2.split(jpg_img)
                img = encoder_obj.blockify([Y_img, Cb_img, Cr_img])
            else:
                img = encoder_obj.blockify([jpg_img])
            #print("Separated successfully")

            #print("beginning dct...")
            img = encoder_obj.DCT_2(img)
            #print("finished dct")

            img = encoder_obj.quantizeAndRound(img)
            #print("finished quantization and round")

            img = encoder_obj.zigZagEncode(img)
            #print("finished zigzag")

            img = [np.reshape(channel, (total_blocks, self.BLOCK_SIZE * self.BLOCK_SIZE)) for channel in img]

            hash_path = self.retrievePath(key)

            if func == 0:
                msg_path = self.formatPathF5(hash_path)
                message = self.extractF5(msg_path, img, False)
            elif func == 1:
                msg_path = self.formatPath(hash_path, mode=1)
                message = self.extractsdcsF5(msg_path, img)
            elif func == 2:
                msg_path, parity = self.formatPath(hash_path, mode=0)
                message = self.extractOptimaldmcss(msg_path, parity, img)
            elif func == 3:
                msg_path = self.formatPathF5(hash_path)
                message = self.extractF5(msg_path, img, True)
            if use_rs:
                rs_obj = rs(self.RS_PARAM)
                polys = self.extractRSPoly(message)
                polys = [polys[i:i+rs_obj.N] for i in range(0, len(polys), rs_obj.N)]
                message = ''
                for poly in polys:
                    corrected_message = rs_obj.detectErrors(poly)
                    message += ''.join([chr(x) for x in corrected_message[:len(corrected_message)-(rs_obj.T*2)]])
            else:
                message = self.extractMsgTxt(message)
                print("non-rs extracted message:", message)
            #return message
            with open(output_file+".txt", 'w', encoding="utf-8") as f:
                f.write(message)
            print("message extracted successfully")
            exit(0)

########################################
########PROGRAM BEGINS HERE#############
########################################

#decoder_obj = decoder(8,256)
#print(decoder_obj.fixMancErrors([-1,-1,1,-1,1,-1,-1], [0,1,0,1,1,1,0]))

#decoder_obj = decoder(8, 256)
#decoder_obj.decode('stego_simplejpeg.jpg', b'Sixteen byte key', func=2, verbose=False)

"""
floor instead of rint? is it rounding up instead of down, hence getting an int rather than a 0?
[[10. -2.  0.  0.  0.  0.  0.  0.]
 [-3. -1.  1.  0.  0.  0.  0.  0.]
 [ 0. -2. -1.  0.  0.  0.  0.  0.]
 [ 1. -1.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]]
 from entirely verbose pipeline. some signs are wrong so its probably not cv2
"""
#with open ('.msgpath', 'rb') as fp:
#    msg_path = pickle.load(fp)



# restore Huffman data to 64-len zigzag arrays


# extract message


# zig zags are stored differently - 2d array here, but a 3d array when encoding positions.
# e.g, rather than {[[],[],[],[]],[[],[],[],[]]}, it is {[],[],[],[],...,[],[],[],[]}

# restore DC values from DPCM


# transform 64-len zigzag array to 8x8 tile

# de-quantize


# inverse DCT


# transform YCbCr to BGR


# collate tiles into 2d image array
#print(img_height, v_img_height)
#print(img_width, v_img_width)

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