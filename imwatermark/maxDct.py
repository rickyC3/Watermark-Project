import numpy as np
import copy
import cv2
import pywt
import math
import pprint

pp = pprint.PrettyPrinter(indent=2)


class EmbedMaxDct(object):
    def __init__(self, watermarks=[], wmLen=8, scales=[0,36,36], block=4, **configs):
        self._watermarks = watermarks
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

    def encode(self, bgr, set_channel, dwt_domain, **configs):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        impl_img = 0
        if (set_channel == 'rgb'):
            impl_img = rgb
        elif (set_channel == 'yuv'):
            impl_img = yuv

#        for channel in range(2):   # range(3)?
        for channel in range(3):
            if self._scales[channel] <= 0:
                continue
            """
                                        -------------------
                                        |        |        |
                                        | cA(LL) | cH(LH) |
                                        |        |        |
            (cA, (cH, cV, cD))  <--->   -------------------
                                        |        |        |
                                        | cV(HL) | cD(HH) |
                                        |        |        |
                                        -------------------
            """
            ca1,(h1,v1,d1) = pywt.dwt2(impl_img[:row//4*4,:col//4*4,channel], 'haar')
            # //：取商數
            # pywt.dwt2(..., 'haar')：這是呼叫 PyWavelets 工具箱，使用最簡單的 Haar 小波函數，對當前色彩通道的矩陣做二維 DWT。
            if (dwt_domain == "LL"):
                self.encode_frame(ca1, self._scales[channel])
            elif (dwt_domain == "LH"):
                self.encode_frame(h1, self._scales[channel])
            elif (dwt_domain == "HL"):
                self.encode_frame(v1, self._scales[channel])
            elif (dwt_domain == "HH"):
                self.encode_frame(d1, self._scales[channel])
            else:
                raise ValueError(f"Unknown DWT domain: {dwt_domain}")

            tmp = pywt.idwt2((ca1,(h1,v1,d1)), 'haar')
            impl_img[:row//4*4, :col//4*4, channel] = np.clip(np.round(tmp), 0, 255).astype(np.uint8)

        if (set_channel == 'rgb'):
            bgr_encoded = cv2.cvtColor(impl_img, cv2.COLOR_RGB2BGR)
        elif (set_channel == 'yuv'):
            bgr_encoded = cv2.cvtColor(impl_img, cv2.COLOR_YUV2BGR)
        
        return bgr_encoded
        # c：代表 Coefficient（係數）。
        # a：代表 Approximation（近似/低頻）。所以 ca 就是 LL。
        # h：代表 Horizontal detail（水平細節）。所以 h 就是 LH。
        # v：代表 Vertical detail（垂直細節）。所以 v 就是 HL。
        # d：代表 Diagonal detail（對角線細節）。所以 d 就是 HH。
        # 1：代表這是「第一層 (Level 1)」的分解。

    def decode(self, bgr, set_channel, dwt_domain, **configs):
        (row, col, channels) = bgr.shape      

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        impl_img = 0
        if (set_channel == 'rgb'):
            impl_img = rgb
        elif (set_channel == 'yuv'):
            impl_img = yuv

        scores = [[] for i in range(self._wmLen)]
#        for channel in range(2):
        for channel in range(3):
            if self._scales[channel] <= 0:
                continue

            ca1,(h1,v1,d1) = pywt.dwt2(impl_img[:row//4*4,:col//4*4,channel], 'haar')
            # yuv[:row//4*4,:col//4*4,channel] 切成可被4*4整除的大小
            # haar: filter type
            if (dwt_domain == "LL"):
                scores = self.decode_frame(ca1, self._scales[channel], scores)
            elif (dwt_domain == "LH"):
                scores = self.decode_frame(h1, self._scales[channel], scores)
            elif (dwt_domain == "HL"):
                scores = self.decode_frame(v1, self._scales[channel], scores)
            elif (dwt_domain == "HH"):
                scores = self.decode_frame(d1, self._scales[channel], scores)
            else:
                raise ValueError(f"Unknown DWT domain: {dwt_domain}")

        avgScores = list(map(lambda l: np.array(l).mean(), scores))
        bits = (np.array(avgScores) * 255 > 127)
        # # 1. 先轉成 0 和 1 的整數陣列
        # bits_array = (np.array(avgScores) * 255 > 127).astype(int)  #加上這段之後，Numpy 就會自動把所有的 True 變成 1，所有的 False 變成 0。

        # # 2. 把陣列裡的數字全部轉成文字，然後黏合在一起
        # bits_string = "".join(map(str, bits_array))

        return bits
        # return bits_string

    # def decode_frame(self, frame, scale, scores):
    #     (row, col) = frame.shape
    #     num = 0

    #     for i in range(row//self._block):
    #         for j in range(col//self._block):
    #             block = frame[i*self._block : i*self._block + self._block,
    #                           j*self._block : j*self._block + self._block]

    #             score = self.infer_dct_matrix(block, scale)
    #             #score = self.infer_dct_svd(block, scale)
    #             wmBit = num % self._wmLen
    #             scores[wmBit].append(score)
    #             num = num + 1

    #     return scores

    def decode_frame(self, frame, scale, scores):
        '''
        透過比對相鄰 4x4 區塊的中頻點大小來解碼
        '''
        (row, col) = frame.shape
        num = 0
        
        freq_r, freq_c = 2, 2

        for i in range(row // self._block):
            for j in range(0, (col // self._block) - 1, 2):
                block_A = frame[i*self._block : i*self._block + self._block,
                                j*self._block : j*self._block + self._block]
                block_B = frame[i*self._block : i*self._block + self._block,
                                (j+1)*self._block : (j+2)*self._block]

                dct_A = cv2.dct(block_A)
                dct_B = cv2.dct(block_B)

                val_A = dct_A[freq_r, freq_c]
                val_B = dct_B[freq_r, freq_c]

                # 確認這對區塊是對應到密碼陣列的哪一個位置
                wmBit = num % self._wmLen

                if val_A > val_B:
                    scores[wmBit].append(1.0)
                else:
                    scores[wmBit].append(0.0)

                num = num + 1

        return scores

    # def diffuse_dct_svd(self, block, wmBit, scale):
    #     u,s,v = np.linalg.svd(cv2.dct(block))

    #     s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
    #     return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

    # def infer_dct_svd(self, block, scale):
    #     u,s,v = np.linalg.svd(cv2.dct(block))

    #     score = 0
    #     score = int ((s[0] % scale) > scale * 0.5)
    #     return score
    #     if score >= 0.5:
    #         return 1.0
    #     else:
    #         return 0.0

    # def diffuse_dct_matrix(self, block, wmBit, scale):
    #     pos = np.argmax(abs(block.flatten()[1:])) + 1
    #     i, j = pos // self._block, pos % self._block
    #     val = block[i][j]
    #     if val >= 0.0:
    #         block[i][j] = (val//scale + 0.25 + 0.5 * wmBit) * scale
    #     else:
    #         val = abs(val)
    #         block[i][j] = -1.0 * (val//scale + 0.25 + 0.5 * wmBit) * scale
    #     return block

    # def infer_dct_matrix(self, block, scale):
    #     pos = np.argmax(abs(block.flatten()[1:])) + 1
    #     i, j = pos // self._block, pos % self._block

    #     val = block[i][j]
    #     if val < 0:
    #         val = abs(val)

    #     if (val % scale) > 0.5 * scale:
    #         return 1
    #     else:
    #         return 0

    # def encode_frame(self, frame, scale):
    #     '''
    #     frame is a matrix (M, N)

    #     we get K (watermark bits size) blocks (self._block x self._block)

    #     For i-th block, we encode watermark[i] bit into it
    #     '''
    #     (row, col) = frame.shape
    #     num = 0
    #     for i in range(row//self._block):
    #         for j in range(col//self._block):
    #             block = frame[i*self._block : i*self._block + self._block,
    #                           j*self._block : j*self._block + self._block]
    #             wmBit = self._watermarks[(num % self._wmLen)]


    #             diffusedBlock = self.diffuse_dct_matrix(block, wmBit, scale)
    #             #diffusedBlock = self.diffuse_dct_svd(block, wmBit, scale)
    #             frame[i*self._block : i*self._block + self._block,
    #                   j*self._block : j*self._block + self._block] = diffusedBlock

    #             num = num+1

    def encode_frame(self, frame, scale):
            '''
            使用 4x4 相鄰區塊差分調變來嵌入浮水印
            '''
            (row, col) = frame.shape
            num = 0    # 宣告一個計數器 num，從 0 開始。它的功能是「記錄我們現在正在塞第幾個 bit」。
            
            # 4x4 矩陣的中頻點座標
            freq_r, freq_c = 2, 2 

            # 每次取水平相鄰的兩個區塊，所以 j 的迴圈每次跳 2 步
            for i in range(row // self._block):    # row // self._block：垂直方向，高度可以切出幾排 4*4 區塊。
                for j in range(0, (col // self._block) - 1, 2):
                # 水平方向。注意那個 , 2！ 這代表 j 每次前進都是跨兩步（例如：0, 2, 4, 6...）。因為我們一次要抓 A、B 兩個區塊，所以每次要跳過兩個區塊的寬度。
                    block_A = frame[i*self._block : i*self._block + self._block,
                                    j*self._block : j*self._block + self._block]
                    block_B = frame[i*self._block : i*self._block + self._block,
                                    (j+1)*self._block : (j+2)*self._block]
                    # block_A在左邊，block_B在右邊相鄰著block_A
                    # frame[a:b, c:d]：這在 Python 中叫做「切片 (Slicing)」。
                        # Python 的切片規則就是「包含開頭，不包含結尾 (Inclusive start, Exclusive end)」。
                        # 垂直方向抓a ~ (b-1) row，水平方向抓c ~ (d-1) column。

                    # 讀取當下要嵌入的 0 或 1
                    wmBit = self._watermarks[(num % self._wmLen)]
                    # %：取餘數。這個餘數計算會讓密碼不斷循環，把密碼重複鋪滿整張圖片

                    dct_A = cv2.dct(block_A)
                    dct_B = cv2.dct(block_B)

                    val_A = dct_A[freq_r, freq_c]
                    val_B = dct_B[freq_r, freq_c]
                    # 拿出這兩個矩陣在 [2, 2] 中頻位置的數值，存成 val_A 與 val_B。

                    mid = (val_A + val_B) / 2.0
                    
                    # 直接將傳入的 scale 參數當作調變強度 (delta)
                    delta = scale

                    if wmBit == 1:
                        dct_A[freq_r, freq_c] = mid + delta / 2.0
                        dct_B[freq_r, freq_c] = mid - delta / 2.0
                    else:
                        dct_A[freq_r, freq_c] = mid - delta / 2.0
                        dct_B[freq_r, freq_c] = mid + delta / 2.0

                    frame[i*self._block : i*self._block + self._block,
                        j*self._block : j*self._block + self._block] = cv2.idct(dct_A)
                    frame[i*self._block : i*self._block + self._block,
                        (j+1)*self._block : (j+2)*self._block] = cv2.idct(dct_B)

                    num = num + 1