import numpy as np
import copy
import cv2
import pywt
import math
import pprint


# embedding setting
embd_freq = (2, 2)
block = 4
set_channel_scale = [10, 10, 10]

def encode(img, watermark_bits, channel_scale, dwt_domain, embd_freq):
    (row, col, channels) = img.shape
    # for channel in range(2):   # range(3)?

    DWT_BLOCK = 2 * block
    for channel in range(3):
        if channel_scale[channel] <= 0:
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
        ca1,(h1,v1,d1) = pywt.dwt2(img[:(row// DWT_BLOCK) * DWT_BLOCK,:(col // DWT_BLOCK) * DWT_BLOCK, channel], 'haar')
        # //：取商數
        # pywt.dwt2(..., 'haar')：這是呼叫 PyWavelets 工具箱，使用最簡單的 Haar 小波函數，對當前色彩通道的矩陣做二維 DWT。
        if (dwt_domain == "LL"):
            ca1 = encode_frame(ca1, watermark_bits, channel_scale[channel], embd_freq)
        elif (dwt_domain == "LH"):
            h1 = encode_frame(h1, watermark_bits, channel_scale[channel], embd_freq)
        elif (dwt_domain == "HL"):
            v1 = encode_frame(v1, watermark_bits, channel_scale[channel], embd_freq)
        elif (dwt_domain == "HH"):
            d1 = encode_frame(d1, watermark_bits, channel_scale[channel], embd_freq)
        else:
            raise ValueError(f"Unknown DWT domain: {dwt_domain}")

        tmp = pywt.idwt2((ca1,(h1,v1,d1)), 'haar')
        img[:(row// DWT_BLOCK) * DWT_BLOCK,:(col // DWT_BLOCK) * DWT_BLOCK,channel] = np.clip(np.round(tmp), 0, 255).astype(np.uint8)
    
    return img
    # c：代表 Coefficient（係數）。
    # a：代表 Approximation（近似/低頻）。所以 ca 就是 LL。
    # h：代表 Horizontal detail（水平細節）。所以 h 就是 LH。
    # v：代表 Vertical detail（垂直細節）。所以 v 就是 HL。
    # d：代表 Diagonal detail（對角線細節）。所以 d 就是 HH。
    # 1：代表這是「第一層 (Level 1)」的分解。

def decode(img, wmLen, channel_scale, dwt_domain, embd_freq):
    (row, col, channels) = img.shape      

    all_scores = [[] for i in range(wmLen)]
    
    #        for channel in range(2):
    DWT_BLOCK = block * 2
    for channel in range(3):
        if channel_scale[channel] <= 0:
            continue
        
        local_scores = [[] for i in range(wmLen)]
        ca1,(h1,v1,d1) = pywt.dwt2(img[:(row// DWT_BLOCK) * DWT_BLOCK,:(col // DWT_BLOCK) * DWT_BLOCK, channel], 'haar')
        # yuv[:row//4*4,:col//4*4,channel] 切成可被4*4整除的大小
        # haar: filter type
        if (dwt_domain == "LL"):
            scores = decode_frame(ca1, wmLen, embd_freq)
        elif (dwt_domain == "LH"):
            scores = decode_frame(h1, wmLen, embd_freq)
        elif (dwt_domain == "HL"):
            scores = decode_frame(v1, wmLen, embd_freq)
        elif (dwt_domain == "HH"):
            scores = decode_frame(d1, wmLen, embd_freq)
        else:
            raise ValueError(f"Unknown DWT domain: {dwt_domain}")

        if (len(scores) != wmLen):
            raise ValueError(f"length is not matched!!!")

        for i in range(wmLen):
            for j in range(len(scores[i])):
                all_scores[i].append(scores[i][j])
                local_scores[i].append(scores[i][j])

        tmpAns = []
        for i in range(wmLen):
            if (local_scores[i].count(1)* 2  >= len(local_scores[i])):
               tmpAns.append(1)
            else:
                tmpAns.append(0)
        print(f"Channel {channel} decode msg is {tmpAns}")

    decodeBits = []
    for i in range(wmLen):
        if (all_scores[i].count(1) * 2 > len(all_scores[i])):
            decodeBits.append(1)
        else:
            decodeBits.append(0)
    print(f"Decode msg is {decodeBits}")
    return decodeBits
    # return bits_string



def decode_frame(frame, wmLen, embd_freq):
    '''
    透過比對相鄰 4x4 區塊的中頻點大小來解碼
    '''
    scores = [[] for i in range(wmLen)]

    (row, col) = frame.shape
    ROW = row // block
    COL = col // block
    TOTAL_BLOCK = ROW * COL
    num = 0
    
    freq_r, freq_c = embd_freq

    for idx in range(0, TOTAL_BLOCK, 2):
        if idx == TOTAL_BLOCK-1:
            break
        r1 = idx // COL
        c1 = idx % COL

        r2 = (idx+1) // COL
        c2 = (idx+1) % COL

        block_A = frame[r1*block : r1*block + block,
                        c1*block : c1*block + block]
        block_B = frame[r2*block : r2*block + block,
                        c2*block : c2*block + block]
        
        dct_A = cv2.dct(block_A)
        dct_B = cv2.dct(block_B)

        val_A = dct_A[freq_r, freq_c]
        val_B = dct_B[freq_r, freq_c]
        # 拿出這兩個矩陣在 [2, 2] 中頻位置的數值，存成 val_A 與 val_B。

        wmIdx = num % wmLen
        
        if val_A > val_B:
            scores[wmIdx].append(1)
        else:
            scores[wmIdx].append(0)

        num = num + 1

    return scores

def encode_frame(frame, watermarks, scale, embd_freq):
    '''
    使用 4x4 相鄰區塊差分調變來嵌入浮水印
    '''
    (row, col) = frame.shape
    num = 0    # 宣告一個計數器 num，從 0 開始。它的功能是「記錄我們現在正在塞第幾個 bit」。
    
    # 4x4 矩陣的中頻點座標
    freq_r, freq_c = embd_freq

    # 每次取水平相鄰的兩個區塊，所以 j 的迴圈每次跳 2 步
    ROW = row // block
    COL = col // block

    TOTAL_BLOCK = ROW * COL
    wmLen = len(watermarks)

    for idx in range(0, TOTAL_BLOCK, 2):
        if idx == TOTAL_BLOCK-1:
            break
        r1 = idx // COL
        c1 = idx % COL

        r2 = (idx+1) // COL
        c2 = (idx+1) % COL

        block_A = frame[r1*block : r1*block + block,
                        c1*block : c1*block + block]
        block_B = frame[r2*block : r2*block + block,
                        c2*block : c2*block + block]

        wmBit = watermarks[(num % wmLen)]

        dct_A = cv2.dct(block_A)
        dct_B = cv2.dct(block_B)

        val_A = dct_A[freq_r, freq_c]
        val_B = dct_B[freq_r, freq_c]
        # 拿出這兩個矩陣在 [2, 2] 中頻位置的數值，存成 val_A 與 val_B。

        mid = (val_A + val_B) / 2.0

        delta = scale
        
        if wmBit == 1:
            dct_A[freq_r, freq_c] = mid + delta / 2.0
            dct_B[freq_r, freq_c] = mid - delta / 2.0
        else:
            dct_A[freq_r, freq_c] = mid - delta / 2.0
            dct_B[freq_r, freq_c] = mid + delta / 2.0

        frame[r1*block : r1*block + block,
            c1*block : c1*block + block] = cv2.idct(dct_A)
        frame[r2*block : r2*block + block,
            c2*block : c2*block + block] = cv2.idct(dct_B)

        num = num + 1

    return frame


def DWTDCT(input_path, output_path, wmMsg = "Ricky", image_format = "YUV"):

    img = cv2.imread(input_path)
    if (image_format == "RGB"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif (image_format == "YUV"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else:
        raise ValueError(f"Unknown image format!")

    wmMsgLen = len(wmMsg)
    wmBits = [int(b) for c in wmMsg for b in format(ord(c), '08b')]
    wmBitsLen = len(wmBits)
    print("wmBits: ", wmBits)
    embd_img = encode(img, wmBits, set_channel_scale, "LL", (0, 0))

    if (image_format == "RGB"):
        embd_img = cv2.cvtColor(embd_img, cv2.COLOR_RGB2BGR)
    elif (image_format == "YUV"):
        embd_img = cv2.cvtColor(embd_img, cv2.COLOR_YUV2BGR)
    else:
        raise ValueError(f"Unknown image format!")
    cv2.imwrite(output_path, embd_img)


    
    

def DWTDCT_Decode(input_path, wmMsg = "Ricky", image_format = "YUV"):
    wmMsgLen = len(wmMsg)
    wmBits = [int(b) for c in wmMsg for b in format(ord(c), '08b')]
    wmBitsLen = len(wmBits)


    decode_img = cv2.imread(input_path)
    if (image_format == "RGB"):
        decode_img = cv2.cvtColor(decode_img, cv2.COLOR_BGR2RGB)
    elif (image_format == "YUV"):
        decode_img = cv2.cvtColor(decode_img, cv2.COLOR_BGR2YUV)
    else:
        raise ValueError(f"Unknown image format!")
    decode_bits = decode(decode_img, wmBitsLen, set_channel_scale, "HH", (3, 3))

    err = 0

    for i in range(wmBitsLen):
        if (wmBits[i] != decode_bits[i]):
            err += 1

    print(f"Bit ERROR Rate: {err / wmBitsLen:.4f}")


def main():
    DWTDCT(r"D:\Ricky\program\invisible-watermark\invisible-watermark\images\image10.jpg", 
           r"D:\Ricky\program\invisible-watermark\invisible-watermark\workspace\DWTDCT_test_img.png", 
           wmMsg = "Ricky", image_format = "YUV")

    # DWTDCT_Decode(r"D:\Ricky\NTHU\Project_DM\print-cam\image\jpeg_test.jpg",
    #                wmMsg = "Ricky", image_format = "YUV")

if (__name__ == '__main__'):
    main()