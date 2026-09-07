import struct
import uuid
import copy
import base64
import cv2
import numpy as np
from .maxDct import EmbedMaxDct
from .dwtDctSvd import EmbedDwtDctSvd
#from .rivaGan import RivaWatermark
import pprint

pp = pprint.PrettyPrinter(indent=2)

class WatermarkEncoder(object):
# class：代表「我要開始寫一張設計圖了」。這張設計圖的名字叫 WatermarkEncoder。
# object：在括號裡面寫 object，意思是「我這張設計圖，是繼承自 Python 最基礎的『萬物之源 (object)』」。
    def __init__(self, content=b''):
    # __init__.py（檔案）：是用來把「資料夾」初始化成「套件」。
    # def __init__（函數）：這是寫在 class 設計圖裡面的一個「出生函數（建構子）」。
    # =b'' 是一個預設值（空的位元組）。意思是：如果你聘請主廚時沒給他食材，他就預設自己拿到的是空氣。
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)
        self._wmType = 'bytes'
    # 為什麼是 self._watermarks 而不是 self.watermarks？
        # 這個底線 _ 在 Python 裡是一個約定俗成的「警告標誌」，意思是：這是私有物品 (Private)，請外面的人不要隨便亂碰！
    # _watermarks、_wmLen等等它們不是輸出，它們是主廚的「內部記憶體 / 狀態」。

    def set_by_ipv4(self, addr):
        bits = []
        ips = addr.split('.')
        for ip in ips:
            bits += list(np.unpackbits(np.array([ip % 255], dtype=np.uint8)))
        self._watermarks = bits
        self._wmLen = len(self._watermarks)
        self._wmType = 'ipv4'
        assert self._wmLen == 32

    def set_by_uuid(self, uid):
        u = uuid.UUID(uid)
        self._wmType = 'uuid'
        seq = np.array([n for n in u.bytes], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_bytes(self, content):
    # def (Define) 就是定義一個函式 / 功能。寫在 class 裡面的函式，我們通常稱呼它為「方法 (Method)」。這等於是規定這位主廚「會做什麼技能」。
        self._wmType = 'bytes'
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)
    # content：這裡接收到的是已經被轉成電腦底層編碼的資料（例如前面提到的 b'nthu'）。
    # dtype=np.uint8：資料型態「無號 8 位元整數 (Unsigned 8-bit Integer)」。這行會把每一個字元轉成 0 到 255 之間的數字。ASCII:A->65

    def set_by_b16(self, b16):
        content = base64.b16decode(b16)
        self.set_by_bytes(content)
        self._wmType = 'b16'

    def set_by_bits(self, bits=[]):
        self._watermarks = [int(bit) % 2 for bit in bits]
        self._wmLen = len(self._watermarks)
        self._wmType = 'bits'

    def set_watermark(self, wmType='bytes', content=''):
        if wmType == 'ipv4':
            self.set_by_ipv4(content)
        elif wmType == 'uuid':
            self.set_by_uuid(content)
        elif wmType == 'bits':
            self.set_by_bits(content)
        elif wmType == 'bytes':
            self.set_by_bytes(content)
        elif wmType == 'b16':
            self.set_by_b16(content)
        else:
            raise NameError('%s is not supported' % wmType)
    # 外面使用者要用的時候:
        # encoder = WatermarkEncoder()
        # 加文字: encoder.set_watermark('bytes', b'abc') 或 encoder.set_watermark('bytes', 'abc'.encode('utf-8'))
        # 加0、1: encoder.set_watermark('bits', [0, 1, 1, 1, 1, 0, 0, 0, 1]) 或 encoder.set_watermark('bits', '011110001')

    def get_length(self):
        return self._wmLen

    # @classmethod
    # def loadModel(cls):
    #     RivaWatermark.loadModel()

    def encode(self, cv2Image, method='dwtDct', **configs):
        (r, c, channels) = cv2Image.shape
        #cv2Image 是已經在外面經過 cv2.imread 的格式
        if r*c < 256*256:
            raise RuntimeError('image too small, should be larger than 256x256')

        if method == 'dwtDct':
            embed = EmbedMaxDct(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image, **configs)
        # embed.encode(cv2Image) 會啟動 maxDct.py 裡面的那個複雜運算，最後吐出一張加密好的新圖片矩陣。
        # return：管家把這張新圖片矩陣，端出去還給外面的使用者。
        elif method == 'dwtDctSvd':
            embed = EmbedDwtDctSvd(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        # elif method == 'rivaGan':
        #     embed = RivaWatermark(self._watermarks, self._wmLen)
        #     return embed.encode(cv2Image)
        else:
            raise NameError('%s is not supported' % method)
    # 當你在外面寫腳本時，可以這樣呼叫：
        # encoder.encode(bgr, method='dwtDct', scales=[36, 0, 0], block=8)
        # 對於 encode 函式來說，它只認得 cv2Image 和 method。剩下的 scales=[36, 0, 0] 和 block=8，它通通會打包進一個叫做 configs 的字典裡。

class WatermarkDecoder(object):
    def __init__(self, wm_type='bytes', length=0):
        self._wmType = wm_type
        if wm_type == 'ipv4':
            self._wmLen = 32
        elif wm_type == 'uuid':
            self._wmLen = 128
        elif wm_type == 'bytes':
            self._wmLen = length
        elif wm_type == 'bits':
            self._wmLen = length
        elif wm_type == 'b16':
            self._wmLen = length
        else:
            raise NameError('%s is unsupported' % wm_type)

    def reconstruct_ipv4(self, bits):
        ips = [str(ip) for ip in list(np.packbits(bits))]
        return '.'.join(ips)

    def reconstruct_uuid(self, bits):
        nums = np.packbits(bits)
        bstr = b''
        for i in range(16):
            bstr += struct.pack('>B', nums[i])

        return str(uuid.UUID(bytes=bstr))

    def reconstruct_bits(self, bits):
        #return ''.join([str(b) for b in bits])
        return bits

    def reconstruct_b16(self, bits):
        bstr = self.reconstruct_bytes(bits)
        return base64.b16encode(bstr)

    def reconstruct_bytes(self, bits):
        nums = np.packbits(bits)
        bstr = b''
        for i in range(self._wmLen//8):
            bstr += struct.pack('>B', nums[i])
        return bstr
    # np.packbits：它會自動把陣列裡面的 0 和 1，每 8 個綁成一捆，然後還原成原本的數學整數。
    # 執行完這行，nums 就會變成像 [110, 116, 104, 117] 這樣的整數陣列。
    # bstr = b''：準備一個空字串來裝最後的結果。前面的 b 代表這不是一般的文字字串，而是「位元組字串 (Byte String)」。
    # 在電腦的眼裡，b'nthu' 跟數字陣列 [110, 116, 104, 117] 是一模一樣的東西。
    # 第 1 圈 (i = 0)：
        # 電腦拿到 nums[0]，也就是數字 110。
        # struct.pack('>B', 110) 會把整數 110 嚴格壓縮成電腦底層的 1 個 byte。Python 會把它顯示為 b'n'。
        # bstr += b'n'
        # 目前的 bstr：b'n'
    # 第 2 圈 (i = 1)：
        # 電腦拿到 nums[1]，也就是數字 116。
        # struct.pack('>B', 116) 把它變成 b't'。
        # 把 b't' 黏到原本的箱子後面。
        # 目前的 bstr：b'nt'

    def reconstruct(self, bits):
        if len(bits) != self._wmLen:
            raise RuntimeError('bits are not matched with watermark length')

        if self._wmType == 'ipv4':
            return self.reconstruct_ipv4(bits)
        elif self._wmType == 'uuid':
            return self.reconstruct_uuid(bits)
        elif self._wmType == 'bits':
            return self.reconstruct_bits(bits)
        elif self._wmType == 'b16':
            return self.reconstruct_b16(bits)
        else:
            return self.reconstruct_bytes(bits)

    def decode(self, cv2Image, method='dwtDct', **configs):
        (r, c, channels) = cv2Image.shape
        if r*c < 256*256:
            raise RuntimeError('image too small, should be larger than 256x256')

        bits = []
        if method == 'dwtDct':
            embed = EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image, **configs)
        elif method == 'dwtDctSvd':
            embed = EmbedDwtDctSvd(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        # elif method == 'rivaGan':
        #     embed = RivaWatermark(watermarks=[], wmLen=self._wmLen, **configs)
        #     bits = embed.decode(cv2Image)
        else:
            raise NameError('%s is not supported' % method)
        return self.reconstruct(bits)

    # @classmethod
    # def loadModel(cls):
    #     RivaWatermark.loadModel()
