import argparse
import hashlib
import bchlib
r"""
char. cnt
company: 4
product: 6
auth.:   3
use base64 --> 6 bits
13 * 6 = 78 
BCH (127, 78)
total bits: 128 bits
"""

#BCH_POLYNOMIAL = 137
#BCH_BITS = 8

SETCRET_KEY = "RICKY"

BASE64_TABLE = ""

def init_base64_table():
    global BASE64_TABLE
    for i in range(26):
        BASE64_TABLE += str(chr(ord("A")+i))
    for i in range(26):
        BASE64_TABLE += str(chr(ord("a")+i))
    for i in range(10):
        BASE64_TABLE += str(i)
    BASE64_TABLE += "+/"

    print("BASE64_TABLE initialize success")
    print(BASE64_TABLE)

def generate_auth(wm_msg):
    CONST_LENGTH = 10
    bitstring = []
    if len(wm_msg) > 10:
        raise ValueError("length is longer than 10!")
        return
    while len(wm_msg) != CONST_LENGTH:
        wm_msg += '0'
    
    for c in wm_msg:
        if (c not in BASE64_TABLE):
            raise ValueError("watermark msg shuold be base64!")
            return 
        else:
            idx = BASE64_TABLE.find(c)
            temp_bits = [i for i in range(6)]
            for i in range(5, -1, -1):
                temp_bits[i] = idx % 2
                idx = idx // 2

            bitstring += temp_bits
    print(f"wm msg: {wm_msg}")
    
    shaBits = bitstring + [0, 0, 0, 0] # 6 * 10 = 60
    #print("shaBits", shaBits)

    result = bytearray()

    for i in range(0, len(shaBits), 8):
        byte = 0

        for bit in shaBits[i:i+8]:
            byte = (byte << 1) | bit

        result.append(byte) 

    digest = hashlib.sha256(result).digest()

    # 取前 18 bits
    value = int.from_bytes(digest[:3], "big") >> 6 # 8*3 = 24 >> 6 留下18個bits

    value_bits = [i for i in range(18)]
    for i in range(17, -1, -1):
        value_bits[i] = value % 2
        value //= 2
    #bitstring += value_bits
    print(f"wm bits string length {len(bitstring)}")
    bitstring = bitstring
    print(bitstring)
    return bitstring, value_bits

def generate_bitstring(wm_msg):
    
    init_base64_table()
    
    bitstring = []
    
    for c in wm_msg:
        if (c not in BASE64_TABLE):
            raise ValueError("watermark msg shuold be base64!")
            return 
        else:
            idx = BASE64_TABLE.find(c)
            temp_bits = [i for i in range(6)]
            for i in range(5, -1, -1):
                temp_bits[i] = idx % 2
                idx = idx // 2

            bitstring += temp_bits
    bitstring += [0, 0, 0, 0] # padding to 64
    print(f"wm msg: {wm_msg}")
    print(f"wm msg bits: {bitstring}")
    print(f"length of wm_msg_bits: {len(bitstring)}")
    return bitstring
    
def BCH_ENCODE(wm_bitstring:list, BCH_BITS:int = 8, BCH_POLYNOMIAL:int = 137):

    bch = bchlib.BCH(BCH_BITS, prim_poly=BCH_POLYNOMIAL)
    
    # create bytes array
    packet_bitstring = bytearray()
    for i in range(0, len(wm_bitstring), 8):
        byte = 0
        
        for bit in wm_bitstring[i:i+8]:
            byte = (byte << 1) | bit
            
        packet_bitstring.append(byte)
        
    ecc = bch.encode(packet_bitstring)
    data = packet_bitstring + ecc
    
    data_binary = list("".join(format(x, "08b") for x in data))
    print("Encode finish")
    print("="*7)
    print("after BCH bits string:",list(data_binary))
    print("after BCH bits length:", len(data_binary))
    print("="*7)
    return data_binary

def BCH_DECODE(data_binary: list, BCH_BITS:int = 8, BCH_POLYNOMIAL:int = 137, DEBUG = 0):

    bch = bchlib.BCH(BCH_BITS, prim_poly=BCH_POLYNOMIAL)
    # bitstring list -> bytearray
    data = bytearray()

    for i in range(0, len(data_binary), 8):
        byte = 0

        for bit in data_binary[i:i+8]:
            byte = (byte << 1) | int(bit)

        data.append(byte)
    
    if (DEBUG == 2):
        print("data: ", data)
        print("data length: ", len(data))

    ecc_len = bch.ecc_bytes
    #print("BCH ECC Bytes", ecc_len)

    # 分離 data 和 ECC
    packet_bitstring = data[:-ecc_len]
    ecc = data[-ecc_len:]

    # BCH decode / correct
    bitflips = 0
    try:
    
        bitflips = bch.decode(packet_bitstring, ecc)
        
        if bitflips < 0:
            if (DEBUG == 1):print("BCH decode failed")
            return None
        
        if (DEBUG == 2):print("before correct length packet_bitstring: ", len(packet_bitstring))
        bch.correct(packet_bitstring, ecc)
        if (DEBUG == 2):print("after correct length packet_bitstring: ", len(packet_bitstring))

    except Exception as e:
        print("BCH decode error:", e)
        return None

    # bytearray -> bitstring list
    wm_bitstring = list(
        "".join(format(x, "08b") for x in packet_bitstring)
    )
    if (DEBUG == 1):
        print("wm_bitstring: ", wm_bitstring)
        print("decode wm_bitstring length: ", len(wm_bitstring))
        print("bitflips cnt:", bitflips)

    return wm_bitstring, bitflips

def DECODE_CHAR(bitstring:list):
    BITS_LEN = 64

    if (len(bitstring) != BITS_LEN):
        print(f"bits string should be 64 (10 char with each char 6 bits, and aligned to byte)")

    decode_msg = ""
    extract_bits = bitstring[:-4]
    for i in range(0, len(extract_bits), 6):
        c_bits = list(extract_bits[i: i+6])
        c_bits.reverse()
        idx = 0
        for j in range(6):
            idx += (1<<j)*int(c_bits[j])
        decode_msg += BASE64_TABLE[idx]

    print(f"decode msg is: {decode_msg}")
    return decode_msg


def get_wm_bitstring(wm_msg:str = "0123456789" , BCH_BITS:int = 8, BCH_POLYNOMIAL:int = 137):
    #init_base64_table()
    bits = generate_bitstring(wm_msg)
    secret_list = BCH_ENCODE(bits)
    return secret_list
    #secret_result = BCH_DECODE(secret_list)
    
def main():

    #init_base64_table()

    parser = argparse.ArgumentParser()# 10 char : 4 + 6
    parser.add_argument("--msg", default = "TSMC000001", help = f"the msg you want to embed. The length is 3+7(company + product id)")

    args = parser.parse_args()
    wm_msg = args.msg

    
    secret_list = get_wm_bitstring("NTHUEECS27")
    secret_result, _ = BCH_DECODE(secret_list, DEBUG = 1)
    result_msg = DECODE_CHAR(secret_result)

    """
    bitstring, auth_bits = generate_auth(wm_msg)
    
    bch = bchlib.BCH(BCH_BITS, prim_poly=BCH_POLYNOMIAL)

    pend_bitstring = bitstring
    packet_bitstring = bytearray()
    for i in range(0, len(pend_bitstring), 8):
        byte = 0

        for bit in pend_bitstring[i:i+8]:
            byte = (byte << 1) | bit

        packet_bitstring.append(byte) 
    print("n =", bch.n)
    print("ecc_bytes =", bch.ecc_bytes)
    print(len(pend_bitstring))
    ecc = bch.encode(packet_bitstring)
    data = packet_bitstring + ecc

    data_binary = "".join(format(x, "08b") for x in data)
    
    print(list(data_binary))
    print(len(data_binary))
    """
    


if __name__ == '__main__':
    main()