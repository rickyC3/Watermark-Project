ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = 36
BITS_PER_DIGIT = 6          # 2^6 = 64 > 36，固定用 6 bits 表示一個 Base36 字元


def string_to_base36_bitstring(s: str) -> str:
    """字串 → Base36 → bit string"""
    if not s:
        return ""

    # 1. 字串 → bytes → 大整數
    data = s.encode("utf-8")
    num = int.from_bytes(data, byteorder="big") 

    # 2. 大整數 → Base36 字串
    if num == 0:
        base36 = "0"
    else:
        base36 = ""
        while num:
            num, rem = divmod(num, BASE)
            base36 = ALPHABET[rem] + base36

    # 3. Base36 每個字元 → 固定 6 bits
    bitstring = "".join(
        f"{ALPHABET.index(c):0{BITS_PER_DIGIT}b}" for c in base36
    )
    return bitstring


def base36_bitstring_to_string(bitstring: str) -> str:
    """bit string → Base36 → 原始字串（反函式）"""
    if not bitstring:
        return ""

    # 1. 檢查長度必須是 6 的倍數
    if len(bitstring) % BITS_PER_DIGIT != 0:
        raise ValueError("bit string 長度必須是 6 的倍數")

    # 2. 每 6 bits → Base36 字元
    base36 = ""
    for i in range(0, len(bitstring), BITS_PER_DIGIT):
        chunk = bitstring[i:i + BITS_PER_DIGIT]
        val = int(chunk, 2)
        if val >= BASE:
            raise ValueError(f"無效的 6-bit 值: {val} (必須 < 36)")
        base36 += ALPHABET[val]

    # 3. Base36 → 大整數
    num = 0
    for c in base36:
        num = num * BASE + ALPHABET.index(c)

    # 4. 大整數 → bytes → 字串
    if num == 0:
        return ""

    # 計算需要的 byte 長度
    byte_length = (num.bit_length() + 7) // 8
    data = num.to_bytes(byte_length, byteorder="big")
    return data.decode("utf-8")