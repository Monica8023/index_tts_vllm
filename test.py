import hashlib


def md5_encrypt(input_string):
    """
    使用MD5算法对字符串进行加密

    参数:
    input_string (str): 要加密的字符串

    返回:
    str: 32位小写的MD5加密结果
    """
    # 创建一个md5 hash对象
    md5_hash = hashlib.md5()

    # 更新hash对象，需要将字符串编码为bytes
    md5_hash.update(input_string.encode('utf-8'))

    # 获取16进制的MD5散列值
    encrypted_string = md5_hash.hexdigest()

    return encrypted_string


# 使用示例
if __name__ == "__main__":
    text = input("请输入要加密的字符串: ")
    result = md5_encrypt(text)
    print(f"MD5加密结果: {result}")
