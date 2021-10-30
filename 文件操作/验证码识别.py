import time
import os
import random

import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from captcha.image import ImageCaptcha

FILE_PATH = ".//verification_code"
LETTER_FILE_PATH = ".//letters"

letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
letters_length = len(letters) - 1


# 生成验证码
def generate_random_verification_code(code_len=4):
    verification_code = ""
    # print(index, letters[index])
    for i in range(code_len):
        # 随机生成待选取字母的下标
        index = random.randint(0, letters_length)
        # 将letters[index] 添加到verification_code
        verification_code += letters[index]
    # 保存验证码图片
    file_path = FILE_PATH + "//" + verification_code + '.png'
    return verification_code, file_path


# 生成单字符验证码
def generate_letter_code():
    length = len(letters) - 1
    suffix = ""
    # 随机生成10位乱码后缀
    for i in range(10):
        suffix += letters[random.randint(0, letters_length)]
    letter = letters[random.randint(0, letters_length)]
    file_path = f"{LETTER_FILE_PATH}//{letter}_{suffix}.png"
    return letter, file_path


# 写入文件
def generate_verification_code(verification_code, file_path):
    try:
        image = ImageCaptcha(width=int(40 * len(verification_code)), height=60,
                             fonts=['./JetBrainsMono-Regular.ttf'])
        # cheezy_captcha = WheezyCaptcha(width=75, height=75,
        #                                fonts=['./JetBrainsMono-Regular.ttf'])
        # 检查该图片是否存在
        if not os.path.exists(file_path):
            # with open(file_path, 'wb+') as f:
            #     # 调用ImageCaptcha.generate 生成验证码图片
            #     data = ImageCaptcha.generate(verification_code)
            #     f.write(data)
            #     image.write(data, file_path)
            data = image.generate(verification_code)
            image.write(verification_code, file_path)
            # data = cheezy_captcha.generate(verification_code)
            # cheezy_captcha.write(verification_code, file_path)
        else:
            print('文件已存在')
    except Exception as e:
        print(e)
    finally:
        pass


# 批量生成验证码
def batch(number, method):
    # 生成方法
    generate_method = {
        'random_code': generate_random_verification_code,
        'letter': generate_letter_code
    }
    # 延时
    delay = 0.5
    for i in range(number):
        code, file_path = generate_method[method]()
        generate_verification_code(code, file_path)
        print(code, f"还剩{int(delay * (2 * number - i))}秒, 剩余{number - i}张")
        time.sleep(delay)
    print("已完成")


# 生成独热编码
def convert2onehot(l):
    onehot = np.zeros((1, letters_length + 1))
    index = letters.find(l)
    onehot[0, index] = 1
    return onehot


# 读取文件并转换为数据集
def load_file(dir_path):
    # 获取目录下的所有文件的列表
    file_list = os.listdir(dir_path)
    data, tag = None, None
    for i in range(len(file_list)):
        print(f"loading Picture No.{i + 1}")
        tag_tmp = convert2onehot(file_list[i][0].split('-')[0])
        # print(tag_tmp)
        file_path = dir_path + "//" + file_list[i]
        # 读取图片
        img = mpimg.imread(file_path)
        # 转换为灰度图像，二维数组->一维数组
        convert_img = np.dot(img, [0.299, 0.587, 0.114]).flatten()
        tmp = convert_img
        # print(convert_img, np.shape(convert_img))
        # 数组拼接
        if i <= 0:
            data = tmp
            tag = tag_tmp
        else:
            data = np.vstack((data, tmp))
            tag = np.vstack((tag, tag_tmp))
        print(f"load Picture No.{i + 1} complete")
    data_set = np.concatenate((data, tag), axis=1)
    # print(data_set, data_set.shape)
    return data_set


# 读取数据集二进制文件
def load_data_set(file_path):
    with open(file_path, 'rb') as data:
        return data.read()


# 将数据集写入二进制文件
def save_data_set(data_set, file_path):
    with open(file_path, 'wb+') as data:
        data.write(data_set)


# 神经网络训练
def train_main():
    t1 = time.time()
    print("loading file...")
    data_set = load_file(LETTER_FILE_PATH)
    t2 = time.time()
    print("loading completed")
    print(t2 - t1)


if __name__ == "__main__":
    if os.path.exists(FILE_PATH):
        print(f"{FILE_PATH} 目录已存在")
    else:
        os.mkdir(FILE_PATH)
    if os.path.exists(LETTER_FILE_PATH):
        print(f"{LETTER_FILE_PATH} 目录已存在")
    else:
        os.mkdir(LETTER_FILE_PATH)

    # batch(500, 'letter')
    train_main()
