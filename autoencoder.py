# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:01:34 2019

@author: allen
"""

# -*- coding:utf-8 -*-
#尝试实现能够去除不同类型的噪声的降噪自编码网络

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model


def get_data():
    (x_train, _), (x_test, _) = datasets.mnist.load_data()
    # 处理成0-1之间的值
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # 重新构造一个N × 1 × 28 × 28 的四维tensor
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train,x_test
 
def add_noise(x_train,x_test):
    """随机添加噪音"""
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    # 值仍在0-1之间
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy,x_test_noisy
 
def remove_noisy_model(x_train_noisy,x_test_noisy):
    """去燥"""
    input_img = Input(shape=(28, 28, 1,)) # N * 28 * 28 * 1
    # 实现 encoder 部分，由两个 3 × 3 × 8 的卷积和两个 2 × 2 的最大池化组 成。
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_img) # 28 * 28 * 8
    x = MaxPooling2D((2, 2), padding='same')(x) # 14 * 14 * 8
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 8
    encoded = MaxPooling2D((2, 2), padding='same')(x) # 7 * 7 * 8
    # 实现 decoder 部分，由两个 3 × 3 × 32 的卷积和两个 2 × 2 的上采样组成。
    # 7 * 7 * 32
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(encoded) # 7 * 7 * 8
    x = UpSampling2D((2, 2))(x) # 14 * 14 * 4
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 8
    x = UpSampling2D((2, 2))(x) # 28 * 28 * 8
    decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x) # 28 * 28 *1
 
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
 
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=30,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
 
    autoencoder.save('autoencoder.h5')
 
def remove_noisy(x_test_noisy):
    autoencoder = load_model('autoencoder.h5')
    decoded_imgs = autoencoder.predict(x_test_noisy)
    return decoded_imgs
 
 
def plot1(x_data):
    """画图"""
    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x_data[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
 
def plot2(x_test_noisy,decoded_imgs):
    """画图"""
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
 
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
 
x_train,x_test =get_data()
x_train_noisy, x_test_noisy = add_noise(x_train,x_test)
#remove_noisy_model(x_train_noisy,x_test_noisy)
decoded_imgs = remove_noisy(x_test_noisy)
plot2(x_test_noisy,decoded_imgs)

