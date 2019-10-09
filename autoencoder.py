# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:01:34 2019

@author: allen
"""

# -*- coding:utf-8 -*-
#尝试实现能够去除不同类型的噪声的降噪自编码网络
import copy,math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
def get_data():
    (x_train, _), (x_test, _) = datasets.mnist.load_data()
    # 处理成0-1之间的值
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # 重新构造一个N × 1 × 28 × 28 的四维tensor
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train,x_test

def Gaussian_noise(x_train_noise,index_arr):
    noise_factor = 0.5
    
    for i in index_arr:
        x_train_noise[i,:,:,:] = x_train_noise[i,:,:,:] +  noise_factor * np.random.normal(
                loc=0.0, scale=1.0, size=[28,28,1])
    return x_train_noise

def Salt_Pepper_noise(x_train_noise,index_arr):
    #https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1569248827164&di=cf30d1824958967ba07eaf609bcc6776&imgtype=0&src=http%3A%2F%2Fupload.semidata.info%2Fnew.eefocus.com%2Farticle%2Fimage%2F2015%2F10%2F15%2F561ef070117b7.jpg
    threshold = 1
    for i in index_arr:
        mask_mat = np.random.normal(
                loc=0.0, scale=1.0, size=[28,28,1])
        x_train_noise[i,:,:,:] = np.where(mask_mat>threshold,1.,x_train_noise[i,:,:,:])
        x_train_noise[i,:,:,:] = np.where(mask_mat<0-threshold,0.,x_train_noise[i,:,:,:])
    return x_train_noise

def add_noise(x_train,x_test,noise_type=['Gaussian_noise','Pepper_noise']):
    """添加噪音"""
    train_arr = np.arange(60000)
    np.random.shuffle(train_arr)
    test_arr = np.arange(10000)
    np.random.shuffle(test_arr)
    x_train_noisy=copy.deepcopy(x_train)
    x_test_noisy=copy.deepcopy(x_test)
    if 'Pepper_noise' in noise_type :     
        x_train_noisy = Gaussian_noise(x_train_noisy,train_arr[0:20000])
        x_test_noisy = Gaussian_noise(x_test_noisy,test_arr[0:3333])
        x_train_noisy = Salt_Pepper_noise(x_train_noisy,train_arr[20000:40000])
        x_test_noisy = Salt_Pepper_noise(x_test_noisy,test_arr[3333:6666])    
    else:
        x_train_noisy = Gaussian_noise(x_train_noisy,train_arr[0:40000])
        x_test_noisy = Gaussian_noise(x_test_noisy,test_arr[0:6666])
    # 值仍在0-1之间
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy,x_test_noisy
 
def remove_noisy_model(x_train_noisy,x_test_noisy):
    """去燥"""
    input_img = Input(shape=(28, 28, 1,)) # N * 28 * 28 * 1
    # 实现 encoder 部分，由两个 3 × 3 × 8 的卷积和两个 2 × 2 的最大池化组 成。
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img) # 28 * 28 * 16
    x = MaxPooling2D((2, 2), padding='same')(x) # 14 * 14 * 8
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 32
    encoded = MaxPooling2D((2, 2), padding='same')(x) # 7 * 7 * 32
    # 实现 decoder 部分
    # 7 * 7 * 32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(encoded) # 7 * 7 * 32
    x = UpSampling2D((2, 2))(x) # 14 * 14 * 32
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 16
    x = UpSampling2D((2, 2))(x) # 28 * 28 * 16
    decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x) # 28 * 28 *1
 
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
 
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=30,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
 
    autoencoder.save('autoencoder_0.h5')#autoencoder_0，基于两种噪音二维卷积自编码器

def remove_noisy_model_NN(x_train_noisy,x_test_noisy):
    """去燥"""
    input_img = Input(shape=(28, 28, 1,)) # N * 28 * 28 * 1

    # 实现 encoder 部分
    flatten_img = layers.Flatten()(input_img)
    x = layers.Dense(256, activation='relu')(flatten_img)
    encoded = layers.Dense(64, activation='relu')(x)
    # 实现 decoder 部分
    x = layers.Dense(256, activation='relu')(encoded)
    decoded = layers.Dense(784, activation='relu')(x)
    
    decoded = layers.Reshape([28,28,1])(decoded)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
 
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=30,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
 
    autoencoder.save('autoencoder_1.h5')#autoencoder_1一维线性自编码器

def remove_noisy(x_test_noisy,model_name='autoencoder.h5'):
    autoencoder = load_model(model_name)
    decoded_imgs = autoencoder.predict(x_test_noisy)
    return autoencoder,decoded_imgs

def cal_avg_psrn(x_origin,x_decoded):
    mse = np.mean((x_origin - x_decoded) ** 2 )
    if mse < 1.0e-10:
      return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def plot3(x_test,x_test_noisy,decoded_imgs):
    """画图"""
    n = 10
    plt.figure(figsize=(30, 6))
    for i in range(n):
        # display noisy image
        
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
 
        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display origin
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
               
    plt.show()

def visualizing_channel(autoencoder_model,img_index):
    layer_names = []
    for layer in autoencoder_model.layers[1:9]:
        layer_names.append(layer.name)
        
    layer_outputs = [layer.output for layer in autoencoder_model.layers[1:9]]
    activation_model = Model(inputs=autoencoder_model.input, outputs=layer_outputs)
    images_per_row = 16
    img_tensor = np.expand_dims(x_test_noisy[img_index], axis=0)
    activations = activation_model.predict(img_tensor)
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
    
    plt.show()

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(model,layer_name, filter_index, size=28):

    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.
    for i in range(500):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1
        
    img = input_img_data[0]
    return deprocess_image(img)

def visualizing_fliter(autoencoder_model):

    layer_names = []
    for layer in autoencoder_model.layers[1:9]:
        if (layer.name.find('conv') >= 0) :    
            layer_names.append(layer.name)
    for layer_name in layer_names:
        size = 28
        margin = 0
    
        # This a empty (black) image where we will store our results.
        results = np.zeros((2 * size , 8 * size , 1))
    
        for i in range(2):  # iterate over the rows of our results grid
            for j in range(8):  # iterate over the columns of our results grid
                # Generate the pattern for filter `i + (j * 8)` in `layer_name`
                print(i+(j*8))
                filter_img = generate_pattern(autoencoder_model,layer_name, j+i*8, size=size)
    
                # Put the result in the square `(i, j)` of the results grid
                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    
        # Display the results grid
        plt.figure(figsize=(0.05*results.shape[1],
                            0.05*results.shape[0]))
        results=np.reshape(results,((2 * size , 8 * size )))
        plt.title(layer_name)
        plt.imshow(results, cmap='gray')
    plt.show()
#        import cv2
#        cv2.imwrite(layer_name+'.jpg', results)
    
        

x_train,x_test =get_data()
x_train_noisy, x_test_noisy = add_noise(
        x_train,x_test,noise_type=['Gaussian_noise','Pepper_noise'])
#remove_noisy_model(x_train_noisy,x_test_noisy)
autoencoder_model , decoded_imgs = remove_noisy(
        x_test_noisy,model_name='autoencoder_0.h5')
plot3(x_test,x_test_noisy,decoded_imgs)
visualizing_channel(autoencoder_model,2)
#visualizing_fliter(autoencoder_model)
print(cal_avg_psrn(x_test,decoded_imgs))


