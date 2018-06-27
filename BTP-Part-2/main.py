from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import prepare_data as pd
import numpy
import math
import h5py
import cv2
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import tensorflow as tf
from numpy import array
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from PIL import Image
import os


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train():
    srcnn_model = model()
    print(srcnn_model.summary())
    data, label = pd.read_training_data("./crop_train.h5")
    val_data, val_label = pd.read_training_data("./test.h5")

    checkpoint = ModelCheckpoint("SRCNN_check_reg2.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, nb_epoch=50, verbose=0)
    # srcnn_model.load_weights("m_model_adam.h5")


def predict():
    srcnn_model = predict_model()
    srcnn_model.load_weights("SRCNN_check_reg2.h5")
    IMG_NAME = "/home/kunal/Desktop/SRCNN-keras-master/Test/Set5/baby_GT.bmp"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "pre.jpg"

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)    
    #Y_img = cv2.resize(src=img[:, :, 0], dsize=(shape[1]/2 , shape[0]/2 ),interpolation=cv2.INTER_CUBIC)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    shape = img.shape
    Y_img = cv2.resize(src=img[:, :, 0], dsize=(shape[1]/2 , shape[0]/2 ),interpolation=cv2.INTER_CUBIC)
    #ima=cv2.cvtColor(Y_img,cv2.COLOR_YCR_CB2BGR)
    #cv2.imwrite("Output/LR.jpg", Y_img)
    Y_img = cv2.resize(Y_img, dsize=(shape[1], shape[0]),interpolation= cv2.INTER_CUBIC)
    #img=cv2.resize(img,dsize=(shape[1],shape[0]), interpolation=cv2.INTER_CUBIC)
    #img2=np.zeros((shape[1]*2,shape[0]*2,3))2
    #print(Y_img.shape)	
    #print(img.shape)
    #img2=np.zeros((1024,1024,3))
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    #Y_img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite(INPUT_NAME, img)
    #cv2.imwrite(INPUT_NAME, Y_img)

    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[6: -6, 6: -6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCR_CB)[6: -6, 6: -6, 0]


    print "bicubic:"
    print psnr(im1, im2)
    print "SRCNN:"
    print psnr(im1, im3)


if __name__ == "__main__":
    	#train()
    predict()
