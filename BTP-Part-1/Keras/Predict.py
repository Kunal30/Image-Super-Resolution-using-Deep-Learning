from keras.models import load_model
from keras.utils import plot_model
import pydot
import cv2
from numpy import array
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_rows, img_cols = 28, 28


new_model= load_model('mnist_cnn1.h5')
new_model.summary()
img = cv2.imread('jpg/7.jpg',0)
if img.shape != [28,28]:
        	img2 = cv2.resize(img,(28,28))
    	        img = img2.reshape(28,28,-1);
else:
    		img = img.reshape(28,28,-1);
  
img = 1.0 - img/255.0
predict_data=np.asarray([img.transpose(2,0,1)])
predict_data = predict_data.astype('float32')
#plt.imshow(predict_data, cmap='gray')	
#plt.show()

predict_data = predict_data.astype('float32')
predict_data/= 255

if K.image_data_format() == 'channels_first':
    img = predict_data.reshape(predict_data.shape[0], 1, img_rows, img_cols)    
    input_shape = (1, img_rows, img_cols)
else:
    img = predict_data.reshape(predict_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

predicts=new_model.predict_classes(img, batch_size= None, verbose= 0)
print(predicts)
#print(new_model.get_weights())
#plot_model(new_model, to_file='model.png')
#print(new_model.optimizer)
