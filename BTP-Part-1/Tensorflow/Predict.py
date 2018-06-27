from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import cv2
from numpy import array

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
 # print(logits)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
#  input= tf.Variable(tf.random_normal([1,10,10,1]))
#  filter= tf.Variable(tf.random_normal([3,3,1,1]))
#  op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')
#  op2= tf.nn.conv2d(input, filter,strides=[1,1,1,1], padding='SAME')
#  init = tf.global_variables_initializer()
  #with tf.Session() as sess:
#	sess.run(init)	
#	coord = tf.train.Coordinator()
 #       threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	#print('input lat \n')
	#res=sess.run(input_layer)
	#print(res)
	#print('conv1 \n')
	#res1=sess.run(conv1)
	#print(res1)
	#print('pool1 \n')
	#res2=sess.run(pool1)
	#print(res2)
	#print('conv2 \n')
	#res3=sess.run(conv2)
	#print(res3)
	#print('pool2 \n')
	#res4=sess.run(pool2)
	#print(res4)
	#print('pool2flat \n')
	#res5=sess.run(pool2_flat)
	#print(res5)
	#print('dense \n')
	#res6=sess.run(dense)
	#print(res6)
	
#	coord.request_stop()
#	coord.join(threads)
#	sess.close()

		
	#print('logits\n')
        #res2=sess.run(tf.nn.softmax(logits,name="softmax_tensor"))
	#print(res2)

	
  if mode == tf.estimator.ModeKeys.PREDICT:

  	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	loaded_model=pickle.load(open('mnist_cnn.sav','rb'))
	img = cv2.imread('png/6.png',0)
    

        if img.shape != [28,28]:
        	img2 = cv2.resize(img,(28,28))
    	        img = img2.reshape(28,28,-1);
        else:
    		img = img.reshape(28,28,-1);
  
        img = 1.0 - img/255.0
	    predict_data=np.asarray([img.transpose(2,0,1)])
        predict_data = predict_data.astype('float32')
    
  	predict_input_fn = tf.estimator.inputs.numpy_input_fn( 
  	    x={"x": predict_data},
  	    batch_size=1,
  	    num_epochs=1,
  	    shuffle = False) 
  	
  	predict_results = loaded_model.predict(input_fn=predict_input_fn)
 
        res=list(predict_results)
        print(res[0].get('classes'))
	#print(tf.argmax(eval_results,1))

if __name__ == "__main__":
	tf.app.run()
