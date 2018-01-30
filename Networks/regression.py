# How to run from the command line:
# python regression.py <train> <test> <predictons> <batch size> <num epochs> <hidden layers> <dropout rate> <learning rate> <lambda>
# For example:
# python regression.py data/all.csv data/d13.csv predict.csv 50 500 24,8 0.5 0.1 0.001

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import math
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf

# Training and test data filenames
TRAIN=sys.argv[1]
TEST=sys.argv[2]

# File to drop the predictions to
PREDICTIONS=sys.argv[3]

# Minibatch size parameter
BATCH_SIZE=int(sys.argv[4])

# Number of passes through the training data
NUM_EPOCHS=int(sys.argv[5])

# Configuration of the hidden layers
HIDDEN_CONFIG=[int(x) for x in sys.argv[6].split(",")]

# Dropout probability
DROPOUT=float(sys.argv[7])

# Learning rate
LEARNING_RATE=float(sys.argv[8])

# Regularlisation strength
LAMBDA=float(sys.argv[9])

# Main function
def main(unused_argv):

  # Turn off warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

  # Turn on logging. Training loss metrics are sent to std err.
  #tf.logging.set_verbosity(tf.logging.INFO)
  
  # Load the datasets into memory
  print("\nReading data...")
  train=np.genfromtxt(TRAIN,delimiter=',',skip_header=1)
  test =np.genfromtxt(TEST,delimiter=',',skip_header=1)
  print("training data:\t",TRAIN)
  print("#training ex.:\t",train.shape[0])
  print("testing data:\t",TEST)
  print("#testing ex.:\t",test.shape[0])
  target_column=train.shape[1]-1  # assume target is last column
  print("target column:\t",target_column)
      
  # Split the data into x and y
  x_train=train[:,0:target_column]
  y_train=train[:,target_column]
  x_test=test[:,0:target_column]
  y_test=test[:,target_column]

  # Blitz original copies of train/test
  train=None
  test=None
  
  # Train a network
  print("\nTraining DNN...")
  print("batch size:\t",BATCH_SIZE)
  print("num. epochs:\t",NUM_EPOCHS)
  print("hid. config:\t",HIDDEN_CONFIG)
  print("drop. rate:\t",DROPOUT)
  print("learn. rate:\t",LEARNING_RATE)
  print("lambda:    \t",LAMBDA)

  # Standardize the training data
  scaler = preprocessing.StandardScaler()
  x_train = scaler.fit_transform(x_train)

  # Build the network
  feature_columns = [
      tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1:])]
  regressor = tf.estimator.DNNRegressor(
      feature_columns=feature_columns,
      hidden_units=HIDDEN_CONFIG,
      dropout=DROPOUT,
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=LEARNING_RATE,
        l1_regularization_strength=LAMBDA)
      )

  # Train the DNN
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train}, y=y_train,
      batch_size=BATCH_SIZE,num_epochs=NUM_EPOCHS,
      shuffle=True)
  regressor.train(input_fn=train_input_fn,
      max_steps=None)

  # Test the network on the test data
  x_transformed = scaler.transform(x_test)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_transformed}, y=y_test, num_epochs=1, shuffle=False)
  predictions = regressor.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['predictions'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)
  
  # Score the test predictions with sklearn
  score_sklearn = metrics.mean_absolute_error(y_predicted, y_test)
  print('\nResults...\nTest MAE:\t{0:f}'.format(score_sklearn))

  # Construct an array containing the (x,y,z) points which are
  # expected to be the first three columns of x_test, followed
  # by columns for the ground truth and the prediction at each point
  print("\nWriting test predictions...")
  print("pred. file:\t",PREDICTIONS)
  predictions=np.concatenate((x_test[:,0:3],y_test[:,None],y_predicted[:,None]),axis=1)
  np.savetxt(PREDICTIONS,predictions,delimiter=",",header="x,y,z,w,w-dashed",comments="")
  
 
# Run the main function
if __name__ == '__main__':
  tf.app.run()


    x = tf.placeholder(tf.float32, [None, 4])
    y = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([4, 1]))
    b = tf.Variable(tf.zeros([1]))

    pred = tf.matmul(x,W) + b

    cost = tf.reduce_mean(-tf.reduce_sum(
        y * tf.log(pred), reduction_indices=1
    ))
    opti = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(0,100):
            avgc = 0
            for j in range(0,1000):
                yele = y_train.reshape(y_train.shape[0], 1)
                xele = x_train#.reshape(1, 4)

                x,c = sess.run([opti, cost],
                    feed_dict={
                        x: xele,
                        y: yele
                    }
                )
            print(x)
            print(c)