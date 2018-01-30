from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
#from six.moves.urllib.request import urlopen

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

# Data sets
TRAINING = sys.argv[1]
VAL = sys.argv[2]
#TEST10 = sys.argv[3]
#TEST20 = sys.argv[4]
#TEST30 = sys.argv[5]
#TEST40 = sys.argv[6]
#TEST50 = sys.argv[7]
#TEST60 = sys.argv[8]
#TEST66 = sys.argv[9]

TRAIN_SIZE=8192
BATCH_SIZE=4096

def createXandY(load, target_column):
    train=np.genfromtxt(load,delimiter=',',skip_header=1)
    y_train_one = train[:,target_column]
    x_train = np.delete(train, np.s_[3:4], axis=1)
    N,M = x_train.shape
    #allone = np.ones((N, M + 1))
    #allone[:, 1:] = x_train
    #x_train = allone
    #y_train_ones = np.ones(N)
    #y_train = np.concatenate((y_train,y_train_ones.T))
    y_train = y_train_one#.T
    y_train = np.reshape(y_train, (N, 1))
    return (x_train, y_train)

def main():
    """ Main """
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(TRAINING):
        sys.exit()

    #res = pd.read_csv(TRAINING, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])
    #val = pd.read_csv(VAL, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])
    #print(res.head(10))


    tf.logging.set_verbosity(tf.logging.ERROR)

    # Load the datasets into memory
    print("\nReading data...")
    train=np.genfromtxt(TRAINING,delimiter=',',skip_header=1)
    test =np.genfromtxt(VAL,delimiter=',',skip_header=1)
    print("#training ex.:\t",train.shape[0])
    print("#testing ex.:\t",test.shape[0])
    print(train.shape[1])
    target_column=train.shape[1]-1  # assume target is last column
    print("target column:\t",target_column)

    #x_train=train
    y_train_one = train[:,target_column]
    x_train=train[:,0:target_column]
    #x_train = np.delete(train, np.s_[3:4], axis=1)
    N,M = x_train.shape
    #allone = np.ones((N, M + 1))
    #allone[:, 1:] = x_train
    #x_train = allone
    #y_train_ones = np.ones(N)
    #y_train = np.concatenate((y_train,y_train_ones.T))
    y_train = y_train_one#.T
    y_train = np.reshape(y_train, (N, 1))
    print(y_train.shape)



    y_test = test[:,target_column]
    N,M = test.shape
    y_test = np.reshape(y_test, (N, 1))

    x_test=test[:,0:target_column]
    #x_test = np.delete(test, np.s_[3:4], axis=1)
    #N,M = x_test.shape
    #allone = np.ones((N, M + 1))
    #allone[:, 1:] = x_test
    #x_test = allone
    

    # Blitz original copies of train/test
    train=None
    test=None

    # Specify that all features have real-value data
    FEATURES = ['X','Y','Z','VI']
    #feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
    feature_cols=[tf.feature_column.numeric_column('x', shape=[4])]
    export_input_fn = tf.contrib.learn.utils.build_parsing_serving_input_fn(feature_cols)

    print(feature_cols)
    print(FEATURES)

    inputNodes = 5 # X,Y,Z,VI,Theta
    outputNodes = 1 # Vp
    hiddenLayers = [500,500,500,500] # Using 4 of 4 layers

    # TF symbols
    X = tf.placeholder("float", shape=[None, inputNodes])
    Y = tf.placeholder("float", shape=[None, outputNodes])

    # weight inits
    def init_weights(name, shape):
        """Init Weights"""
        weights = tf.random_normal(shape, stddev=0.1)
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(
            name=name, 
            shape=shape,
            initializer=initializer
            )

    w_1 = init_weights( "w1", (inputNodes, hiddenLayers[0]) ) ## L1
    w_2 = init_weights( "w2", (hiddenLayers[0], hiddenLayers[1]) ) ## L2
    w_3 = init_weights( "w3", (hiddenLayers[1], hiddenLayers[2]) ) ## L3
    w_4 = init_weights( "w4", (hiddenLayers[2], hiddenLayers[3]))
    w_out = init_weights( "wout", (hiddenLayers[3], outputNodes) )

    bias_e = tf.Variable(tf.zeros([]))
    #Bias
    bias_1 = tf.Variable(tf.zeros([hiddenLayers[0]]))
    bias_2 = tf.Variable(tf.zeros([hiddenLayers[1]]))
    bias_3 = tf.Variable(tf.zeros([hiddenLayers[2]]))
    bias_4 = tf.Variable(tf.zeros([hiddenLayers[3]]))
    bias_out = tf.Variable(tf.zeros([outputNodes]))

    # Prop

    def forwardprop(x, y, z):
        h = tf.nn.relu(tf.matmul(x, y))
        return tf.matmul(h, z)

    X_ = tf.nn.dropout(X, 0.2)

    l1opts = tf.nn.relu(tf.matmul(X_, w_1) + bias_1)
    l1opts_ = tf.nn.dropout(l1opts, 0.2)

    l2opts = tf.nn.relu(tf.matmul(l1opts_, w_2) + bias_2)
    l2opts_ = tf.nn.dropout(l2opts, 0.2)

    l3opts = tf.nn.relu(tf.matmul(l2opts_, w_3) + bias_3)
    l3opts_ = tf.nn.dropout(l3opts, 0.2)

    l4opts = tf.nn.relu(tf.matmul(l3opts_, w_4) + bias_4)
    l4opts_ = tf.nn.dropout(l4opts, 0.2)

    loutopts = tf.matmul(l4opts_, w_out)


    predict = tf.argmax(loutopts, axis=1)
    predict = loutopts

    yp = tf.nn.softmax(Y)
    # softmax
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=loutopts)
    cost_ = tf.reduce_mean(cost)

    #mse
    cost = tf.losses.mean_squared_error(
        labels=Y,
        predictions=loutopts,
        weights=1.0,
        scope=None
    )
    cost_ = cost

    #ad
    cost = tf.losses.absolute_difference(
            labels=Y,
            predictions=loutopts,
            weights=1.0,
            scope=None
    )
    #cost_ = cost

    with tf.name_scope('accuracy'):
        # Get the prediction delta
        correct_prediction = tf.subtract(loutopts, Y)
        # absolute
        correct_prediction = tf.abs(correct_prediction)
    accuracy = tf.reduce_mean(correct_prediction)

    #update = tf.matrix_solve_ls(l4opts, Y, 0.01, fast=True)
    #cost = (tf.nn.l2_loss(l4opts - Y) + 
    #    0.01 * tf.nn.l2_loss(bias_1) +
    #    0.01 * tf.nn.l2_loss(bias_2) +
    #    0.01 * tf.nn.l2_loss(bias_3)
    #    ) / float(4096)
    update = tf.train.AdamOptimizer(0.001).minimize(cost_)



    train_acc = tf.equal(tf.argmax(loutopts, 1), tf.argmax(Y, 1))
    acc = loutopts

    # SGD
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer().run()
    #sess.run(init)

    bestLossIdx = 1e20
    bestIdx = 0
    nOver = 0
    bestCheckpointLocation = ""
    lastInc = 0

    graph_location = "/Scratch/cwa3/models/"
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    print("TESTING DATASET: %f %f %f %f" % (x_train[1][0], x_train[1][1], x_train[1][2], x_train[1][3]))

    saver = tf.train.Saver() # Checkpoints

    gi = 0

    for epoch in range(100):
        print("\n\nStarting Epoch: %d" % epoch)
        # Train
        for i in range(0, int(x_train.shape[0] / BATCH_SIZE)):
            gi += 1
#            sess.run(update,
#                feed_dict={
#                    X: x_train[i],
#                    Y: y_train[i]
#                })
        
            if i % 100 == 0:

                i_san = i % x_test.shape[0]
                i_max_san = (i + BATCH_SIZE) % x_test.shape[0]

                a,b,loss,d = sess.run(
                    [accuracy, acc, cost_, Y],
                    feed_dict=
                    {
                        X: x_test[i_san: i_max_san],
                        Y: y_test[i_san: i_max_san],
                    }
                )
                print("\n%d" % gi)
                print('Model Accuracy: %g' % a)
                #print(b)
                print('Mean Squared Loss: %g' % loss)
                #print(d)
                #print('step %d, cost %f' % (i, cost))

                if( loss < bestLossIdx ):
                    print("New best eval loss, model saved")
                    bestLossIdx = loss
                    bestIdx = i
                    nOver = 0
                    lastInc = 0

                    # use checkpoints only
                    bestCheckpointLocation = saver.save(sess, "/Scratch/cwa3/cpkt/model.cpkt")

                    #builder.add_meta_graph_and_variables(
                    #    sess,
                    #    [tag_constants.GPU],
                    #    signature_def_map=foo
                    #)
                elif( loss > lastInc ):
                    lastInc = loss
                    nOver += 1
                elif(nOver > 0):
                    lastInc = loss
                    nOver -= 1
                else:
                    lastInc = loss
                #print(nOver)
                if(nOver > 10):
                    print("ValError climbed for too long")
                    break
                
            update.run(feed_dict={
                X: x_train[i: i + BATCH_SIZE],
                Y: y_train[i: i + BATCH_SIZE]
            })

        #train_acc = np.mean(np.argmax(y_train, axis=1) == 
        #            sess.run(predict, feed_dict={X: x_train, Y: y_train}))

        #restore the network to the best state

    print("Restoring network")
    if(bestCheckpointLocation == ""):
        bestCheckpointLocation = "/Scratch/cwa3/cpkt/model.cpkt"

    saver.restore(sess, bestCheckpointLocation)

    def predictSet(dataset, filename):
        # Predict the xyzv10 set
        x10, y10 = createXandY(dataset, target_column)
        y_pred = (sess.run(predict, feed_dict={X: x10[0:BATCH_SIZE], Y: y10[0:BATCH_SIZE]}))
        nProcessed = BATCH_SIZE
        for i in range(1, int(x10.shape[0] / BATCH_SIZE)):
            nProcessed += BATCH_SIZE
            y_pred = np.append( 
                y_pred,
                (sess.run(predict, feed_dict={X: x10[i: i + BATCH_SIZE], Y: y10[i: i + BATCH_SIZE]})),
                axis=0
            )
        # Do anything left over at the end
        nLeft = x10.shape[0] - nProcessed
        y_pred = np.append( 
                y_pred,
                (sess.run(predict, feed_dict={
                    X: x10[nProcessed: nProcessed + nLeft],
                    Y: y10[nProcessed: nProcessed + nLeft]
                    })),
                axis=0
            )
        np.savetxt(filename, y_pred, delimiter=',')

    predictSet(TEST10, "/Scratch/cwa3/out_10.csv")
    predictSet(TEST20, "/Scratch/cwa3/out_20.csv")
    predictSet(TEST30, "/Scratch/cwa3/out_30.csv")
    predictSet(TEST40, "/Scratch/cwa3/out_40.csv")
    predictSet(TEST50, "/Scratch/cwa3/out_50.csv")
    predictSet(TEST60, "/Scratch/cwa3/out_60.csv")
    predictSet(TEST66, "/Scratch/cwa3/out_66.csv")
    sess.close()

if __name__ == "__main__":
    main()
