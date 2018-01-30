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
MDL1015 = sys.argv[1]
MDL1020 = sys.argv[2]
MDL1520 = sys.argv[3]
TEST1 = sys.argv[4]
TEST10 = sys.argv[5]
TEST20 = sys.argv[6]
TEST30 = sys.argv[7]
TEST40 = sys.argv[8]
TEST50 = sys.argv[9]
TEST60 = sys.argv[10]
TEST66 = sys.argv[11]



TRAIN_SIZE=8192
BATCH_SIZE=4096

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def createXandY(load, target_column):
    train=iter_loadtxt(load,delimiter=',',skiprows=1)
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

def createTrainAndVal(ltrain,lval):
    print("\nReading data...")
    train=iter_loadtxt(TRAINING,delimiter=',',skiprows=1)
    test =iter_loadtxt(VAL,delimiter=',',skiprows=1)
    print("#training ex.:\t",train.shape[0])
    print("#testing ex.:\t",test.shape[0])
    print(train.shape[1])
    target_column=train.shape[1]-3  # assume target is last column
    print("target column:\t",target_column)
    #x_train=train
    y_train_one = train[:,target_column]
    x_train = np.delete(train, np.s_[3:4], axis=1)
    N,M = x_train.shape

    y_train = y_train_one#.T
    y_train = np.reshape(y_train, (N, 1))
    print(y_train.shape)
    y_test = test[:,target_column]
    N,M = test.shape
    y_test = np.reshape(y_test, (N, 1))
    x_test = np.delete(test, np.s_[3:4], axis=1)

    

    # Delete original copies of train/test
    train=None
    test=None

    return (x_train, y_train, x_test, y_test, target_column)

def main():
    """ Main """
    TOOMANYARGS = ""
    try:
        TOOMANYARGS = sys.argv[12]
    except:
        pass
    finally:
        if(TOOMANYARGS != ""):
            print("Too Many Arguments")
            sys.exit(-1)

    # for dry
    target_column = 3


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

    def predictSet(dataset, filename):
        x10, y10 = createXandY(dataset, target_column)
        y_pred = (sess.run(predict, feed_dict={X: x10[0:BATCH_SIZE], Y: y10[0:BATCH_SIZE]}))
        nProcessed = BATCH_SIZE
        for i in range(1, int(x10.shape[0] / BATCH_SIZE)):
            if(i % 1000 == 0):
                print("Batch : %d" % i)
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
        return y_pred
        #np.savetxt(filename, y_pred, delimiter=',')

    def predictAndWrite(dataset, filename, tfsaver):
        # Predict the 1015
        print("Predicting 10.15 for %s" % dataset)
        tfsaver.restore(sess, os.path.join(os.getcwd(), MDL1015))
        p1015 = predictSet(dataset,filename)
        # Predict the 1020
        print("Predicting 10.20 for %s" % dataset)
        tfsaver.restore(sess, os.path.join(os.getcwd(), MDL1020))
        p1020 = predictSet(dataset,filename)
        # Predict the 1520
        print("Predicting 15.20 for %s" % dataset)
        tfsaver.restore(sess, os.path.join(os.getcwd(), MDL1520))
        p1520 = predictSet(dataset,filename)

        avgPred = (p1015 + p1020 + p1520) / 3
        np.savetxt(filename, avgPred, delimiter=',')




    saver = tf.train.Saver() # Checkpoints
    #print("Restoring network")

    # Restore from 1015
    predictAndWrite(TEST1, "/Scratch/cwa3/out_1.csv", saver)
    predictAndWrite(TEST10, "/Scratch/cwa3/out_10.csv", saver)
    predictAndWrite(TEST20, "/Scratch/cwa3/out_20.csv",saver)
    predictAndWrite(TEST30, "/Scratch/cwa3/out_30.csv",saver)
    predictAndWrite(TEST40, "/Scratch/cwa3/out_40.csv",saver)
    predictAndWrite(TEST50, "/Scratch/cwa3/out_50.csv",saver)
    predictAndWrite(TEST60, "/Scratch/cwa3/out_60.csv",saver)
    predictAndWrite(TEST66, "/Scratch/cwa3/out_66.csv",saver)


    sess.close()

if __name__ == "__main__":
    main()
