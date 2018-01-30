from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
#from six.moves.urllib.request import urlopen

import numpy as np
import pandas as pd
import tensorflow as tf

# Data sets
TRAINING = sys.argv[1]
VAL = sys.argv[2]
TEST10 = sys.argv[3]
TEST20 = sys.argv[4]
TEST30 = sys.argv[5]
TEST40 = sys.argv[6]
TEST50 = sys.argv[7]
TEST60 = sys.argv[8]
TEST66 = sys.argv[9]

TRAIN_SIZE=8192
BATCH_SIZE=4096


def main():
    """ Main """
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(TRAINING):
        sys.exit()

    res = pd.read_csv(TRAINING, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])
    val = pd.read_csv(VAL, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])
    #print(res.head(10))


    tf.logging.set_verbosity(tf.logging.ERROR)
#    # Load datasets.
#    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#        filename=TRAINING,
#        target_dtype=np.float32,
#        features_dtype=np.float32)
#    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#        filename=TEST,
#        target_dtype=np.float32,
#        features_dtype=np.float32)

    # Specify that all features have real-value data
    FEATURES = ['X','Y','Z','VI']
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
    export_input_fn = tf.contrib.learn.utils.build_parsing_serving_input_fn(feature_cols)

    print(FEATURES)

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_cols,
        #hidden_units=[50,50,50,50],
        #hidden_units=[31,31,31], # 1.X
        #hidden_units=[23,23,23],
        #hidden_units=[1024,256,64,256,1024], // 4.X
        #hidden_units=[5000,5000], # Terrible on 10 but excellant on 60
        #hidden_units=[70,70,70,70,70,70,70,70,70,70],
        #hidden_units=[50,1000,2000,3000,3000,2000,1000,50],
        #hidden_units=[100,200,200,512,512,200,200,100],
        #hidden_units=[30,30,30,30,30,30,30],
        #        hidden_units=[4,16,32,16,4],
        #hidden_units=[16,512,512,16,4],
        hidden_units=[25,25,25,25,25,25,25,25,25,25,25,25,25,25],
        dropout=0.4,
        model_dir='/Scratch/cwa3/model',
        optimizer=tf.train.AdamOptimizer(
            learning_rate=0.01
            )
    )

    # Define the training inputs
#  train_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": np.array(training_set.data)},
#      y=np.array(training_set.target),
#      num_epochs=None,
#      shuffle=True)

    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: res[k].values for k in FEATURES}),
        y=pd.Series(res['V'].values),
        batch_size=TRAIN_SIZE,
        num_epochs=None,
        shuffle=True
    )

    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: val[k].values for k in FEATURES}),
        y=pd.Series(val['V'].values),
        batch_size=TRAIN_SIZE,
        num_epochs=None,
        shuffle=True
    )
    
    #train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={'x': pd.DataFrame({k: res[k].values for k in FEATURES}).values},
    #    y=np.array(res["VP"].values),
    #    batch_size=128,
    #    num_epochs=50,
    #    shuffle=True
    #)
    
    bestLossIdx = 1e20
    bestIdx = 0
    nOver = 0
    lastInc = 0

    def serving_input_receiver_fn():
        """Build the serving inputs."""
        # The outer dimension (None) allows us to batch up inputs for
        # efficiency. However, it also means that if we want a prediction
        # for a single instance, we'll need to wrap it in an outer list.
        inputs = {
            "X": tf.placeholder(shape=[1], dtype=tf.float32),
            "Y": tf.placeholder(shape=[1], dtype=tf.float32),
            "Z": tf.placeholder(shape=[1], dtype=tf.float32),
            "VI": tf.placeholder(shape=[1], dtype=tf.float32)
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        bestCheckoutPath = ""

    sess = tf.get_default_session()


    for i in range(0,1):
        a = regressor.train(input_fn=train_input_fn,steps=50)
        b = regressor.evaluate(input_fn=train_input_fn, steps=50)
        #print(a["average_loss"])
        #print("AVG LOSS:" + b["average_loss"])
        #print("\nTest Accuracy: {0:f}\n".format(b["accuracy"]))
        print("{} Avg Loss: {}".format(i,b["average_loss"]))
        if( b["average_loss"] < bestLossIdx ):
            print("New best eval loss")
            bestLossIdx = b["average_loss"]
            bestIdx = i
            nOver = 0
            lastInc = 0
            bestCheckoutPath = regressor.export_savedmodel(
                "/Scratch/cwa3/cpkt",
                serving_input_receiver_fn
                #serving_input_fn=export_input_fn
            )
        elif( b["average_loss"] > lastInc ):
            lastInc = b["average_loss"]
            nOver += 1
        elif(nOver > 0):
            lastInc = b["average_loss"]
            nOver -= 1
        else:
            lastInc = b["average_loss"]
        print(nOver)
        if(nOver > 3):
            print("ValError climbed for too long")
            break
        



    #
    #print("Exporting Model")
    #builder = tf.saved_model.builder.SavedModelBuilder("Models/")
    #regressor.export_savedmodel("Models/", )

    #feature_spec = {'foo': tf.FixedLenFeature(...),
    #            'bar': tf.VarLenFeature(...)}

    #def serving_input_receiver_fn():
    #    """An input receiver that expects a serialized tf.Example."""
    #    serialized_tf_example = tf.placeholder(dtype=tf.string,
    #                                     shape=[default_batch_size],
    #                                     name='input_example_tensor')
    #    receiver_tensors = {'examples': serialized_tf_example}
    #    features = tf.parse_example(serialized_tf_example, feature_spec)
    #    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    #estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn)
    test = pd.read_csv(TEST10, delimiter=',', skiprows=1, names=['X', 'Y', 'Z', 'V','VI', 'VP'])
    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )


    predict_fn = tf.contrib.predictor.from_saved_model(bestCheckoutPath, signature_def="predict");
    predictions = predict_fn(
        test_input_fn
    )
    y_pred = list(p['predictions'] for p in predictions)
    np.savetxt("/Scratch/cwa3/out_test.csv",y_pred,delimiter=",")

    sys.exit(1)

    # 10

    test = pd.read_csv(TEST10, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])
    


    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_10.csv",y_pred,delimiter=",")

    del test
    

    # 20
    
    test = pd.read_csv(TEST20, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_20.csv",y_pred,delimiter=",")

    del test
    

    #30
    
    test = pd.read_csv(TEST30, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_30.csv",y_pred,delimiter=",")

    del test
    

    #40
    
    test = pd.read_csv(TEST40, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_40.csv",y_pred,delimiter=",")

    del test
    

    #50
    
    test = pd.read_csv(TEST50, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_50.csv",y_pred,delimiter=",")

    del test
    

    #60

    test = pd.read_csv(TEST60, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_60.csv",y_pred,delimiter=",")

    del test
    

    #66
    
    test = pd.read_csv(TEST66, delimiter=',', skiprows=1,names=['X', 'Y', 'Z', 'V','VI', 'VP'])

    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: test[k].values for k in FEATURES}),
        y=pd.Series(test['V'].values),
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    pred = regressor.predict(test_input_fn)
    y_pred = list(p['predictions'] for p in pred)
    np.savetxt("/Scratch/cwa3/out_66.csv",y_pred,delimiter=",")

    del test
    

    sys.exit(1)
  
    # Define the test inputs
#  test_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": np.array(test_set.data)},
#      y=np.array(test_set.target),
#      num_epochs=None,
#      shuffle=False)

    # Evaluate accuracy.
    #accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    #print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

# Classify two new flower samples.
#    new_samples = np.array(
#        [[6.4, 3.2, 4.5, 1.5],
#         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
#    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#        x={"x": new_samples},
#        num_epochs=1,
#        shuffle=False)

    #predictions = list(classifier.predict(input_fn=predict_input_fn))
    #predicted_classes = [p["classes"] for p in predictions]

    #print(
    #    "New Samples, Class Predictions:    {}\n"
    #    .format(predicted_classes))


if __name__ == "__main__":
    main()
