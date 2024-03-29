'''
Train the Neural Network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
from keras import *
from keras.optimizers import Adam#, Nadam
import time
import pandas
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation#, Dropout, AlphaDropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback
from keras import backend

from matplotlib.backends.backend_pdf import PdfPages
from commonFunctions import assure_path_exists, plotter, getYields, FullFOM#, myClassifier, gridClassifier, getDefinedClassifier,
import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import ks_2samp

import localConfig as cfg
import pickle   # This module is used for serializing and de-serializing a Python object structure

from prepareData import dataLoader

# a custom Callback
# monitor the learning rate
class LearningRateMonitor(Callback):
    # start of training
    def on_train_begin(self, logs={}):
        self.lrates = list()

    # end of each training epoch
    def on_epoch_end(self, epoch, logs={}):
        # get and store the learning rate
        optimizer = self.model.optimizer
        lrate = float(backend.get_value(self.model.optimizer.lr))
        self.lrates.append(lrate)

if __name__ == "__main__":
    import argparse
    import sys

    ## Input arguments. Pay speciall attention to the required ones.
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-a', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-b', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-c', '--decay', type=float, default=0.1, help='Learning rate decay')
    #parser.add_argument('-d', '--dropoutRate', type=float, default=0, help='Drop-out rate')
    parser.add_argument('-r', '--regularizer', type=float, default=0, help='Regularizer')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration number i')
    parser.add_argument('-f', '--fraction', type=float, default=0.3, help="The fraction of available data to be loaded")
    parser.add_argument('-l', '--List', type=str, required=True, help='Defines the architecture of the NN; e.g: -l "14 12 7" -> 3 hidden layers of 14, 12 and 7 neurons respectively (input always 53, output always 1)')

    parser.add_argument('-act', '--act', type=str, default="relu", help='activation function for the hidden neurons')
    parser.add_argument('-ini', '--initializer', type=str, default="he_normal", help='Kernel Initializer for hidden layers')
    parser.add_argument('-bN', '--batchNorm', action='store_true',help='Wether to use Batch Normalization')
    parser.add_argument('-es', '--EarlyStopping', action='store_true',help='Wether to use EarlyStopping Callback')
    parser.add_argument('-rlrop', '--ReduceLROnPlateau', action='store_true',help='Wether to use ReduceLROnPlateau Callback')

    args = parser.parse_args()

    n_epochs = args.epochs
    batch_size = args.batchSize
    learning_rate = args.learningRate
    my_decay = args.decay
    regularizer = args.regularizer
    iteration = args.iteration
    act = args.act # activation function for hidden neurons
    batch_size = args.batchSize
    List = args.List
    architecture = List.split()
    ini = args.initializer
    fraction = args.fraction

    verbose = 0
    if args.verbose:
        verbose = 1

    ## Model compile arguments, training parameters and optimizer.

    # compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}  # compileArgs is a dictionary that contains parameters for model.compile()
    trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
    myOpt = Adam(lr=learning_rate)#, decay=my_decay)
    compileArgs['optimizer'] = myOpt

    # Naming the Model
    name ="Model_Ver_"+str(iteration)

    # Creating the directory where the fileswill be stored
    testpath =cfg.lgbk + "test/"
    filepath = cfg.lgbk+"test/"+name+"/"

    ## Directory to save your NN files. Edit lgbk variable in localConfig.py
    # lgbk = "/home/t3atlas/ev19u056/projetoWH/"

    if os.path.exists(filepath) == False:
        os.mkdir(filepath)

    _, _, dataTest, XDev, YDev, weightDev, XVal, YVal, weightVal, XTest, YTest, weightTest = dataLoader(filepath, name, fraction)

    os.chdir(filepath)
    # Printing stuff and starting time
    if args.verbose:
        print("Dir "+filepath+" created.")
        print("Starting the training")
        start = time.time()

    ## EXERCISE 2: Create your NN model
    model = Sequential()

    if not args.batchNorm:
        if args.verbose:
            print("# WARNING: No BatchNormalization!")
        model.add(Dense(int(architecture[0]), input_dim=53, activation=act , kernel_initializer=ini)) # input + 1st hidden layer
        i=1
        while i < len(architecture) :
            model.add(Dense(int(architecture[i]), activation=act , kernel_initializer=ini))
            i=i+1
        model.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_normal')) # output
    else:
        if args.verbose:
            print("# WARNING: BatchNormalization will be used!")
        model.add(Dense(int(architecture[0]),use_bias=False, input_dim=53, kernel_initializer=ini)) # input + 1st hidden layer
        model.add(BatchNormalization())
        model.add(Activation(act))
        i=1
        while i < len(architecture):
            model.add(Dense(int(architecture[i]),use_bias=False, kernel_initializer=ini))
            model.add(BatchNormalization())
            model.add(Activation(act))
            i=i+1
        model.add(Dense(1, use_bias=False,kernel_initializer='glorot_normal')) # output
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

    # Compile
    model.compile(**compileArgs)

    callbacks = []
    lrm = LearningRateMonitor()
    ModelCheckpoint = ModelCheckpoint(filepath+name+".h5", save_best_only=True, save_weights_only=True)
    callbacks.append(lrm)
    callbacks.append(ModelCheckpoint)

    if args.EarlyStopping:
        EarlyStopping = EarlyStopping(patience=15, verbose=True)
        callbacks.append(EarlyStopping)

    if args.ReduceLROnPlateau:
        ReduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=my_decay, patience=10, verbose=True, cooldown=1, min_lr=0.000001) # argument min_delta is not supported
        callbacks.append(ReduceLROnPlateau)

    # Fitting the Model -> TRAINING
    # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
    history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal),sample_weight=weightDev,shuffle=True,callbacks=callbacks, **trainParams)

    # Time of the training
    training_time = time.time()-start
    if args.verbose:
        print "Training took ", training_time, " seconds"

    lr_list = lrm.lrates

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # assure_path_exists() is defined in commonFunctions.py
    assure_path_exists(filepath+"accuracy/")
    assure_path_exists(filepath+"loss/")

    # Saving accuracy and loss values in a pickle file for later plotting
    pickle.dump(acc, open(filepath+"accuracy/acc_"+name+".pickle", "wb"))
    pickle.dump(loss, open(filepath+"loss/loss_"+name+".pickle", "wb"))
    pickle.dump(val_acc, open(filepath+"accuracy/val_acc_"+name+".pickle", "wb"))
    pickle.dump(val_loss, open(filepath+"loss/val_loss_"+name+".pickle", "wb"))

    # Saving the trainned model
    model_json = model.to_json()    # model structure
    with open(filepath+name + ".json", "w") as json_file:
      json_file.write(model_json)

    # Restoring the best weights
    model.load_weights(filepath+name+".h5")
    model.compile(**compileArgs)

    # Getting predictions
    if args.verbose:
        print("Getting predictions...")
    testPredict = model.predict(XTest) # generates output predictions based on the input you pass it

    # Getting scores
    if args.verbose:
        print("Getting scores...")

    scoreDev = model.evaluate(XDev, YDev,sample_weight=weightDev, verbose = 0) # computes the loss based on the input you pass it, along with any other metrics that you requested in the metrics param when you compiled your model
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)
    scoreTest = model.evaluate(XTest, YTest, sample_weight=weightTest, verbose = 0)
    print "Dev_loss: ", scoreDev[0], "     Dev_acc: ", scoreDev[1]
    print "Val_loss: ", scoreVal[0], "     Val_acc: ", scoreVal[1]
    print "Test_loss: ", scoreTest[0], "   Test_acc: ", scoreTest[1]

    # Creating a text file where all of the model's caracteristics are displayed
    f=open(testpath + "README.md", "a")
    # f.write("\n \n **{}** : Neuron-Layers: 53 {} 1 ; Activation: {} ; Output: Sigmoid ; Batch size: {} ; Epochs: {} ; Step size: {} ; Optimizer: Adam ; Regulizer: {} ; Max FOM : {} ; Weight Initializer: {}   \n ".format(name, List, act, batch_size, n_epochs, learning_rate, regularizer, max_FOM, ini ))
    f.write("\n \n **{}** : Neuron-Layers: 53 {} 1 ; Activation: {} ; Output: Sigmoid ; BatchNormalization: {} ; Batch size: {} ; Epochs: {} ; Optimizer: Adam ; Regulizer: {} ; Weight Initializer: {}\n".format(name, List, act, args.batchNorm, batch_size, len(history.history['loss']), regularizer, ini))
    f.write("Dev_loss:  {}   Dev_acc:  {}\n".format(scoreDev[0], scoreDev[1]))
    f.write("Val_loss:  {}   Val_acc:  {}\n".format(scoreVal[0], scoreVal[1]))
    f.write("Test_loss: {}   Test_acc: {}\n".format(scoreTest[0], scoreTest[1]))
    f.write("Training_time: {0:.2f}\n".format(training_time))
    f.close()
    print("DONE: Creating a text file where all of the model's caracteristics are displayed")

    # Plot accuracy and loss evolution over epochs for both training and validation datasets
    pdf_pages = PdfPages(filepath+name+"_Accuracy_Loss_"+compileArgs['loss']+".pdf")

    fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2,1,1)
    plotter(filepath+"accuracy/acc_"+name+".pickle","accuracy",name+"'s accuracy")
    plotter(filepath+"accuracy/val_acc_"+name+".pickle","Val accuracy",name+"'s Accuracy")
    plt.grid()
    plt.legend(['train', 'val'], loc='lower right')

    plt.subplot(2,1,2)
    plotter(filepath+"loss/loss_"+name+".pickle","loss",name +"loss function")
    plotter(filepath+"loss/val_loss_"+name+".pickle","loss Validation",name+"'s Loss")
    plt.grid()
    plt.legend(['train', 'val'], loc='upper right')
    pdf_pages.savefig(fig)
    plt.close()

    if args.ReduceLROnPlateau:
        fig = plt.figure(figsize=(8.27, 5.845), dpi=100)
        pickle.dump(lrm.lrates, open(filepath+"lr_"+name+".pickle", "wb"))
        plotter(filepath+"lr_"+name+".pickle","learning Rate",name +"'s Learning Rate, Factor=" + str(my_decay), log=True)
        plt.grid()
        plt.legend(['lr'], loc='upper right')
        pdf_pages.savefig(fig)
        plt.close()
    pdf_pages.close()

    if args.verbose:
        print("Accuraccy and loss plotted at {}".format(filepath))
        print ("Model name: "+name)
    sys.exit("Done!")
