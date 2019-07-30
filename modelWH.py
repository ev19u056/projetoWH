'''
Train the Neural Network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
from keras import *
from keras.optimizers import Adam, Nadam
import time
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
#from sklearn.metrics import confusion_matrix, cohen_kappa_score

''' copy to your folder and adapt commonFunctions.py'''
from commonFunctions import assure_path_exists, plotter #getYields, FullFOM, myClassifier, gridClassifier, getDefinedClassifier,
#from scipy.stats import ks_2samp

''' copy to your folder and adapt localConfig.py '''
import localConfig as cfg
import pickle   # This module is used for serializing and de-serializing a Python object structure

from prepareData import *

if __name__ == "__main__":
    import argparse
    import sys

    ## Input arguments. Pay speciall attention to the required ones.
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-z', '--batch', action='store_true', help='Whether this is a batch job, if it is, no interactive questions will be asked and answers will be assumed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    #parser.add_argument('-l', '--layers', type=int, required=True, help='Number of layers')
    #parser.add_argument('-n', '--neurons', type=int, required=True, help='Number of neurons per layer')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-a', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-b', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-c', '--decay', type=float, default=0, help='Learning rate decay')
    parser.add_argument('-d', '--dropoutRate', type=float, default=0, help='Drop-out rate')
    parser.add_argument('-r', '--regularizer', type=float, default=0, help='Regularizer')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration number i')

    parser.add_argument('-l', '--list', type=str, required=True, help='Defines the architecture of the NN; e.g: -l "14 12 7" -> 3 hidden layers of 14, 12 and 7 neurons respectively (input always 62, output always 1)')
    parser.add_argument('-act', '--act', type=str, default="relu", help='activation function for the hidden neurons')
    parser.add_argument('-ini', '--initializer', type=str, default="glorot_uniform", help='Kernel Initializer for hidden layers')

    args = parser.parse_args()

    #n_layers = args.layers
    #n_neurons = args.neurons
    n_epochs = args.epochs
    batch_size = args.batchSize
    learning_rate = args.learningRate
    my_decay = args.decay
    dropout_rate = args.dropoutRate
    regularizer = args.regularizer
    iteration = args.iteration

    act = args.act # activation function for hidden neurons
    batch_size = args.batchSize
    list = args.list
    architecture = list.split()
    ini = args.initializer
    verbose = 0
    if args.verbose:
        verbose = 1

    ## Model compile arguments, training parameters and optimizer.

    # compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}  # compileArgs is a dictionary that contains parameters for model.compile()
    trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
    myOpt = Adam(lr=learning_rate)#, decay=my_decay)
    compileArgs['optimizer'] = myOpt

    '''
    name = "L"+str(n_layers)+"_N"+str(n_neurons)+"_E"+str(n_epochs)+"_Bs"+str(batch_size)+"_Lr"
                +str(learning_rate)+"_De"+str(my_decay)+"_Dr"+str(dropout_rate)+"_L2Reg"+str(regularizer)
                #+"_Tr"+train_DM+"_Te"+test_point+"_DT"+suffix # variables does not defined !!!
    if iteration > 0:
        name = name+"_Ver"+str(iteration)
    '''
    # Naming the Model
    name ="Model_Ver_"+str(iteration)

    # Creating the directory where the fileswill be stored
    testpath =cfg.lgbk + "test/"
    filepath = cfg.lgbk+"test/"+name+"/"

    ## Directory to save your NN files. Edit lgbk variable in localConfig.py
    # lgbk = "/home/t3atlas/ev19u056/projetoWH/"

    if os.path.exists(filepath) == False:
        os.mkdir(filepath)
    os.chdir(filepath)

    # Printing stuff and starting time
    if args.verbose:
        print("Dir "+filepath+" created.")
        print("Starting the training")
        start = time.time()

    ## EXERCISE 2: Create your NN model
    model = Sequential()

    model.add(Dense(int(architecture[0]), input_dim=62, activation=act , kernel_initializer=ini)) # input + 1st hidden layer
    i=1
    while i < len(architecture) :
        model.add(Dense(int(architecture[i]), activation=act , kernel_initializer=ini))
        i=i+1
    model.add(Dense(1, activation='sigmoid')) # output

    # Compile
    model.compile(**compileArgs)

    # Fitting the Model -> TRAINING
    # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
    history = model.fit(XDev, YDev, validation_data=(XVal,YVal),shuffle=True, **trainParams)

    acc = history.history["acc"]
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

    # Time of the training
    if args.verbose:
        print("Training took ", time.time()-start, " seconds")

    # Saving the trainned model and his weights
    model.save(filepath+name+".h5") # model weights
    model_json = model.to_json()    # model structure
    with open(filepath+name + ".json", "w") as json_file:
      json_file.write(model_json)
    # another file to save model weights ???
    model.save_weights(filepath+name  + "_w.h5")

    # Getting predictions
    if args.verbose:
        print("Getting predictions")

    devPredict = model.predict(XDev) # nao se utiliza
    valPredict = model.predict(XVal) # nao se utiliza

    '''
    # Getting scores
    if args.verbose:
        print("Getting scores")

    scoreDev = model.evaluate(XDev, YDev, verbose = 0) # nao se utiliza
    scoreVal = model.evaluate(XVal, YVal, verbose = 0) # nao se utiliza
    '''

    # evaluate the keras model
    loss, accuracy = model.evaluate(X, dummy_y)
    print('Accuracy: %.2f' % (accuracy*100))
    print('Loss: %.2f' % (loss*100))
    # --- CAlculating FOM ---
    #
    #
    #
    # --- CAlculating FOM ---

    # Creating a text file where all of the model's caracteristics are displayed
    f=open(testpath + "README.md", "a")
    f.write("\n \n **{}** : Neuron-Layers: 62 {} 1 ; Activation: {} ; Output: Sigmoid ; Batch size:{} ; Epochs: {} ; Step size: {} ; Optimizer: Adam ; Regulizer: {} ; Weight Initializer: {}   \n ".format(name, list, act, batch_size, n_epochs, learning_rate, regularizer, ini ))
    f.close()
    print("DONE: Creating a text file where all of the model's caracteristics are displayed")

    # Plot accuracy and loss evolution over epochs for both training and validation datasets
    if not args.batch:
        fig=plt.figure()
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(2,1,1)
        plotter(filepath+"accuracy/acc_"+name+".pickle","accuracy",name+"'s accuracy")
        plotter(filepath+"accuracy/val_acc_"+name+".pickle","Val accuracy",name+"'s Accuracy")
        plt.legend(['train', 'test'], loc='upper left')
        #plt.savefig(filepath+"accuracy/Accuracy.pdf")

        plt.subplot(2,1,2)
        plotter(filepath+"loss/loss_"+name+".pickle","loss",name +"loss function")
        plotter(filepath+"loss/val_loss_"+name+".pickle","loss Validation",name+"'s Loss")
        plt.legend(['train', 'test'], loc='upper left')
        #plt.savefig(filepath + "loss/Loss_Validation.pdf")

        plt.savefig(filepath+name+"Accuracy_Loss_"+compileArgs['loss']+".pdf")
        plt.close()

        if args.verbose:
            print("Accuraccy and loss plotted at {}".format(filepath))
            print ("Model name: "+name)
        sys.exit("Done!")
    '''
    #Getting predictions
    if args.verbose:
        print("Getting predictions")

    # predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
    # Generates output predictions for the input samples. Computation is done in batches.
    devPredict = model.predict(XDev)
    valPredict = model.predict(XVal)

    #Getting scores
    if args.verbose:
        print("Getting scores")

    # evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
    # Returns the loss value & metrics values for the model in test mode. Computation is done in batches.
    scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)

    # CAlculating FOM
    if args.verbose:
        print "Calculating FOM:"
    dataVal["NN"] = valPredict

    tmpSig, tmpBkg = getYields(dataVal)
    sigYield, sigYieldUnc = tmpSig
    bkgYield, bkgYieldUnc = tmpBkg
    '''
