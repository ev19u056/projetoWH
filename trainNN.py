# Importing everything
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
from keras import *
from keras.optimizers import Adam, Nadam
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.layers import Dense
import numpy
import time
import pandas
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from commonFunctions import getYields, FullFOM, myClassifier, gridClassifier, getDefinedClassifier, assure_path_exists, plotter, NNarch
import matplotlib.pyplot as plt
#from scipy.stats import ks_2samp
import localConfig as cfg
import pickle


if __name__ == "__main__":
    import argparse
    import sys

    # Input arguments
    parser = argparse.ArgumentParser(description='Process the command line options')
    #parser.add_argument('-L', '--layers', type=int, required=False, help='Number of layers')
    #parser.add_argument('-n', '--neurons', type=int, required=False, help='Number of neurons per layer')
    parser.add_argument('-z', '--batch', action='store_true', help='Whether this is a batch job, if it is, no interactive questions will be asked and answers will be assumed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-a', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-b', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-c', '--decay', type=float, default=0, help='Learning rate decay')
    parser.add_argument('-d', '--dropoutRate', type=float, default=0, help='Drop-out rate')
    parser.add_argument('-r', '--regularizer', type=float, default=0, help='Regularizer')
    parser.add_argument('-i', '--iteration', type=str, default=1.0, help='Version number i')
    parser.add_argument('-l', '--list', type=str, required=True, help='Defines the architecture of the NN; e.g: -l "14 12 7"  ->3 hidden layers of 14, 12 and 7 neurons respectively (input always 12, output always 1)')
    parser.add_argument('-ini', '--initializer', type=str, default="glorot_uniform", help='Kernel Initializer for hidden layers')
    parser.add_argument('-act', '--act', type=str, default="relu", help='activation function for the hidden neurons')
    parser.add_argument('-bk', '--bk', action='store_true', help='Whether or not you choose to load Zinv background samples or only W+jets and TTpow')

    args = parser.parse_args()

    if args.bk:
         from prepareDATA_2_background import *
    else:
        from prepareDATA import *

    #n_layers = args.layers
    #n_neurons = ars.neurons
    act = args.act
    n_epochs = args.epochs
    batch_size = args.batchSize
    learning_rate = args.learningRate
    my_decay = args.decay
    dropout_rate = args.dropoutRate
    regularizer = args.regularizer
    iteration = args.iteration
    ini = args.initializer
    list=args.list
    architecture=list.split()


    verbose = 0
    if args.verbose:
        verbose = 1

    # Model's compile arguments, training parameters and optimizer.
    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
    myOpt = Adam(lr=learning_rate)#, decay=my_decay)
    compileArgs['optimizer'] = myOpt

    # Naming the Model
    name ="Model_Ver_"+iteration

    # Creating the directory where the fileswill be stored
    testpath =cfg.lgbk + "test/"
    filepath =testpath + "{}/".format(name)

    if not os.path.exists(filepath):
        os.mkdir(filepath)

    # Printing stuff and starting time
    if args.verbose:
        print("Dir "+filepath+" created.")
        print("Starting the training")
        start = time.time()

    # Model's architecture
    model = Sequential()

    model.add(Dense(int(architecture[0]), input_dim=12, activation=act , kernel_initializer=ini))
    i=1
    while i < len(architecture) :
        model.add(Dense(int(architecture[i]), activation=act , kernel_initializer=ini))
        i=i+1
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(**compileArgs)

    #Fitting the Model
    history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,shuffle=True, **trainParams)

    acc = history.history["acc"]
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    model.save(filepath+name+".h5")
    model_json = model.to_json()
    with open(filepath+name + ".json", "w") as json_file:
      json_file.write(model_json)
    model.save_weights(filepath+name  + "_w.h5")

    #Getting predictions
    if args.verbose:
        print("Getting predictions")

    devPredict = model.predict(XDev) # não se utiliza
    valPredict = model.predict(XVal) # não se utiliza

    #Getting scores
    if args.verbose:
        print("Getting scores")

    scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0) # não se utiliza
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0) # não se utiliza

    # CAlculating FOM
    if args.verbose:
        print "Calculating FOM:"
    dataVal["NN"] = valPredict

    tmpSig, tmpBkg = getYields(dataVal)
    sigYield, sigYieldUnc = tmpSig
    bkgYield, bkgYieldUnc = tmpBkg

    sigDataVal = dataVal[dataVal.category==1]
    bkgDataVal = dataVal[dataVal.category==0]

    fomEvo = []
    fomCut = []

    for cut in np.arange(0.0, 0.9999999, 0.001):
      sig, bkg = getYields(dataVal, cut=cut)
      if sig[0] > 0 and bkg[0] > 0:
        fom, fomUnc = FullFOM(sig, bkg)
        fomEvo.append(fom)
        fomCut.append(cut)

    max_FOM=0

    # Maximising FOM
    if args.verbose:
        print "Maximizing FOM"

    for k in fomEvo:
      if k>max_FOM:
        max_FOM=k
    if args.verbose:
        print "Signal@Presel:", sigDataVal.weight.sum() * 35866 * 2
        print "Background@Presel:", bkgDataVal.weight.sum() * 35866 * 2
        print "Signal:", sigYield, "+-", sigYieldUnc
        print "Background:", bkgYield, "+-", bkgYieldUnc

        print "Maximized FOM:", max_FOM
        print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]


     # Creating a text file where all of the model's caracteristics are displayed
    if args.bk:
        f=open(testpath + "README.md", "a")
        f.write("\n \n **{}** : Neuron-Layers: 12 {} 1 ; Activation: {} ; Output: Sigmoid ; Batch size:{} ; Epochs: {} ; Step size: {} ; Optimizer: Adam ; Regulizer: {} ; Max FOM : {} ; Weight Initializer: {} (W+jets and TTpow background)  \n ".format(name, list, act, batch_size, n_epochs, learning_rate, regularizer, max_FOM , ini ))
        f.close()
    else:
        f=open(testpath + "README.md", "a")
        f.write("\n \n **{}** : Neuron-Layers: 12 {} 1 ; Activation: {} ; Output: Sigmoid ; Batch size:{} ; Epochs: {} ; Step size: {} ; Optimizer: Adam ; Regulizer: {} ; Max FOM : {} ; Weight Initializer: {}   \n ".format(name, list, act, batch_size, n_epochs, learning_rate, regularizer, max_FOM , ini ))
        f.close()

    # Plot accuracy and loss evolution over epochs for both training and validation datasets
    if not args.batch:
        from commonFunctions import plotter
        fig=plt.figure()
        plt.subplots_adjust(hspace=0.5)


        plt.subplot(2,1,1)
        plotter(filepath+"accuracy/acc_"+name+".pickle","accuracy",name+"'s accuracy")
        plotter(filepath+"accuracy/val_acc_"+name+".pickle","Val accuracy",name+"'s Accuracy")
        #plt.savefig(filepath+"accuracy/Accuracy.pdf")

        plt.subplot(2,1,2)
        plotter(filepath+"loss/loss_"+name+".pickle","loss",name +"loss function")
        plotter(filepath+"loss/val_loss_"+name+".pickle","loss Validation",name+"'s Loss")
        #plt.savefig(filepath + "loss/Loss_Validation.pdf")

        plt.savefig(filepath+name+"Accuracy_Loss_"+compileArgs['loss']+".pdf")
        plt.close()


        if args.verbose:
            print("Accuraccy and loss plotted at {}".format(filepath))
            print ("Model name: "+name)
        sys.exit("Done!")
