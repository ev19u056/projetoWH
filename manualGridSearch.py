"""
Make a Grid Search varying the number of neurons per layer and the number of layers.
Plot the results to using the area under the ROC curve as a metric of performance.
"""
import os
import numpy as np
import pandas
import keras
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense#, Dropout, AlphaDropout
from keras.optimizers import Adam#, Nadam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score#, confusion_matrix
from scipy.stats import ks_2samp
import localConfig as cfg
import pickle
import sys
from math import log

from commonFunctions import FullFOM, getYields, getDefinedClassifier, assure_path_exists
from prepareData import *
from modelWH import LearningRateMonitor

if __name__ == "__main__":
    import argparse
    import sys

    luminosity = 139500 # pb^-1
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-r', '--learningRate', type=float, required=True, help='Learning rate')
    #parser.add_argument('-d', '--decay', type=float, required=True, help='Learning rate decay') # ???
    parser.add_argument('-l', '--layers', type=int, required=False, help='Number of layers')
    parser.add_argument('-n', '--neurons', type=int, required=False, help='Number of neurons per layer')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-b', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-o', '--outputDir', required=True, help='Output directory')
    parser.add_argument('-p', '--dropoutRate', type=float, default=0, help='Dropout Rate')
    parser.add_argument('-f', '--fraction', type=float, default=0.3, help="The fraction of available data to be loaded")

    args = parser.parse_args()

    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    trainParams = {'epochs': args.epochs, 'batch_size': args.batchSize, 'verbose': 0}
    learning_rate = args.learningRate
    #my_decay = args.decay
    fraction = args.fraction
    myAdam = Adam(lr=learning_rate)#, decay=my_decay)
    compileArgs['optimizer'] = myAdam

    # --- CallBacks --- #
    lrm = LearningRateMonitor()
    callbacks = [EarlyStopping(patience=15, verbose=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=True, cooldown=1, min_lr=0), # argument min_delta is not supported
                    lrm]#, ModelCheckpoint(filepath+name+".h5", save_best_only=True, save_weights_only=True)]

    # lgbk = "/home/t3atlas/ev19u056/projetoWH/"
    hyperParam = args.outputDir
    # filepath = "/home/t3atlas/ev19u056/projetoWH/gridSearch/batchSize/"
    filepath = cfg.lgbk + 'gridSearch/' + hyperParam + '/'
    assure_path_exists(filepath)
    dataDev, dataVal, dataTest, XDev, YDev, weightDev, XVal, YVal, weightVal, XTest, YTest, weightTest = dataLoader(filepath, hyperParam, fraction)

    if args.verbose:
        print "Opening files..."

    fileToPlot = "ROC_" + args.outputDir

    assure_path_exists(filepath+"accuracy/"+"dummy.txt")
    assure_path_exists(filepath+"loss/"+"dummy.txt")
    os.chdir(filepath)

    #fileToPlot = "mGS:outputs_run_"+test_point+"_"+str(learning_rate)+"_"+str(my_decay)

    f = open(fileToPlot+'.txt', 'w')

    for layers in [3,4,5]:   # LAYERS
        for neurons in range(2, 101):    # NEURONS
            if args.verbose:
                print "  ==> #LAYERS:", layers, "   #NEURONS:", neurons, " <=="
                print("Starting the training")

            # def getDefinedClassifier(nIn, nOut, compileArgs, neurons, layers, dropout_rate=0, regularizer=0)
            model = getDefinedClassifier(53, 1, compileArgs, neurons, layers)#, args.dropoutRate)

            start = time.time()

            history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,callbacks=callbacks, **trainParams)

            # Time of the training
            training_time = time.time()-start
            if args.verbose:
                print "Training took ", training_time, " seconds"

            name = "L"+str(layers)+"_N"+str(neurons)+"_E"+str(len(history.history['loss']))+"_Bs"+str(args.batchSize)+"_Lr"+str(args.learningRate)+"_Dr"+str(args.dropoutRate)
            if args.verbose:
                print name

            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            pickle.dump(acc, open("accuracy/acc_"+name+".pickle", "wb"))
            pickle.dump(loss, open("loss/loss_"+name+".pickle", "wb"))
            pickle.dump(val_acc, open("accuracy/val_acc_"+name+".pickle", "wb"))
            pickle.dump(val_loss, open("loss/val_loss_"+name+".pickle", "wb"))

            model.save(name+".h5")
            model_json = model.to_json()
            with open(name + ".json", "w") as json_file:
                json_file.write(model_json)
                model.save_weights(name + ".h5")

            if args.verbose:
                print("Getting predictions...")
            dataDev["NN"] = model.predict(XDev) # dataPredict
            dataVal["NN"] = model.predict(XVal)    # valPredict
            dataTest["NN"] = model.predict(XTest)  # testPredict

            if args.verbose:
                print("Getting scores...")
            scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
            scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)
            scoreTest = model.evaluate(XTest, YTest, sample_weight=weightTest, verbose = 0)

            cohen_kappa=cohen_kappa_score(YTest, testPredict.round())

            if args.verbose:
                print "Calculating FOM..."
            sig_dataDev = dataDev[dataDev.category==1];             bkg_dataDev = dataDev[dataDev.category==0]         # separar sig e bkg em dataDev
            sig_dataVal = dataVal[dataVal.category==1];             bkg_dataVal = dataVal[dataVal.category==0]         # separar sig e bkg em dataVal
            sig_dataTest = dataTest[dataTest.category==1];        bkg_dataTest = dataTest[dataTest.category==0]    # separar sig e bkg em dataTest

            tmpSig, tmpBkg = getYields(dataTest)
            sigYield, sigYieldUnc = tmpSig
            bkgYield, bkgYieldUnc = tmpBkg

            fomEvo = []
            fomCut = []

            bkgEff = []
            sigEff = []

            sig_Init = sig_dataTest.EventWeight.sum() * luminosity * 3
            bkg_Init =  bkg_dataTest.EventWeight.sum() * luminosity * 3

            for cut in np.arange(0.0, 0.9999, 0.001):
                sig, bkg = getYields(dataTest, cut=cut, luminosity=luminosity)
                if sig[0] > 0 and bkg[0] > 0:
                    fom, fomUnc = FullFOM(sig, bkg)
                    fomEvo.append(fom)
                    fomCut.append(cut)
                    bkgEff.append(bkg[0]/bkg_Init)
                    sigEff.append(sig[0]/sig_Init)

            max_FOM=0

            if args.verbose:
                print "Maximizing FOM"
            for cv_0 in fomEvo:
                if cv_0>max_FOM:
                    max_FOM=cv_0

            roc_Integral = 0
            for cv_1 in range(0, len(bkgEff)-1):
                roc_Integral=roc_Integral+0.5*(bkgEff[cv_1]-bkgEff[cv_1+1])*(sigEff[cv_1+1]+sigEff[cv_1])

            Eff = zip(bkgEff, sigEff)

            km_value_s = ks_2samp(sig_dataDev["NN"], sig_dataTest["NN"])[1]
            km_value_b = ks_2samp(bkg_dataDev["NN"], bkg_dataTest["NN"])[1]
            km_value = ks_2samp(dataDev["NN"], dataTest["NN"])[1]

            f.write(str(layers)+"\n")
            f.write(str(neurons)+"\n")
            f.write(str(roc_Integral)+"\n")
            f.write(str(km_value_s)+"\n")
            f.write(str(km_value_b)+"\n")
            f.write(str(km_value)+"\n")
            f.write(str(max_FOM)+"\n")

    sys.exit("Done!")
