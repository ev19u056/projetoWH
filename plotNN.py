'''
Test the Neural Network
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import pandas
import numpy as np
import localConfig as cfg

if __name__ == "__main__":
    import sys
    import argparse

    ## Input arguments. Pay speciall attention to the required ones.
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-f', '--file',type=str, required=True, help='File name')
    parser.add_argument('-a', '--allPlots', action='store_true', help='Wether to plot all graphs')
    parser.add_argument('-b', '--loss', action='store_true', help='Loss plot')
    parser.add_argument('-c', '--accuracy', action='store_true', help='Accuracy plot')
    parser.add_argument('-o', '--overtrainingCheck', action='store_true', help='Wether there was overtraining')
    parser.add_argument('-p', '--prediction', action='store_true', help='Predictions plot')
    parser.add_argument('-e', '--efficiencyAndFOM', action='store_true', help='Plot efficiency and FOM')
    parser.add_argument('-r', '--areaUnderROC', action='store_true', help='Area under ROC plot')
    parser.add_argument('-w', '--weights', action='store_true', help='Plot neural network weights')
    parser.add_argument('-z', '--structure', action='store_true', help='Plot neural network structure')
#    parser.add_argument('-l', '--layers', type=int, help='Number of layers')
#    parser.add_argument('-n', '--neurons', type=int, help='Number of neurons per layer')
#    parser.add_argument('-x', '--gridSearch', action='store_true', help='File on grid search')
#    parser.add_argument('-s', '--singleNN', action='store_true', help='Whether this NN is stored in the Searches or SingleNN folder')
    parser.add_argument('-u', '--runNum', type=int, help='Run number')
    parser.add_argument('-k', '--local', action='store_true', help='Local file')
    parser.add_argument('-d', '--preview', action='store_true', help='Preview plots')
#    parser.add_argument('-bk', '--bk', action='store_true', help='Whether or not you choose to load Zinv background samples or only W+jets and TTpow')

#python plotNN.py -v -f Model_Ver_3 -b -c -o -p -r -s

#    parser.add_argument('-b', '--batch', action='store_true', help='Whether this is a batch job, if it is, no interactive questions will be asked and answers will be assumed')
#    parser.add_argument('-p', '--dropoutRate', type=float, default=0, help='Dropout Rate')
#    parser.add_argument('-dc', '--decay', type=float, default=0, help='Learning rate decay')

    from prepareData import *
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from keras.models import model_from_json
    from commonFunctions import assure_path_exists

    if args.file != None:
        model_name = args.file
        #lgbk = "/home/t3atlas/ev19u056/projetoWH/"
        filepath = cfg.lgbk + "test/" + model_name
        loss_path = filepath + "/loss/"
        acc_path = filepath + "/accuracy/"
    else:
        print "ERROR: Missing filename"
        quit()

    os.chdir(filepath+"/")
    plots_path = filepath+"/plots_"+model_name+"/"
    assure_path_exists(plots_path)

    if args.verbose:
        print "Loading Model ..."

    ## Load your trainned model
    with open(model_name+'.json', 'r') as json_file:
      loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name+".h5")
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

    if args.verbose:
        print("Getting predictions ...")

    devPredict = model.predict(XDev)
    valPredict = model.predict(XVal)
    testPredict = model.predict(XTest)

    if args.verbose:
        print("Getting scores ...")

    scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)
    scoreTest = model.evaluate(XTest, YTest, sample_weight=weightTest, verbose = 0)

    if args.verbose:
        print "Calculating parameters ..."

    dataDev["NN"] = devPredict
    dataVal["NN"] = valPredict
    dataTest["NN"] = testPredict

    sig_dataDev = dataDev[dataDev.category==1];     bkg_dataDev = dataDev[dataDev.category==0]
    sig_dataVal = dataVal[dataVal.category==1];    bkg_dataVal = dataVal[dataVal.category==0]
    sig_dataTest = dataTest[dataTest.category==1];    bkg_dataTest = dataTest[YTest.category==0]

    if args.allPlots:
        args.loss = True
        args.accuracy = True
        args.overtrainingCheck = True
        args.prediction = True
        args.efficiencyAndFOM = True
        args.areaUnderROC = True
        args.weights = True
        args.stucture = True

    if args.loss:
        import pickle
        loss = pickle.load(open(loss_path+"loss_"+model_name+".pickle", "rb"))
        val_loss = pickle.load(open(loss_path+"val_loss_"+model_name+".pickle", "rb"))
        if args.verbose:
            print "val_loss = ", str(val_loss[-1])
            print "loss = ", str(loss[-1])
            print "dloss = ", str(val_loss[-1]-loss[-1])

        plt.plot(loss)
        plt.plot(val_loss)
        #plt.ylimit(0.0000012 , 0.0000006)
        plt.grid()
        plt.title('Model loss')
        plt.ylabel('Loss')
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='best')
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(plots_path+'loss_'+model_name+'.pdf')
        if args.preview:
            plt.show()
        plt.close()

    if args.accuracy:
        import pickle
        acc = pickle.load(open(acc_path+"acc_"+model_name+".pickle", "rb"))
        val_acc = pickle.load(open(acc_path+"val_acc_"+model_name+".pickle", "rb"))
        if args.verbose:
            print "val_acc = " + str(val_acc[-1])
        plt.plot(acc)
        plt.plot(val_acc)
        #plt.ylimit(0.7 ,1)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='best')
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(plots_path+'acc_'+model_name+'.pdf')
        if args.preview:
            plt.show()
        plt.close()

    # --- Over Training Check --- #
    # Ver com mais detalhes
    # Add sig_dataTest, bkg_dataTest
    if args.overtrainingCheck:
        from scipy.stats import ks_2samp
        from sklearn.metrics import cohen_kappa_score

        cohen_kappa=cohen_kappa_score(YVal, valPredict.round())
        km_value=ks_2samp((sig_dataDev["NN"].append(bkg_dataDev["NN"])),(sig_dataVal["NN"].append(bkg_dataVal["NN"])))

        if args.verbose:
            print "Cohen Kappa score:", cohen_kappa
            print "KS test statistic:", km_value[0]
            print "KS test p-value:", km_value[1]

        #plt.yscale('log')
        plt.hist(sig_dataDev["NN"], 50, facecolor='blue', alpha=0.7, normed=1)#, weights=sig_dataDev["EventWeight"])
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', alpha=0.7, normed=1)#, weights=bkg_dataDev["EventWeight"])
        plt.hist(sig_dataVal["NN"], 50, color='blue', alpha=1, normed=1, histtype="step")#, weights=sig_dataVal["EventWeight"])
        plt.hist(bkg_dataVal["NN"], 50, color='red', alpha=1, normed=1, histtype="step")#,weights=bkg_dataVal["EventWeight"])
        plt.xlabel('NN output')
        plt.suptitle("MVA overtraining check for classifier: NN", fontsize=13, fontweight='bold')
        plt.title("Cohen's kappa: {0}\nKolmogorov Smirnov test: {1}".format(cohen_kappa, km_value[1]), fontsize=10)
        plt.legend(['Signal (Test sample)', 'Background (Test sample)', 'Signal (Train sample)', 'Background (Train sample)'], loc='best')
        plt.savefig(plots_path+'hist_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

    # --- Predictions plot --- #
    if args.prediction:
        both_dataDev = bkg_dataDev.append(sig_dataDev)
        plt.figure(figsize=(7,6))
        plt.xlabel('NN output')
        plt.title("Number of Events")
        #plt.yscale('log', nonposy='clip')
        plt.legend(['Background + Signal (test sample)', 'Background (test sample)'], loc="best" ) # legend does not appear !!!
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', weights=bkg_dataDev["EventWeight"])
        plt.hist(both_dataDev["NN"], 50, color="blue", histtype="step", weights=both_dataDev["EventWeight"])
        plt.savefig(plots_path+'pred_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()


    # PLOTTING FOM AND Efficiency
    if args.efficiencyAndFOM:
        from commonFunctions import FOM1, FOM2, FullFOM, getYields

        fomEvo = []
        fomCut = []

        bkgEff = []
        sigEff = []

        sig_Init = dataVal[dataVal.category == 1].weight.sum()# * 35866 * 2
        bkg_Init = dataVal[dataVal.category == 0].weight.sum()# * 35866 * 2

        for cut in np.arange(0.0, 0.9999, 0.001):
            sig, bkg = getYields(dataVal, cut=cut, luminosity=luminosity)
            if sig[0] > 0 and bkg[0] > 0:
                fom, fomUnc = FullFOM(sig, bkg)
                fomEvo.append(fom)
                fomCut.append(cut)
                bkgEff.append(bkg[0]/bkg_Init)
                sigEff.append(sig[0]/sig_Init)

        max_FOM=0

        for k in fomEvo:
            if k>max_FOM:
                max_FOM=k

        #SAVE VALUES OF FOM EVO AND CUT TO DO A FOM SUMMARY
        f= open(plots_path+"FOM_evo_data.txt","w+")

        f.write("\n".join(map(str,fomEvo)))
        f.close()

        f= open(plots_path+"FOM_cut_data.txt","w+")

        f.write("\n".join(map(str,fomCut)))
        f.close()

        quite()
        Eff = zip(bkgEff, sigEff)

        if args.verbose:
            print "Maximized FOM:", max_FOM
            print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]

            tmpSig, tmpBkg = getYields(dataVal)
            sigYield, sigYieldUnc = tmpSig
            bkgYield, bkgYieldUnc = tmpBkg

            selectedVal = dataVal[dataVal.NN>fomCut[fomEvo.index(max_FOM)]]
            selectedSig = selectedVal[selectedVal.category == 1]
            selectedBkg = selectedVal[selectedVal.category == 0]
            sigYield = selectedSig.weight.sum()
            bkgYield = selectedBkg.weight.sum()
            sigYield = sigYield * luminosity * 2          #The factor 2 comes from the splitting
            bkgYield = bkgYield * luminosity * 2

            print "Selected events left after cut @", fomCut[fomEvo.index(max_FOM)]
            print "   Number of selected Signal Events:", len(selectedSig)
            print "   Number of selected Background Events:", len(selectedBkg)
            print "   Sig Yield", sigYield
            print "   Bkg Yield", bkgYield

        plt.figure(figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(211)
        plt.plot(fomCut, fomEvo, linewidth = 0.5)
        plt.title("FOM")
        plt.ylabel("FOM")
        plt.xlabel("ND")
        plt.legend(["Max. FOM: {0}".format(max_FOM)], loc='best')
        plt.grid()

        plt.subplot(212)
        plt.semilogy(fomCut, Eff , linewidth = 0.5)
        plt.axvspan(fomCut[fomEvo.index(max_FOM)], 1, facecolor='#2ca02c', alpha=0.3)
        #plt.axvline(x=fomCut[fomEvo.index(max_FOM)], ymin=0, ymax=1)
        plt.title("Efficiency")
        plt.ylabel("Eff")
        plt.xlabel("ND")
        plt.legend(['Background', 'Signal'], loc='best')
        plt.grid()

        plt.savefig(plots_path+'FOM_EFF_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

        #SAME BUT ZOOMED IN , NO LOG yscale
        plt.figure(figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(211)
        plt.plot(fomCut, fomEvo, linewidth = 0.3)
        plt.xlim(0.95 ,1.01)
        plt.xticks(np.arange(0.95 , 1.01, step = 0.01))
        plt.title("FOM")
        plt.ylabel("FOM")
        plt.xlabel("ND")
        plt.legend(["Max. FOM: {0}".format(max_FOM)], loc='best')
        plt.grid()

        plt.subplot(212)
        plt.plot(fomCut, Eff , linewidth = 0.3)
        plt.xlim(0.95 , 1.01)
        plt.xticks(np.arange(0.95 , 1.01, step = 0.01))
        plt.axvspan(fomCut[fomEvo.index(max_FOM)], 1, facecolor='#2ca02c', alpha=0.3)
        #plt.axvline(x=fomCut[fomEvo.index(max_FOM)], ymin=0, ymax=1)
        plt.title("Efficiency")
        plt.ylabel("Eff")
        plt.xlabel("ND")
        plt.legend(['Background', 'Signal'], loc='best')
        plt.grid()

        plt.savefig(plots_path+'FOM_EFF_zoomed_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

    # PLOTTING the ROC function
    if args.areaUnderROC:
        from sklearn.metrics import roc_auc_score, roc_curve

        roc_integralDev = roc_auc_score(dataDev.category, dataDev.NN)
        roc_integralVal = roc_auc_score(dataVal.category, dataVal.NN)
        fprDev, tprDev, _Dev = roc_curve(dataDev.category, dataDev.NN)
        fprVal, tprVal, _Val = roc_curve(dataVal.category, dataVal.NN)
        if args.verbose:
            print "ROC Curve IntegralDev:", roc_integralDev
            print "ROC Curve IntegralVal:", roc_integralVal

        plt.plot(fprDev, tprDev, '--')
        plt.plot(fprVal, tprVal, linewidth=0.5)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        rocLegend = ["Dev Integral: {0}".format(roc_integralDev),"Val Integral: {0}".format(roc_integralVal)]
        plt.legend(rocLegend, loc='best')
        plt.savefig(plots_path+'ROC_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

    '''        #PLOTTING ROCK ZOOMED
        plt.plot(fprDev, tprDev, '--')
        plt.plot(fprVal, tprVal, linewidth=0.5)
        plt.xlim(0 , 0.3)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        rocLegend = ["Dev Integral: {0}".format(roc_integralDev),"Val Integral: {0}".format(roc_integralVal)]
        plt.legend(rocLegend, loc='best')
        plt.savefig(plots_path+'ROC_zoomed_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()
    '''
    if args.weights:
        import math
        from matplotlib.colors import LinearSegmentedColormap

        #Color maps
        cdict = {'red':   ((0.0, 0.97, 0.97),
                           (0.25, 0.0, 0.0),
                           (0.75, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.25, 0.25),
                           (0.25, 0.15, 0.15),
                           (0.75, 0.39, 0.39),
                           (1.0, 0.78, 0.78)),

                 'blue':  ((0.0, 1.0, 1.0),
                           (0.25, 0.65, 0.65),
                           (0.75, 0.02, 0.02),
                           (1.0, 0.0, 0.0))
                }
        myColor = LinearSegmentedColormap('myColorMap', cdict)

        nLayers = 0
        for layer in model.layers:
            if len(layer.get_weights()) == 0:
                continue
            nLayers+=1

        maxWeights = 0

        figure = plt.figure()
        figure.suptitle("Weights", fontsize=12)

        i=1
        nRow=2
        nCol=3

        if nLayers < 5:
            nRow = 2.0
            nCol = 2

        elif nLayers < 10:
            nRow = math.ceil(nLayers / 3)
            nCol = 3

        else:
            nRow = math.ceil(nLayers / 4)
            nCol = 4

        for layer in model.layers:
            if len(layer.get_weights()) == 0:
                continue
            ax = figure.add_subplot(nRow, nCol,i)

            im = plt.imshow(layer.get_weights()[0], interpolation="none", vmin=-2, vmax=2, cmap=myColor)
            plt.title(layer.name, fontsize=10)
            plt.xlabel("Neuron", fontsize=9)
            plt.ylabel("Input", fontsize=9)
            plt.colorbar(im, use_gridspec=True)

            i+=1

        plt.tight_layout()
        plt.savefig(plots_path+'Weights_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()

    if args.structure:
        from keras.utils import plot_model
        plot_model(model, to_file='model.pdf', show_shapes=True)
