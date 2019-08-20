import os
import matplotlib.pyplot as plt
import numpy as np

import localConfig as cfg

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-a', '--hyperParam', type=str, required=True, help='HyperParameter to study')
    args = parser.parse_args()

    hyperParam = args.hyperParam
    #learning_rate = str(float(model_name[model_name.find("Lr")+2:model_name.find("_D")]))
    filepath = cfg.lgbk + 'gridSearch/' + hyperParam + '/'
    os.chdir(filepath)
    name = "ROC_" + hyperParam

    f = open(name + '.txt', 'r')

    layer = []
    neurons = []
    roc_AUC = []
    ks = []
    ks_s = []
    ks_b = []
    FOM = []
    line_index=0

    for line in f:
        if line_index%7==0:
            layer.append(float(line,))
        if line_index%7==1:
            neurons.append(float(line,))
        if line_index%7==2:
            roc_AUC.append(float(line,))
        if line_index%7==3:
            ks_s.append(float(line,))
        if line_index%7==4:
            ks_b.append(float(line,))
        if line_index%7==5:
            ks.append(float(line,))
        if line_index%7==6:
            FOM.append(float(line,))
        line_index=line_index+1

    layers_legend = ["3 layers"]
    nLayers = len(layers_legend)

    plt.figure(figsize=(7,6))
    plt.xlabel("Number of Neurons")
    plt.ylabel('Roc AUC')
    plt.suptitle("Roc curve integral for several configurations of Neural Nets", fontsize=13, fontweight='bold')
    #plt.title("Learning rate: {0}\nDecay: {1}".format(learning_rate, my_decay), fontsize=10)

    neurons = range(2,61)
    lim = len(neurons)/nLayers
    print "lim=", lim
    for i in range(0,nLayers):
        print "i=", i
        plt.plot(neurons[i*lim:(i+1)*lim], roc_AUC[i*lim:(i+1)*lim])
    plt.grid()
    plt.legend(layers_legend, loc='best')
#   plt.savefig(name+'.pdf')
    plt.savefig('ROC_'+hyperParam+'.pdf')
    plt.show()
