'''
Train the Neural Network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.optimizers import Adam, Nadam
import time
import keras
import pandas
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, AlphaDropout
from sklearn.metrics import confusion_matrix, cohen_kappa_score

''' copy to your folder and adapt commonFunctions.py'''
#from commonFunctions import getYields, FullFOM, myClassifier, gridClassifier, getDefinedClassifier, assure_path_exists
#from scipy.stats import ks_2samp

''' copy to your folder and adapt localConfig.py '''
#import localConfig as cfg
import pickle

''' copy to your folder and adapt prepareDATA.py '''
#from prepareDATA import *

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
