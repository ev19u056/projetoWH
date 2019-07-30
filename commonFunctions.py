'''
Functions used in different files are gathered here to avoid redundance.
'''

import os
import pandas
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, AlphaDropout
#from keras.optimizers import Adam, Nadam
#from keras.regularizers import l1,l2
#from math import log

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Plots
import matplotlib.pyplot as plt
import sys
import pickle

def plotter(path,Ylabel,Title):
    open_= open(path,'rb')
    plot_= pickle.load(open_)
    plt.plot(plot_)
    plt.ylabel(Ylabel)
    plt.xlabel("Epochs")
    plt.legend()
    plt.title(Title)
