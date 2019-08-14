'''
Functions used in different files are gathered here to avoid redundance.
'''

import os
import pandas
import numpy as np
from math import log

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

def FOM1(sIn, bIn):
    s, sErr = sIn
    b, bErr = bIn
    fom = s / (b**0.5)
    fomErr = ((sErr / (b**0.5))**2+(bErr*s / (2*(b)**(1.5)) )**2)**0.5
    return (fom, fomErr)

def FOM2(sIn, bIn):
    s, sErr = sIn
    b, bErr = bIn
    fom = s / ((s+b)**0.5)
    fomErr = ((sErr*(2*b + s)/(2*(b + s)**1.5))**2  +  (bErr * s / (2*(b + s)**1.5))**2)**0.5
    return (fom, fomErr)

def FullFOM(sIn, bIn, fValue=0.2):
    s, sErr = sIn
    b, bErr = bIn
    fomErr = 0.0 # Add the computation of the uncertainty later
    fomA = 2*(s+b)*log(((s+b)*(b + (fValue*b)**2))/(b**2 + (s + b) * (fValue*b)**2))
    fomB = log(1 + (s*b*b*fValue*fValue)/(b*(b+(fValue*b)**2)))/(fValue**2)
    fom = (fomA - fomB)**0.5
    return (fom, fomErr)

def getYields(dataTest, cut=0.5, splitFactor=3, luminosity=139500):
    #defines the selected test data
    selectedTest = dataTest[dataTest.NN>cut] # selecionam-se aqueles eventos que foram classificados como signal

    #separates the true positives from false negatives
    selectedSig = selectedTest[selectedTest.category == 1] # true positives
    selectedBkg = selectedTest[selectedTest.category == 0] # false negatives

    sigYield = selectedSig.EventWeight.sum()
    sigYieldUnc = np.sqrt(np.sum(np.square(selectedSig.EventWeight))) # Signal Yield Uncertainty
    bkgYield = selectedBkg.EventWeight.sum()
    bkgYieldUnc = np.sqrt(np.sum(np.square(selectedBkg.EventWeight))) # Background Yield Uncertainty

    sigYield = sigYield * splitFactor * luminosity           # The factor 3 comes from the splitting
    sigYieldUnc = sigYieldUnc * splitFactor * luminosity
    bkgYield = bkgYield * splitFactor * luminosity
    bkgYieldUnc = bkgYieldUnc * splitFactor * luminosity

    return ((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))

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

# Classifiers

from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.optimizers import Adam, Nadam
from keras.regularizers import l1,l2

def getDefinedClassifier(nIn, nOut, compileArgs, neurons, layers, dropout_rate=0, regularizer=0):
  model = Sequential()
  model.add(Dense(neurons, input_dim=nIn, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(regularizer)))
  #model.add(Dropout(dropout_rate))
  for i in range(0,layers-1):
      model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(regularizer)))
      #model.add(Dropout(dropout_rate))
  model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal', kernel_regularizer=l2(regularizer)))
  model.compile(**compileArgs)
  return model
