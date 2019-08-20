# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback
from keras import backend

from prepareData import dataLoader
import localConfig as cfg

import argparse
import sys

## Input arguments. Pay speciall attention to the required ones.
parser = argparse.ArgumentParser(description='Process the command line options')
parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration number i')
parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')

args = parser.parse_args()
iteration = args.iteration

verbose = 0
if args.verbose:
    verbose = 1

# Naming the Model
name ="GS_Model_Ver_"+str(iteration)

# lgbk = "/home/t3atlas/ev19u056/projetoWH/"
testpath = cfg.lgbk+"GridSearch/"
filepath = cfg.lgbk+"GridSearch/"+name+"/"

if os.path.exists(filepath) == False:
    os.mkdir(filepath)

# Function to create model, required for KerasClassifier
def create_model():
    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    neurons = 71
    layers = 4
    model = Sequential()
    model.add(Dense(neurons, input_dim=53, kernel_initializer='he_normal', activation='relu'))
    #model.add(Dropout(dropout_rate))
    for i in range(0,layers-1):
        model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu'))
        #model.add(Dropout(dropout_rate))
    model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(**compileArgs)
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataDev, dataVal, dataTest, XDev, YDev, weightDev, XVal, YVal, weightVal, XTest, YTest, weightTest = dataLoader(filepath+"/", model_name, fraction)

# create model
lrm = LearningRateMonitor()
callbacks = [EarlyStopping(patience=15, verbose=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=True, cooldown=1, min_lr=0), # argument min_delta is not supported
                ModelCheckpoint(filepath+name+".h5", save_best_only=True, save_weights_only=True), lrm]
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
# --- tune BATCH SIZE and NUMBER OF EPOCHS --- #
batch_size = [100, 200, 500, 1000, 2000, 3000, 10000, 20000]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

if args.verbose:
    print("Dir "+filepath+" created.")
    print("Starting the training")
    start = time.time()
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
