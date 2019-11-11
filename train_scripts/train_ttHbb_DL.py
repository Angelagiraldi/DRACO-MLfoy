# global imports
from __future__ import print_function
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse


import numpy as np

import hyperas
from hyperopt import Trials, STATUS_OK, tpe

# option handler
import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df


from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
import hyperas
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    x_train = DNN.x_train
    y_train = DNN.y_train
    x_test = DNN.x_test
    y_test = DNN.y_test

    return x_train, y_train, x_test, y_test

def model_to_optimize(X_train, Y_train, X_test, Y_test):

    number_of_input_neurons = X_train.shape[1]
    print(number_of_input_neurons)


    #Model providing function:
    #Create Keras model with double curly brackets dropped-in as needed.
    model = Sequential()
    model.add(Dense(512, input_shape=(number_of_input_neurons,)))
    model.add(Activation({{choice(["selu", "sigmoid"])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['selu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Activation({{choice(['selu', 'sigmoid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(self.data.n_output_neurons))
    model.add(Activation('softmax'))

    adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2,  10**-1])}})
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd
    # compile the model
    model.compile(
        loss        =   'categorical_crossentropy',
        metrics     =   ['accuracy'],
        optimizer   =   optim)
    # save the model
    self.model = model

    # add early stopping if activated
    callbacks = None
    if self.architecture["earlystopping_percentage"] or self.architecture["earlystopping_epochs"]:
        callbacks = [EarlyStopping(monitor         = "loss",
        value           = self.architecture["earlystopping_percentage"],
        min_epochs      = 50,
        stopping_epochs = self.architecture["earlystopping_epochs"],
        verbose         = 1)]

    # train main net
    self.trained_model = self.model.fit(
        x                   = self.data.get_train_data(as_matrix = True),
        y                   = self.data.get_train_labels(),
        batch_size          = {{choice([4,8,16,32,64,128])}},
        epochs              = 10,
        shuffle             = True,
        callbacks           = callbacks,
        validation_split    = 0.25,
        sample_weight       = self.data.get_train_weights())

    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(self.trained_model.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)

    #Creat a python dictionary with two customary keys:
    #    - loss: Specify a numeric evaluation metric to be minimized (e.g. 'loss': -accuracy, the negative of accuracy.
    #            That's because under the hood hyperopt will always minimize whatever metric you provide
    #   - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    #The last one is optional, though recommended, namely:
    #    - model: specify the model just created so that we can later use it again.
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}



#### main
if __name__ == '__main__':

    options.initArguments()

    # load samples
    input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

    # during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
    # thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
    input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

    #input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    #input_samples.addSample(options.getDefaultName("tt2b"), label = "tt2b" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    #input_samples.addSample(options.getDefaultName("ttb"), label = "ttb"  , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    input_samples.addSample(options.getDefaultName("ttbar"), label = "ttbar"  , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

    input_samples.addSample(options.getDefaultName("ttcc"), label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    input_samples.addSample(options.getDefaultName("ttlf"), label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')


    if options.isBinary():
       input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

    category_cutString_dict = {

        '3j_'+  '2t': '(N_jets == 3) & (N_btags == 2)',
        '3j_'+  '3t': '(N_jets == 3) & (N_btags == 3)',
      'ge3j_'+'ge3t': '(N_jets >= 3) & (N_btags >= 3)',
      'ge4j_'+  '2t': '(N_jets >= 4) & (N_btags == 2)',
      'ge4j_'+  '3t': '(N_jets >= 4) & (N_btags == 3)',
      'ge4j_'+'ge4t': '(N_jets >= 4) & (N_btags >= 4)',

      'ge4j_'+'ge3t': '(N_jets >= 4) & (N_btags >= 3)',
    }

    category_label_dict = {

        '3j_'+  '2t': 'N_jets = 3, N_btags = 2',
        '3j_'+  '3t': 'N_jets = 3, N_btags = 3',
      'ge3j_'+'ge3t': 'N_jets \\geq 3, N_btags \\geq 3',
      'ge4j_'+  '2t': 'N_jets \\geq 4, N_btags = 2',
      'ge4j_'+  '3t': 'N_jets \\geq 4, N_btags = 3',
      'ge4j_'+'ge4t': 'N_jets \\geq 4, N_btags \\geq 4',

      'ge4j_'+'ge3t': 'N_jets \\geq 4, N_btags \\geq 3',
    }

    # initializing DNN training class

    dnn = DNN.DNN(

        save_path       =  options.getOutputDir(),
        input_samples   = input_samples,

        category_name   = options.getCategory(),

        category_cutString = category_cutString_dict[options.getCategory()],
        category_label     = category_label_dict[options.getCategory()],

        train_variables = options.getTrainVariables(),

        # number of epochs
        train_epochs    = options.getTrainEpochs(),
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ['acc'],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = options.getTestPercentage(),
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.doBalanceSamples(),
        evenSel         = options.doEvenSelection(),
        norm_variables  = options.doNormVariables()
    )


    if not options.doHyperparametersOptimization():
        # build DNN model
        dnn.build_model(options.getNetConfig())

        # perform the training
        dnn.train_model()

        # evalute the trained model
        dnn.eval_model()

        # save information
        dnn.save_model(sys.argv, filedir, options.getNetConfigName())

        # save configurations of variables for plotscript
        #dnn.variables_configuration()

        # save and print variable ranking according to the input layer weights
        dnn.get_input_weights()

        # save and print variable ranking according to all layer weights
        dnn.get_weights()

        dnn.get_gradients(options.isBinary())


        # plotting
        if options.doPlots():
            # plot the evaluation metrics
            dnn.plot_metrics(privateWork = options.isPrivateWork())

            if options.isBinary():
                # plot output node
                bin_range = options.getBinaryBinRange()
                dnn.plot_binaryOutput(
                    log         = options.doLogPlots(),
                    privateWork = options.isPrivateWork(),
                    printROC    = options.doPrintROC(),
                    bin_range   = bin_range,
                    nbins       = 20,
                    name        = options.getName())

            else:
                # plot the confusion matrix
                dnn.plot_confusionMatrix(
                    privateWork = options.isPrivateWork(),
                    printROC    = options.doPrintROC())

                # plot the output discriminators
                dnn.plot_discriminators(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork(),
                    printROC            = options.doPrintROC(),
                    sigScale            = options.getSignalScale())

                # plot the output nodes
                dnn.plot_outputNodes(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork(),
                    printROC            = options.doPrintROC(),
                    sigScale            = options.getSignalScale())

                # plot event yields
                dnn.plot_eventYields(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork(),
                    sigScale            = options.getSignalScale())

                # plot closure test
                dnn.plot_closureTest(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork())
    else:
        # Minimize the "loss" to define the most performing set of hyper-parameters
        #''' Return the chosen hyper-parameter for the best performing model '''
        dnn.get_data()

        best_run, best_model = hyperas.optim.minimize(model_to_optimize, data, tpe.suggest, 20, Trials())
        X_train, Y_train, X_test, Y_test = data()
        print("Evalutation of best performing model:")
        print(best_model.evaluate(X_test, Y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)
