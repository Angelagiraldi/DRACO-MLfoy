# global imports
from __future__ import print_function
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse

import numpy as np


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

#### main
if __name__ == '__main__':

    options.initArguments()

    # load samples
    input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

    # during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
    # thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
    input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

    input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    input_samples.addSample(options.getDefaultName("tt2b"), label = "tt2b" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    input_samples.addSample(options.getDefaultName("ttb"), label = "ttb"  , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
    #input_samples.addSample(options.getDefaultName("ttbar"), label = "ttbar"  , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

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
        #dnn.get_input_weights()

        # save and print variable ranking according to all layer weights
        #dnn.get_weights()

        #dnn.get_gradients(options.isBinary())
        dnn.predict_event_query()


        # plotting
        if options.doPlots():
            # plot the evaluation metrics
            print("plot_metrics")
            #dnn.plot_metrics(privateWork = options.isPrivateWork())

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
        import hyperopt as hp
        from hyperopt import fmin, STATUS_OK, tpe, space_eval, Trials
        from hyperopt.pyll import scope
        from hyperas.distributions import choice, uniform, loguniform, quniform

        opt_search_space = choice('name',
                                  [
                                      {'name': 'adam',
                                       'learning_rate': loguniform('learning_rate_adam', -10, 0),
                                       #'beta_1': loguniform('beta_1_adam', -10, -1),# Note the name of the label to avoid duplicates
                                       #'beta_2': loguniform('beta_2_adam', -10, -1),
                                      },
                                      {'name': 'sgd',
                                       'learning_rate': loguniform('learning_rate_sgd', -15, 0), # Note the name of the label to avoid duplicates
                                       #'momentum': uniform('momentum_sgd', 0, 1.0),
                                      },
                                      {'name': 'RMSprop',
                                       'learning_rate': loguniform('learning_rate_rmsprop', -15, 0), # Note the name of the label to avoid duplicates
                                      },
                                      {'name': 'Adagrad',
                                       'learning_rate': loguniform('learning_rate_adagrad', -15, 0), # Note the name of the label to avoid duplicates
                                      },
                                      {'name': 'Adadelta',
                                       'learning_rate': loguniform('learning_rate_adadelta', -10, 0), # Note the name of the label to avoid duplicates
                                      },
                                      {'name': 'Adamax',
                                       'learning_rate': loguniform('learning_rate_adamax', -10, 0), # Note the name of the label to avoid duplicates
                                       #'beta_1': loguniform('beta_1_adamax', -10, 0),# Note the name of the label to avoid duplicates
                                       #'beta_2': loguniform('beta_2_adamax', -10, 0),
                                      }
                                      ])
        fourth_layer_search_space = choice('four_layer',
                                    [
                                      {
                                        'include': False,
                                      },
                                      {
                                        'include': True,
                                        'layer_size_4': choice('layer_size_4', [ 32, 64, 128,256,512, 1024]),
                                      }

                                    ])
        @scope.define
        def power_of_two(a):
             return 2.0 ** a

        search_space = {
          'layer_size_1'        : choice('layer_size_1', [ 32, 64, 128, 256, 512, 1024]),
          'layer_size_2'        : choice('layer_size_2', [ 32, 64, 128, 256, 512, 1024]),
          'layer_size_3'        : choice('layer_size_3', [ 32, 64, 128, 256, 512, 1024]),
          'four_layer'          : fourth_layer_search_space,
          'dropout'             : uniform('dropout', 0, 1),
          'batch_size'          : scope.power_of_two(quniform('batch_size', 3, 12, q=1)),
          'optimizer'           : opt_search_space,
          'l2_regularizer'      : loguniform('l2_regularizer', -10,-1)
        }

        trials = Trials()
        best = fmin(dnn.hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=20, trials=trials)
        params = space_eval(search_space, best)
        f = open(options.getOutputDir()+"/.txt","w")
        f.write( str(params) )
        f.close()
        print(params)

        if options.doPlots():
            import matplotlib.pyplot as plt
            plt.figure()
            xs = [t['tid'] for t in trials.trials]
            ys = [-t['result']['loss'] for t in trials.trials]
            plt.xlim(xs[0]-1, xs[-1]+1)
            plt.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
            plt.xlabel('Iteration', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.savefig(options.getOutputDir()+"/Accuracy.png", bbox_inches='tight')
            plt.savefig(options.getOutputDir()+"/Accuracy.pdf", bbox_inches='tight')
            plt.close()
            # Some additional visualization
            parameters = search_space.keys()
            cmap = plt.cm.Dark2

            for t in trials.trials:
                for i, name in enumerate(t['misc']['vals']):
                    plt.figure()

                    xs = np.array([t['misc']['vals'][name]]).ravel()
                    if xs.size == 0: continue
                    ys = [-t['result']['loss']]
                    zs = [t['tid'] for t in trials.trials]
                    xs, ys, zs = zip(*sorted(zip(xs, ys, zs)))
                    ys = np.array(ys)
                    plt.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/len(parameters)))
                    plt.savefig(options.getOutputDir()+"/"+name+"_vs_Accuracy.png", bbox_inches='tight')
                    plt.savefig(options.getOutputDir()+"/"+name+"_vs_Accuracy.pdf", bbox_inches='tight')
                    plt.close()

                    plt.figure()
                    plt.scatter(zs, xs, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/len(parameters)))
                    plt.savefig(options.getOutputDir()+"/"+name+"_vs_Iteration.png", bbox_inches='tight')
                    plt.savefig(options.getOutputDir()+"/"+name+"_vs_Iteration.pdf", bbox_inches='tight')
                    plt.close()
