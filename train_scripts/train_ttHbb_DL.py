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

    df.xsec_ttbarH125toBBbar_2L = 0.5071 * 0.58240 * 0.10608
    df.xsec_ttbar_DL = 831.76 * 0.10608
    df.xsec_ttbb_PowhegOpenLoops_4FS_DL = df.xsec_ttbar_DL * 0.020574086/0.51103793

    sum_weights_Powheg_0_2016 = 8408747.746740535
    sum_weights_TuneCP5_2016 =  490159100.40832114 + 490130836.80188614 + 490168889.96922886 + 488537460.1397215 + 490185354.9467821
    sum_weights_ttbarH125toBBbar_2L_2016 = 39765.99884442469 + 39771.183613112145 + 34058.02468345683 + 34059.37673503202 + 34077.40377754729 + 34102.410952828635 + 34175.41132666297 \
                                            + 37681.14978037304 +39733.110723826794 + 39690.48396339876 + 39711.10705024548 + 39818.30865110878 + 39730.609552379145 + 39776.97717344325 + 34027.00030933365

    sum_weights_Powheg_ext1_2017 = 8247621.656529294
    sum_weights_PSweights_2017 =  648729877.2955494 + 499566945.11193895 + 499858376.58322227 + 499660945.01436853 + 499023147.24216413 + 498016562.91250074
    sum_weights_ttbarH125toBBbar_2L_2017 = 39502.816208213015 + 36790.15734001045 + 38319.73621853188 + 38032.58055960547 + 32257.256983167223 + 33190.72819521076 + 32631.445217642915 \
                                            + 37431.35466692021 + 38840.303006490045 + 38080.21581990127 + 37843.861321620985 + 37915.100467954726 + 38062.14019095547 + 40396.62012507156 + 37572.02308020294

    sum_weights_Powheg_0_2018 = 8181838.0385628445
    sum_weights_PSweights_2018 =  468521499.31969476 + 468527555.2109918 + 468588683.43705857 + 468578106.77498674 + 461346316.60127276
    sum_weights_ttbarH125toBBbar_2L_2018 = 43050.188694801975 + 35200.73227474732 + 30144.02653294645 + 38703.815337536515 + 34571.193831927594 + 33851.30924416335 + 34568.64807854568 \
                                            + 38842.94220910299 + 34094.639342342256 + 35929.31804776111 + 34191.452638792 + 38090.98543298225 + 37630.73484200643 + 40196.88382084637 + 35168.496076980984

    if options.getYear() == 2016:
        lumi = 35.9
        df.sum_weights_Powheg = sum_weights_Powheg_0_2016
        df.sum_weights_PSweights = sum_weights_TuneCP5_2016
        df.sum_weights_ttbarH125toBBbar_2L = sum_weights_ttbarH125toBBbar_2L_2016
    elif options.getYear() == 2017:
        lumi = 41.5
        df.sum_weights_Powheg = sum_weights_Powheg_ext1_2017
        df.sum_weights_PSweights = sum_weights_PSweights_2017
        df.sum_weights_ttbarH125toBBbar_2L = sum_weights_ttbarH125toBBbar_2L_2017
    elif options.getYear() == 2018:
        lumi = 59.7
        df.sum_weights_Powheg = sum_weights_Powheg_0_2018
        df.sum_weights_PSweights = sum_weights_PSweights_2018
        df.sum_weights_ttbarH125toBBbar_2L = sum_weights_ttbarH125toBBbar_2L_2018
    else:
        lumi = 0

    #input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight(), total_weight_expr=' x.weight ')
    input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight(), total_weight_expr=' xsec_ttbarH125toBBbar_2L * x.weight / sum_weights_ttbarH125toBBbar_2L')

    #input_samples.addSample(options.getDefaultName("binary_bkg"), label = "binary_bkg" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

    #input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr='xsec_ttbb_PowhegOpenLoops_4FS_DL * x.weight / sum_weights_Powheg')
    #input_samples.addSample(options.getDefaultName("tt2b"), label = "tt2b" , normalization_weight = options.getNomWeight(), total_weight_expr='xsec_ttbb_PowhegOpenLoops_4FS_DL * x.weight / sum_weights_Powheg')
    #input_samples.addSample(options.getDefaultName("ttb"), label = "ttb"  , normalization_weight = options.getNomWeight(), total_weight_expr='xsec_ttbb_PowhegOpenLoops_4FS_DL * x.weight / sum_weights_Powheg')

    input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb"  , normalization_weight = options.getNomWeight(), total_weight_expr='xsec_ttbb_PowhegOpenLoops_4FS_DL * x.weight / sum_weights_Powheg')
    #input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb"  , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

    input_samples.addSample(options.getDefaultName("ttcc"), label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr=' xsec_ttbar_DL * x.weight / sum_weights_PSweights')
    input_samples.addSample(options.getDefaultName("ttlf"), label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr=' xsec_ttbar_DL  * x.weight / sum_weights_PSweights')
    #input_samples.addSample(options.getDefaultName("ttcc"), label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr=' x.weight ')
    #input_samples.addSample(options.getDefaultName("ttlf"), label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr=' x.weight ')

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

        lumi = lumi,

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

        # plotting
        if options.doPlots():

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

                # plot closure test
                dnn.plot_binaryclosureTest(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork())
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
                    privateWork         = options.isPrivateWork(),
                    bin_range           = [0,1])
            # plot the evaluation metrics
            print("plot_metrics")
            dnn.plot_metrics(privateWork = options.isPrivateWork())

            #dnn.get_gradients(options.isBinary())
            #dnn.predict_event_query()

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
        best = fmin(dnn.hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=25, trials=trials)
        params = space_eval(search_space, best)
        f = open(options.getOutputDir()+"/hyperparamsOptimizations.txt","w")
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
