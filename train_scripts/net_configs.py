from keras import optimizers

config_dict = {}

config_dict["example_config"] = {
        "layers":                   [200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "batch_size":               5000,
        "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["test_config"] = {
        "layers":                   [1000,1000,200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.,
        "L2_Norm":                  0.,
        "batch_size":               5000,
        "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttZ_2018_final"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.4,
        "L2_Norm":                  1e-4,
        "batch_size":               200,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_2017"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["Legacy_ttH_2017"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_2017_baseline"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     100,
        }

config_dict["legacy_2018"] = {
        "layers":                   [200,100,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.20,
        "L2_Norm":                  1e-5,
        "batch_size":               50000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["dnn_config"] = {
        "layers":                   [20],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.1,
        "L2_Norm":                  0.,
        "batch_size":               2000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }


config_dict["ttH_2017_DL"] = {
        "layers":                   [200,100,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.03,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-4,
        "batch_size":               32,
        "optimizer":                optimizers.Adamax(lr=0.0021180073206877614),
        "activation_function":      "elu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }


config_dict["ttH_2017_DL_4nodes_ge4jge4t"] = {
        "layers":                   [256,16,64],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.02485037521340286,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0008586865422068512,
        "batch_size":               64,
        "optimizer":                optimizers.Adamax(lr=0.00015699397972759966, beta_1=0.04512761581882773, beta_2=0.006377085081329351),
        "activation_function":      "selu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4jge4t_v3"] = {
        "layers":                   [64,32,512,32],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.013438304708440652,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0001927657682628117,
        "batch_size":               16,
        "optimizer":                optimizers.Adamax(lr=0.0021180073206877614),
        "activation_function":      "selu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4jge4t_v2"] = {
        "layers":                   [256,64,512,64],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5712878861549381,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.011876838797264847,
        "batch_size":               1024,
        "optimizer":                optimizers.Adadelta(lr=0.02838587504529099),
        "activation_function":      "selu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4j3t"] = {
        "layers":                   [16,8,256],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.34212950879705056,
        "L1_Norm":                  0.,
        "L2_Norm":                  9.954028818206712e-05,
        "batch_size":               32,
        "optimizer":                optimizers.Adamax(lr=0.0016781633980941539, beta_1=0.07369659899517526, beta_2=0.006847469786029943),
        "activation_function":      "selu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4jge3t"] = {
        "layers":                   [512,1024,256,1024],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.03465780173759353,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0006205881963424858,
        "batch_size":               16,
        "optimizer":                optimizers.Adadelta(lr=0.03286515184928511),
        "activation_function":      "selu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4jge3t_v2"] = {
        "layers":                   [256,64,512,64],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.013438304708440652,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0001927657682628117,
        "batch_size":               1024,
        "optimizer":                optimizers.Adamax(lr=0.0021180073206877614),
        "activation_function":      "elu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4jge3t_v3"] = {
        "layers":                   [256,64,512,64],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.013438304708440652,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0001927657682628117,
        "batch_size":               1024,
        "optimizer":                optimizers.Adadelta(lr=0.03286515184928511),
        "activation_function":      "leakyrelu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_4nodes_ge4jge3t_v4"] = {
        "layers":                   [128,1024,32],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5772135524951899,
        "L1_Norm":                  0.,
        "L2_Norm":                  5.084306953295207e-05,
        "batch_size":               256,
        "optimizer":                optimizers.Adadelta(lr=0.6659221416024017),
        "activation_function":      "relu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }


config_dict["ttH_2017_DL_binary0_ge4jge4t"] = {
        "layers":                   [128,512,256,51],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.6868387033050857,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.00923354032026579,
        "batch_size":               2,
        "optimizer":                optimizers.Adadelta(lr=0.017662112980417394),
        "activation_function":      "selu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_binary0_ge4jge4t_v2"] = {
        "layers":                   [128,256,256,128],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.10503711070235566,
        "L1_Norm":                  0.,
        "L2_Norm":                  7.537493666229303e-05,
        "batch_size":               64,
        "optimizer":                optimizers.RMSprop(lr= 4.306361720403694e-06),
        "activation_function":      "selu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }


config_dict["ttH_2017_DL_binary0_ge4j3t"] = {
        "layers":                   [256,64,8],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.3357936928188098,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0004182492246455418,
        "batch_size":               64,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "selu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }


config_dict["ttH_2017_DL_binary-1_ge4jge4t"] = {
        "layers":                   [512,128,32,512],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.10064369953731123,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0005763555161387924,
        "batch_size":               128,
        "optimizer":                optimizers.Adadelta(lr=0.2688292513163337),
        "activation_function":      "selu",
        "output_activation":        "tanh",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_binary-1_ge4j3t"] = {
        "layers":                   [32,16,128,16],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.26701418366108753,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.00014636065527099827,
        "batch_size":               512,
        "optimizer":                optimizers.Adam(lr=0.008633165563976454, beta_1=4.743065068433159e-05, beta_2=0.00015771267489309254),
        "activation_function":      "selu",
        "output_activation":        "tanh",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }
