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
        "earlystopping_percentage": 0.02,
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


config_dict["ttH_DL_baseline_4nodes"] = {
        "layers":                   [200,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.3,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-5,
        "batch_size":               128,
        "optimizer":                optimizers.Adam(),
        "activation_function":      "relu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }


config_dict["ttH_DL_baseline_binary"] = {
        "layers":                   [100,50,50],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.2,
        "L2_Norm":                  1e-4,
        "batch_size":               4096,
        "optimizer":                optimizers.Adamax(),
        "activation_function":      "elu",
        "output_activation":        "tanh",
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
        "layers":                   [128,512,512],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.55161589329477,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.000984937846098407,
        "batch_size":               4096,
        "optimizer":                optimizers.RMSprop(lr=0.008099908285974919),
        "activation_function":      "relu",
        "output_activation":        "softmax",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["ttH_2017_DL_binary0_ge4jge4t"] = {
        "layers":                   [512,32,256,256],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.06091399436800815,
        "L1_Norm":                  0.,
        "L2_Norm":                  9.726351240316057e-05,
        "batch_size":               256,
        "optimizer":                optimizers.Adamax(lr=0.018185321261173837),
        "activation_function":      "relu",
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
        "layers":                   [512,256,64,32],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.2030704494373083,
        "L1_Norm":                  0.,
        "L2_Norm":                  0.0006721936112696179,
        "batch_size":               4096,
        "optimizer":                optimizers.Adamax(lr=0.014030904693772628),
        "activation_function":      "relu",
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


config_dict["ttH_DL_baseline_binary"] = {
        "layers":                   [200,100,50],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }


config_dict["ttH_DL_baseline_4nodes"] = {
        "layers":                   [200,100,50],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adamax(1e-4),
        "activation_function":      "selu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }



config_dict["ttH_DL_baseline_binary_2016_cate3"] = {
        "layers":                   [128,32,128,64],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.12748404837328636,
        "L2_Norm":                  4.5772100366125575e-05,
        "batch_size":               64,
        "optimizer":                optimizers.Adam(0.0005674365300580737),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2016_cate4"] = {
        "layers":                   [128,32,32],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.13270448688899927,
        "L2_Norm":                  0.00041008021768239024,
        "batch_size":               512,
        "optimizer":                optimizers.Adamax(0.015456236337080285),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }


config_dict["ttH_DL_baseline_binary_2016_cate6"] = {
        "layers":                   [1024,128,32,128],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.003026900377096131,
        "L2_Norm":                  5.6833588304861874e-05,
        "batch_size":               256,
        "optimizer":                optimizers.Adamax(0.0020101073191092344),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2016_cate7"] = {
        "layers":                   [128,32,512,128],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.1457494694862035,
        "L2_Norm":                  4.637244266655528e-05,
        "batch_size":               256,
        "optimizer":                optimizers.Adagrad(0.010558352789053937),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }


config_dict["ttH_DL_baseline_binary_2017_cate3"] = {
        "layers":                   [1024,32,256],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.5822008887747226,
        "L2_Norm":                  0.00010017918217081143,
        "batch_size":               256,
        "optimizer":                optimizers.Adagrad(0.00928036435299591),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }


config_dict["ttH_DL_baseline_binary_2017_cate4"] = {
        "layers":                   [256,1024,128],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.28181902982790064,
        "L2_Norm":                  0.0001202797748884616,
        "batch_size":               512,
        "optimizer":                optimizers.Adamax(0.0006140227485583491),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2017_cate6"] = {
        "layers":                   [512,128,64,32],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.394124958819545,
        "L2_Norm":                  0.00011621119496674054,
        "batch_size":               128,
        "optimizer":                optimizers.Adamax(0.006024121660925635),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2017_cate7"] = {
        "layers":                   [256,1024,64],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.21037685345643553,
        "L2_Norm":                  0.0011995393522218549,
        "batch_size":               128,
        "optimizer":                optimizers.Adamax(0.0005063116291515677),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2017_cate8"] = {
        "layers":                   [512,256,128],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.39806223226633575,
        "L2_Norm":                  0.001224792816526228,
        "batch_size":               256,
        "optimizer":                optimizers.Adamax(0.0040680025941310675),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2018_cate3"] = {
        "layers":                   [256,512,256,1024],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.05872983652493688,
        "L2_Norm":                  0.0007282444019616257,
        "batch_size":               64,
        "optimizer":                optimizers.Adadelta(0.08328163286824546),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2018_cate4"] = {
        "layers":                   [128,1024,128],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.23895904430513185,
        "L2_Norm":                  0.0008867021524403034,
        "batch_size":               256,
        "optimizer":                optimizers.Adam(0.0002582956951491525),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2018_cate6"] = {
        "layers":                   [512,32,512,512],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.21785140099313832,
        "L2_Norm":                  0.00017734887733640974,
        "batch_size":               128,
        "optimizer":                optimizers.RMSprop(0.00021580781103098322),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2018_cate7"] = {
        "layers":                   [256,512,256,32],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.030545095557474644,
        "L2_Norm":                  0.00019392339508446794,
        "batch_size":               2048,
        "optimizer":                optimizers.Adadelta(0.7598793710164607),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_binary_2018_cate8"] = {
        "layers":                   [256,1024,64,32],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.2828152964152687,
        "L2_Norm":                  0.019413253542910437,
        "batch_size":               64,
        "optimizer":                optimizers.RMSprop(0.000913536447897391),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }


config_dict["ttH_DL_baseline_binary_cate8"] = {
        "layers":                   [256,1024,64,32],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.10064369953731123,
        "L2_Norm":                  0.0005763555161387924,
        "batch_size":               128,
        "optimizer":                optimizers.Adadelta(0.00913536447897391),
        "activation_function":      "relu",
        "output_activation":        "tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_4nodes"] = {
        "layers":                   [200,100,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adamax(1e-4),
        "activation_function":      "selu",
        "output_activation":        "softmax",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_4nodes_2017_cate9"] = {
        "layers":                   [1024,1024,64],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.11912344948687953,
        "L2_Norm":                  0.00020058059537553181,
        "batch_size":               8,
        "optimizer":                optimizers.Adamax(4.6697054716212054e-05),
        "activation_function":      "relu",
        "output_activation":        "softmax",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_DL_baseline_4nodes_2018_cate9"] = {
        "layers":                   [1024,64,32],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.349743248488446,
        "L2_Norm":                  7.562564129389872e-05,
        "batch_size":               2048,
        "optimizer":                optimizers.Adam(0.0011281878455195152),
        "activation_function":      "relu",
        "output_activation":        "softmax",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        }
