# Train NNs
## Before Usage
## Package-requirements
- execute on NAF with `CMSSW_9_4_9` or newer
- uproot `pip install --user uproot==3.2.7`
- `pip install --user uproot-methods==0.2.6`
- `pip install --user awkward==0.4.2`
- TensorFlow backend for KERAS `export KERAS_BACKEND=tensorflow`

------------------------------------------------------------------------------------------

# Preprocessing Samples
## Adjust settings in `preprocessing_ttHbb_DL_${YEAR}.py`

-tag associated to a Data set (corresponds to year: 2016, 2017, 2018)
```bash
YEAR=2018
```

- change/add event categories (default event categories are `ttH_categories`, `ttbar_bb_categories`,`ttbar_b_categories`,`ttbar_2b_categories`,`ttcc_categories`,`ttlf_categories`; `ttbb_categories` is for ttbb+tt2b+ttb and `binary_bkg_categories` for ttbb+tt2b+ttb+ttcc+ttlf)
```python
	EVENTCATEGORYNAME=root2pandas.EventCategories()
```

- change/add categories of event categories (`SELECTION` can be `None`)
```python
   EVENTCATEGORYNAME.addCategory(CATEGORYNAME, selection = SELECTION)
 ```

- `ntuplesPath` as absolute path to ntuples

- change/add samples of the dataset used with
```python
	dataset.addSample(SampleName  = SAMPLENAME,
    			ntuples     = PATHTONTUPLES,
    			categories  = EVENTCATEGORYNAME,
    			selections  = SELECTION)
```

- change `additional_variables` for variables needed to preprocess, that are not defined in the selection and not needed for training


## Usage
To execute with default options use
```bash
python preprocessing_ttHbb_DL_${YEAR}.py
```
or use the following for options
- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path (default is `InputFeatures`)
- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed (default is `variables_ttHbb_DL_inputvalidation_${YEAR}`)
- `-t STR` to select the tree (or trees for ge4jge3t) corresponding to the right category  (remember to ALWAYS define it, by default is empty)
- `-e INT` to change the maximal number of entries for each batch to restrict memory usage (default is `100000`)
- `-n STR` to change the naming of the output file

```bash
python preprocessing.py -o DIR -v FILE -e INT -n STR
```
An example is:
```bash
python preprocessing.py -o InputFeatures/${YEAR}/${category}  -t liteTreeTTH_step7_${category}
```

------------------------------------------------------------------------------------------
# Run first-order derivatives for the Taylor Expansion of the ANNs outputs

To compute the first-order derivatives for the DNN Taylor expansion add the funciton "dnn.get_gradients(options.isBinary())" in `train_template.py`.
Whenever an architecture in `net_configs.py` is added or its name is changed, remember to  change the corresponding TensorFlow architecture in `net_configs_tensorflow.py`.
The TensorFlow architecture has to have the same name as the keras one with the additional `_tensorflow` at the end.

## Usage
To execute with default options use
```bash
python train_template.py
```
or use the following options
1. Category used for training
    - `-c STR` name of the category `(ge/le)[nJets]j_(ge/le)[nTags]t`
    (default is `ge4j_ge3t`)

2. Sample Options
    - `-o DIR` to change the name of the output directory (absolute path or path relative to `workdir`)
        (default is `test_training`)
    - `-i DIR` to change the name of the input directory where the preprocessed h5 files are stored. Which files in this directory are used for training has to be adjusted in the script itself
        (default is `InputFeatures`)
    - `--naming=STR` to adjust the naming of the input files.
        (default is `_dnn.h5`)
    - `--even` to select only events with `Evt_Odd==0`
    - `--odd` to select only events with `Evt_Odd==1`

3. Training Options
    - `-v FILE` to change the variable selection (absolute path to the variable set file or path relative to `variable_sets` directory)
        (default is `example_variables`)
    - `-e INT` change number of training epochs
        (default is `1000`)
    - `-n STR` STR of the config name in`net_configs.py` file to adjust the architecture of the neural network
    - `--balanceSamples` activates an additional balancing of train samples. With this options the samples which have fewer events are used multiple times in one pass over the training set (epoch). As a default options the sample weights are balanced, such that the sum of train weights is equal for all used samples.
    - `-a` comma separated list of samples to be activated for training. If this option is not used all samples are used as a default.
    - `-u` to NOT perform a normalization of input features to mean zero and std deviation one.

4. Plotting Options
    - `-p` to create plots of the output
    - `-L` to create plots with logarithmic y-axis
    - `-R` to print ROC value for confusion matrix
    - `-P` to create private work label
    - `-S STR` to change the plotted signal class (not part of the background stack plot), possible to do a combination, for example `ttH,ttbb`
        (default is `None`)
    - `-s FLOAT` to scale the signal histograms in the output plots (default is -1 and scales to background integral)

5. Binary Training Options
    - `--binary` activate binary training by defining one signal and one background class. Which samples are set as signal is defined by the `signal` option
    - `-t` target value for background samples during training.
        (default is 0, default for signal 1 and cannot be changed)
    - `--signal=STR` signal class for binary classification, possible to do a combination, for example `ttH,ttbb` (default is `None`)

6. Adversary Training Options
    - `--adversary` activate adversary training with an additional network competing with the classifier. Which samples are set as nominal is defined by `naming` and additional samples by `addsamplenaming`.
    - `--penalty=FLOAT` change the penalty parameter in the adversary loss function (default ist `10`)
    - `--addsamplenaming=STR` to adjust the naming of the input files of additional samples from other generators (default is `_dnn_OL.h5`)
    - specify hyperparameters for adversarial training, like training epochs and adversary iterations, in `net_configs.py`.


Example:
```bash
python train_template.py -i /path/to/input/files/ -o testRun --netconfig=test_config --plot --printroc -c ge6j_ge3t --epochs=1000 --signalclass=ttHbb,ttbb
```


Re-evaluate DNN after training with
```bash
python eval_template.py
```
using the option `-i DIR` to specify the path to the already trained network.

The plotting options of `train_template.py` are also avaiable for this script.

Example:
```bash
python eval_template.py -i testRun_ge6j_ge3t -o eval_testRun --plot --printroc
```


Compute the average of the weights for different layers and different seeds:

Run train_template.py calling the function get_weights() (not only get_input_weights()) in order to have the "absolute_weight_sum_layer *.csv" for the input layer and each dropout layer.
If you want to compute the average over multiple trainings with different seed, run train_template.py several times and copy all the folders inside one folder (/path/to/trained/networks/)

```bash
train_scripts/average_weights.py -i /path/to/trained/networks/
```
using the option `-i DIR` to specify the path to the directory inside which there are the folders for the already trained networks with different seeds.

using the option `-l INT` to specify the number of layers to consider in the computation.


To compute the first-order derivatives for the DNN Taylor expansion add the funciton "dnn.get_gradients(options.isBinary())" in `train_template.py`. Whenever you add/change an architecture in `net_configs.py`, remember to  change the corresponding TensorFlow architecture in `net_configs_tensorflow.py`. The TensorFlow architecture has to have the same name as the keras one with the additional "_tensorflow" at the end.


## Interface to pyroot plotscripts
The DNNs which are trained with this framework can be evaluated with the `pyroot-plotscripts` framework.
For this purpose, generate a directory (e.g `DNNSet`) containing subdirectories (e.g. `4j_ge3t_dnn`, etc.) for each separately trained DNN.
Copy the content of the `checkpoints` directory created after the DNN training to these subdirectories.
The directory of sets of DNNs can be set in the plotLimits script at the `checkpointFiles` option.
To generate a config for plotting the DNN discriminators and the input features, execute `python util/dNNInterfaces/MLfoyInterface.py -c /PATH/TO/DNNSet/` which generates a file `autogenerated_plotconfig.py`. Move this file to the `configs` directory and rename it if wanted.
Specify this filename in the plotLimits script as `plot_cfg`.
