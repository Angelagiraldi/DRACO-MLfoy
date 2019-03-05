import ROOT
import os
import sys
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

from evaluationScripts.plotVariables import variablePlotter

# location of input dataframes
data_dir = "/ceph/vanderlinden/MLFoyTrainData/ttbarMatcher/"

# output location of plots
plot_dir = "/ceph/vanderlinden/ttH_2017/ttbarMatcher"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "ratio":        False,
    "logscale":     False,
    "scaleSignal":  -1,
    "lumiScale":    1,
    "unweighted":   True,
    }
# additional variables to plot
additional_variables = [
    ]


# initialize plotter
plotter = variablePlotter(
    output_dir      = plot_dir,
    variable_set    = None,
    add_vars        = additional_variables,
    plotOptions     = plotOptions
    )

# add samples
plotter.addSample(
    sampleName      = "ttbar",
    sampleFile      = data_dir+"/ttbar_input.h5",
    plotColor       = ROOT.kRed,
    signalSample    = False)


# add JT categories
plotter.addCategory("SL")


# perform plotting routine
plotter.plot()
