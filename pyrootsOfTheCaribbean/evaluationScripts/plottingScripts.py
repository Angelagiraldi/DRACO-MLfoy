import os
import sys
import numpy as np
import ROOT
import copy
from array import array

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from root_numpy import hist2array

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
pyrootdir = os.path.dirname(filedir)
basedir  = os.path.dirname(pyrootdir)
sys.path.append(pyrootdir)
sys.path.append(basedir)

import plot_configs.setupPlots as setup


class plotDiscriminators:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, sigScale = -1):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax( self.prediction_vector, axis = 1)

        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale
        self.sigScale          = sigScale
        self.signalIndex       = []
        self.signalFlag        = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
                self.signalFlag.append(self.data.get_class_flag(signal))

        # default settings
        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio = False, printROC = False, privateWork = False):
        self.printROCScore = printROC
        self.privateWork = privateWork

        allBKGhists = []
        allSIGhists = []
        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            print("\nPLOTTING OUTPUT NODE '"+str(node_cls))+"'"

            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag  = self.signalFlag
            else:
                signalIndex = [nodeIndex]
                signalFlag  = [self.data.get_class_flag(node_cls)]

            # get output values of this node
            out_values = self.prediction_vector[:,i]

            if self.printROCScore and len(signalIndex)==1:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag[0], out_values)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            sig_values = []
            sig_labels = []
            sig_weights = []

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex]

                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex]

                print("{} events in discriminator: {}\t(Integral: {})".format(truth_cls, len(filtered_values), sum(filtered_weights)))

                if j in signalIndex:
                    # signal histogram
                    sig_values.append(filtered_values)
                    sig_labels.append(str(truth_cls))
                    sig_weights.append(filtered_weights)
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)

                    histogram = setup.setupHistogram(
                        values    = filtered_values,
                        weights   = filtered_weights,
                        nbins     = self.nbins,
                        bin_range = self.bin_range,
                        color     = setup.GetPlotColor(truth_cls),
                        xtitle    = str(truth_cls)+" at "+str(node_cls)+" node",
                        ytitle    = setup.GetyTitle(self.privateWork),
                        filled    = True)

                    bkgHists.append( histogram )

                    bkgLabels.append( truth_cls )
            allBKGhists.append( bkgHists )
            sigHists = []
            scaleFactors = []
            for iSig in range(len(sig_labels)):
                # setup signal histogram
                sigHist = setup.setupHistogram(
                    values    = sig_values[iSig],
                    weights   = sig_weights[iSig],
                    nbins     = self.nbins,
                    bin_range = self.bin_range,
                    color     = setup.GetPlotColor(sig_labels[iSig]),
                    xtitle    = str(sig_labels[iSig])+" at "+str(node_cls)+" node",
                    ytitle    = setup.GetyTitle(self.privateWork),
                    filled    = False)

                # set signal histogram linewidth
                sigHist.SetLineWidth(3)

                # set scalefactor
                if self.sigScale == -1:
                    scaleFactor = weightIntegral/(sum(sig_weights[iSig])+1e-9)
                else:
                    scaleFactor = float(self.sigScale)
                allSIGhists.append(sigHist.Clone())
                sigHist.Scale(scaleFactor)
                sigHists.append(sigHist)
                scaleFactors.append(scaleFactor)

            # rescale histograms if privateWork is enabled
            if privateWork:
                for sHist in sigHists:
                    sHist.Scale(1./sHist.Integral())
                for bHist in bkgHists:
                    bHist.Scale(1./weightIntegral)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}

            # initialize canvas
            canvas = setup.drawHistsOnCanvas(
                sigHists, bkgHists, plotOptions,
                canvasName = node_cls+" final discriminator")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            for i, h in enumerate(sigHists):
                legend.AddEntry(h, sig_labels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            if self.printROCScore and len(signalIndex)==1:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi or private work label to plot
            if self.privateWork:
                setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
            else:
                setup.printLumi(canvas, ratio = plotOptions["ratio"])

            # add category label
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir + "/finaldiscr_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/finaldiscr_*.pdf "+str(workdir)+"/discriminators.pdf"
        print(cmd)
        os.system(cmd)

        # create combined histos for max Likelihood fit
        h_bkg = np.array([])
        h_sig = np.array([])
        for l_h in allBKGhists:
            h_tmp=l_h[0].Clone()
            h_tmp.Reset()
            for h in l_h:
                h_tmp.Add(h)
            h_bkg = np.concatenate((h_bkg,hist2array(h_tmp)), axis=None)

        for h in allSIGhists:
            h_sig = np.concatenate((h_sig,hist2array(h)), axis=None)
        return h_bkg, h_sig

class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, sigScale = -1):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale
        self.sigScale          = sigScale
        self.signalIndex       = []
        self.signalFlag        = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
                self.signalFlag.append(self.data.get_class_flag(signal))

        # default settings
        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio = False, printROC = False, privateWork = False):
        self.printROCScore = printROC
        self.privateWork = privateWork

        f = ROOT.TFile(self.plotdir + "/Discriminator.root", "RECREATE")
        f.cd()

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            # get output values of this node
            out_values = self.prediction_vector[:,i]

            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag  = self.signalFlag
            else:
                signalIndex = [nodeIndex]
                signalFlag  = [self.data.get_class_flag(node_cls)]

            if self.printROCScore and len(signalIndex)==1:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag[0], out_values)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex]

                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex]

                if j in signalIndex:
                    # signal histogram
                    sig_values  = filtered_values
                    sig_label   = str(truth_cls)
                    sig_weights = filtered_weights
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)

                    histogram = setup.setupHistogram(
                        values    = filtered_values,
                        weights   = filtered_weights,
                        nbins     = self.nbins,
                        bin_range = self.bin_range,
                        color     = setup.GetPlotColor(truth_cls),
                        xtitle    = str(truth_cls)+" at "+str(node_cls)+" node",
                        ytitle    = setup.GetyTitle(self.privateWork),
                        filled    = True)

                    bkgHists.append( histogram )
                    bkgLabels.append( truth_cls )

            # setup signal histogram
            sigHist = setup.setupHistogram(
                values    = sig_values,
                weights   = sig_weights,
                nbins     = self.nbins,
                bin_range = self.bin_range,
                color     = setup.GetPlotColor(sig_label),
                xtitle    = str(sig_label)+" at "+str(node_cls)+" node",
                ytitle    = setup.GetyTitle(self.privateWork),
                filled    = False)

            # set signal histogram linewidth
            sigHist.SetLineWidth(3)

            # set scalefactor
            if self.sigScale == -1:
                scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            else:
                scaleFactor = float(self.sigScale)
            sigHist.Scale(scaleFactor)

            # rescale histograms if privateWork enabled
            if privateWork:
                sigHist.Scale(1./sigHist.Integral())
                for bHist in bkgHists:
                    bHist.Scale(1./weightIntegral)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}

            # initialize canvas
            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHists, plotOptions,
                canvasName = node_cls+" node")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            legend.AddEntry(sigHist, sig_label+" x {:4.0f}".format(scaleFactor), "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            if self.printROCScore and len(signalIndex)==1:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi or private work label to plot
            if self.privateWork:
                setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
            else:
                setup.printLumi(canvas, ratio = plotOptions["ratio"])

            # add category label
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir + "/outputNode_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        f.cd()
        f.Write()
        f.Close()
        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/outputNode_*.pdf "+str(workdir)+"/outputNodes.pdf"
        print(cmd)
        os.system(cmd)



class plotClosureTest:
    def __init__(self, data, test_prediction, train_prediction, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False):
        self.data               = data
        self.test_prediction    = test_prediction
        self.train_prediction   = train_prediction

        self.pred_classes_test  = np.argmax(self.test_prediction, axis = 1)
        self.pred_classes_train = np.argmax(self.train_prediction, axis = 1)

        self.event_classes      = event_classes
        self.bin_range          = bin_range

        self.signal_class       = signal_class
        self.event_category     = event_category
        self.plotdir            = plotdir
        self.logscale           = logscale
        self.signalIndex       = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])


        # generate sub directory
        self.plotdir += "/ClosurePlots/"
        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

        # default settings
        self.privateWork = False

    def plot(self, ratio = False, privateWork = False):
        self.privateWork = privateWork

        # loop over output nodes
        for i, node_cls in enumerate(self.event_classes):
            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            print("nodeIndex")
            print(nodeIndex)
            if self.signal_class:
                signalIndex = self.signalIndex
                signalClass = self.signal_class
            else:
                signalIndex = [nodeIndex]
                signalClass = node_cls
            print("signalIndex")
            print(signalIndex)
            print("signalClass")
            print(signalClass)

            # get output values of this node
            test_values = self.test_prediction[:,i]
            train_values = self.train_prediction[:,i]


            test_values_ttH = [test_values[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 0  \
                and self.pred_classes_test[k] == nodeIndex]

            test_values_ttbb = [test_values[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 1  \
                and self.pred_classes_test[k] == nodeIndex]

            test_values_ttcc = [test_values[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 2  \
                and self.pred_classes_test[k] == nodeIndex]

            test_values_ttlf = [test_values[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 3  \
                and self.pred_classes_test[k] == nodeIndex]



            train_values_ttH = [train_values[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 0  \
                and self.pred_classes_train[k] == nodeIndex]

            train_values_ttbb = [train_values[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 1  \
                and self.pred_classes_train[k] == nodeIndex]

            train_values_ttcc = [train_values[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 2  \
                and self.pred_classes_train[k] == nodeIndex]

            train_values_ttlf = [train_values[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 3  \
                and self.pred_classes_train[k] == nodeIndex]



            test_weights_ttH = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 0  \
                and self.pred_classes_test[k] == nodeIndex]

            test_weights_ttbb = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 1  \
                and self.pred_classes_test[k] == nodeIndex]

            test_weights_ttcc = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 2  \
                and self.pred_classes_test[k] == nodeIndex]

            test_weights_ttlf = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == 3  \
                and self.pred_classes_test[k] == nodeIndex]



            train_weights_ttH = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 0  \
                and self.pred_classes_train[k] == nodeIndex]

            train_weights_ttbb = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 1  \
                and self.pred_classes_train[k] == nodeIndex]

            train_weights_ttcc = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 2  \
                and self.pred_classes_train[k] == nodeIndex]

            train_weights_ttlf = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == 3  \
                and self.pred_classes_train[k] == nodeIndex]

            if node_cls == "ttH":
                nbins = 20
            elif node_cls == "ttbb":
                nbins = 20
            elif node_cls == "ttcc":
                nbins = 10
            elif node_cls == "ttlf":
                nbins = 10

            # setup train histograms
            ttH_train = setup.setupHistogram(
                values      = train_values_ttH,
                weights     = train_weights_ttH,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kBlue,
                xtitle      = "ttH train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            ttH_train.Scale(1./ttH_train.Integral())
            ttH_train.SetLineWidth(1)
            ttH_train.SetFillColorAlpha(ROOT.kBlue, 0.25)

            ttbb_train = setup.setupHistogram(
                values      = train_values_ttbb,
                weights     = train_weights_ttbb,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kRed,
                xtitle      = "ttbb train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            ttbb_train.Scale(1./ttbb_train.Integral())
            ttbb_train.SetLineWidth(1)
            ttbb_train.SetFillColorAlpha(ROOT.kRed, 0.25)

            ttcc_train = setup.setupHistogram(
                values      = train_values_ttcc,
                weights     = train_weights_ttcc,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kGreen,
                xtitle      = "ttcc train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            ttcc_train.Scale(1./ttcc_train.Integral())
            ttcc_train.SetLineWidth(1)
            ttcc_train.SetFillColorAlpha(ROOT.kGreen, 0.25)

            ttlf_train = setup.setupHistogram(
                values      = train_values_ttlf,
                weights     = train_weights_ttlf,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kOrange,
                xtitle      = "ttlf train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            ttlf_train.Scale(1./ttlf_train.Integral())
            ttlf_train.SetLineWidth(1)
            ttlf_train.SetFillColorAlpha(ROOT.kOrange, 0.25)


            # setup test histograms
            ttH_test = setup.setupHistogram(
                values      = test_values_ttH,
                weights     = test_weights_ttH,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kBlue,
                xtitle      = "ttH test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            ttH_test.Scale(1./ttH_test.Integral())
            ttH_test.SetLineWidth(1)
            ttH_test.SetMarkerStyle(20)
            ttH_test.SetMarkerSize(1)

            ttbb_test = setup.setupHistogram(
                values      = test_values_ttbb,
                weights     = test_weights_ttbb,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kRed,
                xtitle      = "ttbb test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            ttbb_test.Scale(1./ttbb_test.Integral())
            ttbb_test.SetLineWidth(1)
            ttbb_test.SetMarkerStyle(20)
            ttbb_test.SetMarkerSize(1)

            ttcc_test = setup.setupHistogram(
                values      = test_values_ttcc,
                weights     = test_weights_ttcc,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kGreen,
                xtitle      = "ttcc test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            ttcc_test.Scale(1./ttcc_test.Integral())
            ttcc_test.SetLineWidth(1)
            ttcc_test.SetMarkerStyle(20)
            ttcc_test.SetMarkerSize(1)

            ttlf_test = setup.setupHistogram(
                values      = test_values_ttlf,
                weights     = test_weights_ttlf,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kOrange,
                xtitle      = "ttlf test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            ttlf_test.Scale(1./ttlf_test.Integral())
            ttlf_test.SetLineWidth(1)
            ttlf_test.SetMarkerStyle(20)
            ttlf_test.SetMarkerSize(1)

            plotOptions = {"logscale": self.logscale}

            # init canvas
            canvas = setup.drawClosureTestOnCanvas(
                ttH_train, ttbb_train, ttcc_train, ttlf_train, ttH_test, ttbb_test, ttcc_test, ttlf_test, plotOptions,
                canvasName = "closure test at {} node".format(node_cls))

            # setup legend
            legend = setup.getLegend()

            legend.SetTextSize(0.02)
            ks_ttH = ttH_train.KolmogorovTest(ttH_test)
            ks_ttbb = ttbb_train.KolmogorovTest(ttbb_test)
            ks_ttcc = ttcc_train.KolmogorovTest(ttcc_test)
            ks_ttlf = ttlf_train.KolmogorovTest(ttlf_test)

            # add entries
            legend.AddEntry(ttH_train, "train ttH", "F")
            legend.AddEntry(ttbb_train, "train ttbb", "F")
            legend.AddEntry(ttcc_train, "train ttcc", "F")
            legend.AddEntry(ttlf_train, "train ttlf", "F")

            legend.AddEntry(ttH_test,  "test ttH (KS = {:.3f})".format(ks_ttH), "L")
            legend.AddEntry(ttbb_test,  "test ttbb (KS = {:.3f})".format(ks_ttbb), "L")
            legend.AddEntry(ttcc_test,  "test ttcc (KS = {:.3f})".format(ks_ttcc), "L")
            legend.AddEntry(ttlf_test,  "test ttlf (KS = {:.3f})".format(ks_ttlf), "L")
            print(node_cls)
            print(ks_ttH)
            print(ks_ttbb)
            print(ks_ttcc)
            print(ks_ttlf)

            # draw legend
            legend.Draw("same")

            # prit private work label if activated
            if self.privateWork:
                setup.printPrivateWork(canvas)
            # add category label
            setup.printCategoryLabel(canvas, self.event_category)

            # add private work label if activated
            # if self.privateWork:
            #     setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)

            out_path = self.plotdir+"/closureTest_at_{}_node.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(os.path.dirname(self.plotdir[:-1]))
        cmd = "pdfunite "+str(self.plotdir)+"/closureTest_*.pdf "+str(workdir)+"/closureTest.pdf"
        print(cmd)
        os.system(cmd)

class plotBinaryClosureTest:
    def __init__(self, data, test_prediction, train_prediction, nbins, bin_range, signal_class, event_category, plotdir, logscale = False):
        self.data               = data
        self.test_prediction    = test_prediction
        self.train_prediction   = train_prediction

        self.bin_range          = bin_range
        self.nbins              = 20
        self.signal_class       = signal_class
        self.event_category     = event_category
        self.plotdir            = plotdir
        self.logscale           = logscale

        # generate sub directory
        self.plotdir += "/ClosurePlots/"
        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

        # default settings
        self.privateWork = False

    def plot(self, ratio = False, privateWork = False):
        self.privateWork = privateWork

        # loop over output nodes
        for i, node_cls in enumerate(self.event_classes):

            test_values_signal = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
                if self.data.get_test_labels()[k] == 1 ]

            test_values_background = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
                if not self.data.get_test_labels()[k] == 1 ]

            train_values_signal = [ self.train_predictions[k] for k in range(len(self.train_predictions)) \
                if self.data.get_train_labels()[k] == 1 ]

            train_values_background = [ self.train_predictions[k] for k in range(len(self.train_predictions)) \
                if not self.data.get_train_labels()[k] == 1 ]



            test_weights_signal = [ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
                if self.data.get_test_labels()[k] == 1]

            test_weights_background = [ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
                if not self.data.get_test_labels()[k] == 1]

            train_weights_signal = [ self.data.get_lumi_weights()[k] for k in range(len(self.train_predictions)) \
                if self.data.get_train_labels()[k] == 1]

            train_weights_background = [ self.data.get_lumi_weights()[k] for k in range(len(self.train_predictions)) \
                if not self.data.get_train_labels()[k] == 1]


            # setup train histograms
            ttH_train = setup.setupHistogram(
                values      = train_values_signal,
                weights     = train_weights_signal,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kBlue,
                xtitle      = "signal train",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            ttH_train.Scale(1./ttH_train.Integral())
            ttH_train.SetLineWidth(1)
            ttH_train.SetFillColorAlpha(ROOT.kBlue, 0.25)

            ttbb_train = setup.setupHistogram(
                values      = train_values_background,
                weights     = train_weights_background,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kRed,
                xtitle      = "background train",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            ttbb_train.Scale(1./ttbb_train.Integral())
            ttbb_train.SetLineWidth(1)
            ttbb_train.SetFillColorAlpha(ROOT.kRed, 0.25)

            # setup test histograms
            ttH_test = setup.setupHistogram(
                values      = test_values_signal,
                weights     = test_weights_signal,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kBlue,
                xtitle      = "signal test",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            ttH_test.Scale(1./ttH_test.Integral())
            ttH_test.SetLineWidth(1)
            ttH_test.SetMarkerStyle(20)
            ttH_test.SetMarkerSize(1)

            ttbb_test = setup.setupHistogram(
                values      = test_values_background,
                weights     = test_weights_background,
                nbins       = nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kRed,
                xtitle      = "background test",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            ttbb_test.Scale(1./ttbb_test.Integral())
            ttbb_test.SetLineWidth(1)
            ttbb_test.SetMarkerStyle(20)
            ttbb_test.SetMarkerSize(1)
            plotOptions = {"logscale": self.logscale}

            # init canvas
            canvas = setup.drawBinaryClosureTestOnCanvas(
                ttH_train, ttbb_train, ttcc_train, ttlf_train, ttH_test, ttbb_test, ttcc_test, ttlf_test, plotOptions,
                canvasName = "closure test at {} node".format(node_cls))

            # setup legend
            legend = setup.getLegend()

            legend.SetTextSize(0.02)
            ks_ttH = ttH_train.KolmogorovTest(ttH_test)
            ks_ttbb = ttbb_train.KolmogorovTest(ttbb_test)

            # add entries
            legend.AddEntry(ttH_train, "train ttH", "F")
            legend.AddEntry(ttbb_train, "train ttbb", "F")


            legend.AddEntry(ttH_test,  "test ttH (KS = {:.3f})".format(ks_ttH), "L")
            legend.AddEntry(ttbb_test,  "test ttbb (KS = {:.3f})".format(ks_ttbb), "L")

            print(node_cls)
            print(ks_ttH)
            print(ks_ttbb)

            # draw legend
            legend.Draw("same")

            # prit private work label if activated
            if self.privateWork:
                setup.printPrivateWork(canvas)
            # add category label
            setup.printCategoryLabel(canvas, self.event_category)

            # add private work label if activated
            # if self.privateWork:
            #     setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)

            out_path = self.plotdir+"/closureTest_at_{}_node.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(os.path.dirname(self.plotdir[:-1]))
        cmd = "pdfunite "+str(self.plotdir)+"/closureTest_*.pdf "+str(workdir)+"/closureTest.pdf"
        print(cmd)
        os.system(cmd)



class plotConfusionMatrix:
    def __init__(self, data, prediction_vector, event_classes, event_category, plotdir):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes     = event_classes
        self.n_classes         = len(self.event_classes)

        self.event_category    = event_category
        self.plotdir           = plotdir

        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # default settings
        self.ROCScore = None

    def plot(self, norm_matrix = True, privateWork = False, printROC = False):
        if printROC:
            self.ROCScore = roc_auc_score(
                self.data.get_test_labels(), self.prediction_vector)

        # norm confusion matrix if activated
        if norm_matrix:
            new_matrix = np.empty( (self.n_classes, self.n_classes), dtype = np.float64)
            for yit in range(self.n_classes):
                evt_sum = float(sum(self.confusion_matrix[yit,:]))
                for xit in range(self.n_classes):
                    new_matrix[yit,xit] = self.confusion_matrix[yit,xit]/(evt_sum+1e-9)

            self.confusion_matrix = new_matrix


        # initialize Histogram
        cm = setup.setupConfusionMatrix(
            matrix      = self.confusion_matrix.T,
            ncls        = self.n_classes,
            xtitle      = "predicted class",
            ytitle      = "true class",
            binlabel    = self.event_classes)

        canvas = setup.drawConfusionMatrixOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore, privateWork = privateWork)
        setup.saveCanvas(canvas, self.plotdir+"/confusionMatrix.pdf")

class plotEventYields:
    def __init__(self, data, prediction_vector, event_classes, event_category, signal_class, plotdir, logscale, sigScale = -1):
        self.data               = data
        self.prediction_vector  = prediction_vector
        self.predicted_classes  = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes      = event_classes
        self.n_classes          = len(self.event_classes)
        self.signal_class       = signal_class
        self.signalIndex       = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
        else:
            self.signalIndex = [self.data.class_translation["ttHbb"]]

        self.event_category     = event_category
        self.plotdir            = plotdir

        self.logscale           = logscale
        self.sigScale           = sigScale

        self.privateWork = False

    def plot(self, privateWork = False, ratio = False):
        self.privateWork = privateWork

        # loop over processes
        sigHists = []
        sigLabels = []
        bkgHists = []
        bkgLabels = []

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}
        yTitle = "event Yield"
        if privateWork:
            yTitle = setup.GetyTitle(privateWork)

        totalBkgYield = 0

        # generate one plot per output node
        for i, truth_cls in enumerate(self.event_classes):
            classIndex = self.data.class_translation[truth_cls]

            class_yields = []

            # loop over output nodes
            for j, node_cls in enumerate(self.event_classes):

                # get output values of this node
                out_values = self.prediction_vector[:,i]

                nodeIndex = self.data.class_translation[node_cls]

                # get yields
                class_yield = sum([ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex])
                class_yields.append(class_yield)



            if i in self.signalIndex:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = False)

                # set signal histogram linewidth
                histogram.SetLineWidth(2)
                sigHists.append(histogram)
                sigLabels.append(truth_cls)


            else:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = True)
                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)

                totalBkgYield += sum(class_yields)



        # scale histograms according to options
        scaleFactors=[]
        for sig in sigHists:
            if self.sigScale == -1:
                scaleFactors.append(totalBkgYield/sig.Integral())
            else:
                scaleFactors.append(float(self.sigScale))
        if privateWork:
            for sig in sigHists:
                sig.Scale(1./sig.Integral())
            for h in bkgHists:
                h.Scale(1./totalBkgYield)
        else:
            for i,sig in enumerate(sigHists):
                sig.Scale(scaleFactors[i])

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sigHists, bkgHists, plotOptions,
            canvasName = "event yields per node")

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        for i,sig in enumerate(sigHists):
            legend.AddEntry(sig, sigLabels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

        # add background entries
        for i, h in enumerate(bkgHists):
            legend.AddEntry(h, bkgLabels[i], "F")

        # draw legend
        legend.Draw("same")

        # add lumi
        setup.printLumi(canvas, ratio = plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

        out_path = self.plotdir + "/event_yields.pdf"
        setup.saveCanvas(canvas, out_path)


class plotBinaryOutput:
    def __init__(self, data, test_predictions, train_predictions, full_predictions, nbins, bin_range, event_category, plotdir, logscale = False, sigScale = -1):
        self.data               = data
        self.test_predictions   = test_predictions
        self.train_predictions  = train_predictions

        self.nbins              = nbins
        self.bin_range          = bin_range
        self.event_category     = event_category
        self.plotdir            = plotdir
        self.logscale           = logscale
        self.sigScale           = sigScale

        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio = False, printROC = False, privateWork = False, name = "binary discriminator"):
        self.printROCScore = printROC
        self.privateWork = privateWork

        if self.printROCScore:
            roc = roc_auc_score(self.data.get_test_labels(), self.test_predictions)
            print("ROC: {}".format(roc))

        f = ROOT.TFile(self.plotdir + "/binaryDiscriminator.root", "RECREATE")
        f.cd()

        sig_values = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
            if self.data.get_test_labels()[k] == 1 ]
        sig_weights =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
            if self.data.get_test_labels()[k] == 1]

        # sig_values_full = [ self.full_predictions[k] for k in range(len(self.full_predictions)) \
        #     if self.data.get_labels()[k] == 1 ]
        # sig_weights_full =[ self.data.get_full_lumi_weights()[k] for k in range(len(self.full_predictions)) \
        #     if self.data.get_labels()[k] == 1]
        # sig_weights_weights =[ self.data.get_full_weights()[k] for k in range(len(self.full_predictions)) \
        #     if self.data.get_labels()[k] == 1]
        #
        # sig_hist_full = ROOT.TH1F("sgn_full", "Signal distribution; binary DNN output", 20,-1,1)
        # for i in range(len(sig_values_full)):
        #     sig_hist_full.Fill(sig_values_full[i],sig_weights_full[i])
        #
        # sig_hist_full_weights = ROOT.TH1F("sgn_full_weights", "Signal distribution; binary DNN output", 20,-1,1)
        # for i in range(len(sig_values_full)):
        #     sig_hist_full.Fill(sig_values_full[i],sig_weights_weights[i])
        #
        # sig_hist_full_noweights = ROOT.TH1F("sgn_full_noweights", "Signal distribution; binary DNN output", 20,-1,1)
        # for i in range(len(sig_values_full)):
        #     sig_hist_full_noweights.Fill(sig_values_full[i])
        #
        # sig_hist_noweights = ROOT.TH1F("sgn_test_noweights", "Signal distribution; binary DNN output", 20,-1,1)
        # for i in range(len(sig_values)):
        #     sig_hist_noweights.Fill(sig_values[i])

        sig_hist = setup.setupHistogram(
            values      = sig_values,
            weights     = sig_weights,
            nbins       = self.nbins,
            bin_range   = self.bin_range,
            color       = ROOT.kCyan,
            xtitle      = "signal_test",
            ytitle      = setup.GetyTitle(self.privateWork),
            filled      = False)
        sig_hist.SetLineWidth(3)

        bkg_values = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
            if not self.data.get_test_labels()[k] == 1 ]
        bkg_weights =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
            if not self.data.get_test_labels()[k] == 1]

        # bkg_values_cc = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttcc" ]
        # bkg_weights_cc =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttcc"]
        #
        # histogram_cc = setup.setupHistogram(
        #     values    = bkg_values_cc,
        #     weights   = bkg_weights_cc,
        #     nbins     = self.nbins,
        #     bin_range = self.bin_range,
        #     color     = setup.GetPlotColor("ttcc"),
        #     xtitle    = "ttcc",
        #     ytitle    = setup.GetyTitle(self.privateWork),
        #     filled    = True)
        #
        # bkg_values_lf = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttlf"]
        # bkg_weights_lf =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttlf"]
        #
        # histogram_lf = setup.setupHistogram(
        #     values    = bkg_values_lf,
        #     weights   = bkg_weights_lf,
        #     nbins     = self.nbins,
        #     bin_range = self.bin_range,
        #     color     = setup.GetPlotColor("ttlf"),
        #     xtitle    = "ttlf",
        #     ytitle    = setup.GetyTitle(self.privateWork),
        #     filled    = True)
        #
        # bkg_values_bb = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttbb"]
        # bkg_weights_bb =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttbb" ]
        #
        # histogram_bb = setup.setupHistogram(
        #     values    = bkg_values_bb,
        #     weights   = bkg_weights_bb,
        #     nbins     = self.nbins,
        #     bin_range = self.bin_range,
        #     color     = setup.GetPlotColor("ttbb"),
        #     xtitle    = "ttbb",
        #     ytitle    = setup.GetyTitle(self.privateWork),
        #     filled    = True)
        #
        # bkg_values_b = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttb"]
        # bkg_weights_b =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttbb" ]
        #
        # histogram_b = setup.setupHistogram(
        #     values    = bkg_values_b,
        #     weights   = bkg_weights_b,
        #     nbins     = self.nbins,
        #     bin_range = self.bin_range,
        #     color     = setup.GetPlotColor("ttb"),
        #     xtitle    = "ttb",
        #     ytitle    = setup.GetyTitle(self.privateWork),
        #     filled    = True)
        #
        # bkg_values_2b = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "tt2b"]
        # bkg_weights_2b =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "tt2b" ]
        #
        # histogram_2b = setup.setupHistogram(
        #     values    = bkg_values_2b,
        #     weights   = bkg_weights_2b,
        #     nbins     = self.nbins,
        #     bin_range = self.bin_range,
        #     color     = setup.GetPlotColor("tt2b"),
        #     xtitle    = "tt2b",
        #     ytitle    = setup.GetyTitle(self.privateWork),
        #     filled    = True)

        # bkg_values_ttbar = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttbar"]
        # bkg_weights_ttbar =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
        #     if self.data.get_class_label()[k] == "ttbar" ]
        #
        # histogram_ttbar = setup.setupHistogram(
        #     values    = bkg_values_ttbar,
        #     weights   = bkg_weights_ttbar,
        #     nbins     = self.nbins,
        #     bin_range = self.bin_range,
        #     color     = setup.GetPlotColor("ttbar"),
        #     xtitle    = "ttbar",
        #     ytitle    = setup.GetyTitle(self.privateWork),
        #     filled    = True)
        #
        # bkgHists = []
        # bkgLabels = []
        # bkgHists.append( histogram_bb )
        # bkgLabels.append("ttbb")
        # bkgHists.append( histogram_2b )
        # bkgLabels.append("tt2b")
        # bkgHists.append( histogram_b )
        # bkgLabels.append("ttb")
        # bkgHists.append( histogram_ttbar )
        # bkgLabels.append("ttbar")
        # bkgHists.append( histogram_cc )
        # bkgLabels.append("ttcc")
        # bkgHists.append( histogram_lf )
        # bkgLabels.append("ttlf")


        # bkg_values_full = [ self.full_predictions[k] for k in range(len(self.full_predictions)) \
        #     if not self.data.get_labels()[k] == 1 ]
        # bkg_weights_full =[ self.data.get_full_lumi_weights()[k] for k in range(len(self.full_predictions)) \
        #     if not self.data.get_labels()[k] == 1]
        # bkg_weights_weights =[ self.data.get_full_weights()[k] for k in range(len(self.full_predictions)) \
        #     if not self.data.get_labels()[k] == 1]
        #
        # bkg_hist_full = ROOT.TH1F("bkg_full", "Background distribution; binary DNN output", 20,-1,1)
        # for i in range(len(bkg_values_full)):
        #     bkg_hist_full.Fill(bkg_values_full[i],bkg_weights_full[i])
        #
        # bkg_hist_full_weights = ROOT.TH1F("bkg_full_weights", "Background distribution; binary DNN output", 20,-1,1)
        # for i in range(len(bkg_values_full)):
        #     bkg_hist_full_weights.Fill(bkg_values_full[i],bkg_weights_weights[i])
        #
        # bkg_hist_full_noweights = ROOT.TH1F("bkg_full_noweights", "Background distribution; binary DNN output", 20, -1,1)
        # for i in range(len(bkg_values_full)):
        #     bkg_hist_full_noweights.Fill(bkg_values_full[i])
        #
        # bkg_hist_noweights = ROOT.TH1F("bkg_test_noweights", "Background distribution; binary DNN output", 20,-1,1)
        # for i in range(len(bkg_values)):
        #     bkg_hist_noweights.Fill(bkg_values[i])

        bkg_hist = setup.setupHistogram(
            values      = bkg_values,
            weights     = bkg_weights,
            nbins       = self.nbins,
            bin_range   = self.bin_range,
            color       = ROOT.kOrange,
            xtitle      = "background_test",
            ytitle      = setup.GetyTitle(self.privateWork),
            filled      = True)

        if self.sigScale == -1:
            scaleFactor = sum(bkg_weights)/(sum(sig_weights)+1e-9)
        else:
            scaleFactor = float(self.sigScale)

        sig_hist_unscaled = sig_hist.Clone()
        sig_hist.Scale(scaleFactor)

        # rescale histograms if privateWork enabled
        if privateWork:
            sig_hist.Scale(1./sig_hist.Integral())
            bkg_hist.Scale(1./bkg_hist.Integral())
            # sig_hist_full.Scale(1./sig_hist_full.Integral())
            # bkg_hist_full.Scale(1./bkg_hist_full.Integral())
            # sig_hist_full_noweights.Scale(1./sig_hist_full_noweights.Integral())
            # bkg_hist_full_noweights.Scale(1./bkg_hist_full_noweights.Integral())
            # sig_hist_noweights.Scale(1./sig_hist_noweights.Integral())
            # bkg_hist_noweights.Scale(1./bkg_hist_noweights.Integral())

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sig_hist, bkg_hist, plotOptions,
            canvasName = name)

        # canvas_detailed = setup.drawHistsOnCanvas(
        #     sig_hist, bkgHists, plotOptions,
        #     canvasName = name+"detailed")

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        legend.AddEntry(sig_hist, "signal x {:4.0f}".format(scaleFactor), "L")

        # add background entries
        legend.AddEntry(bkg_hist, "background", "F")
        # for i, h in enumerate(bkgHists):
        #     legend.AddEntry(h, bkgLabels[i], "F")
        # draw legend
        legend.Draw("same")

        # draw legend
        legend.Draw("same")

        # add ROC score if activated
        if self.printROCScore:
            setup.printROCScore(canvas, roc, plotOptions["ratio"])
            # setup.printROCScore(canvas_detailed, roc, plotOptions["ratio"])

        # add lumi or private work label to plot
        if self.privateWork:
            setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
            # setup.printPrivateWork(canvas_detailed, plotOptions["ratio"], nodePlot = True)
        else:
            setup.printLumi(canvas, ratio = plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])
        # setup.printCategoryLabel(canvas_detailed, self.event_category, ratio = plotOptions["ratio"])

        out_path = self.plotdir + "/binaryDiscriminator.pdf"
        setup.saveCanvas(canvas, out_path)
        # out_path = self.plotdir + "/binaryDiscriminator_detailed.pdf"
        # setup.saveCanvas(canvas_detailed, out_path)

        # returns = np.round_(hist2array(bkg_hist)).astype(int), np.round(hist2array(sig_hist_unscaled)).astype(int)
        # returns = np.round_(hist2array(bkg_hist)), np.round(hist2array(sig_hist_unscaled))
        returns = hist2array(bkg_hist), hist2array(sig_hist_unscaled)

        f.cd()
        f.Write()
        f.Close()
        return returns




class plotEventYields:
    def __init__(self, data, prediction_vector, event_classes, event_category, signal_class, plotdir, logscale):
        self.data               = data
        self.prediction_vector  = prediction_vector
        self.predicted_classes  = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes      = event_classes
        self.n_classes          = len(self.event_classes)
        self.signal_class       = signal_class
        self.signalIndex       = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
        else:
            self.signalIndex = [self.data.class_translation["ttH"]]

        self.event_category     = event_category
        self.plotdir            = plotdir

        self.logscale           = logscale

        self.privateWork = False

    def plot(self, privateWork = False, ratio = False):
        self.privateWork = privateWork

        # loop over processes
        sigHists = []
        sigLabels = []
        bkgHists = []
        bkgLabels = []

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}
        yTitle = "event Yield"
        if privateWork:
            yTitle = setup.GetyTitle(privateWork)

        totalBkgYield = 0

        # generate one plot per output node
        for i, truth_cls in enumerate(self.event_classes):
            classIndex = self.data.class_translation[truth_cls]

            class_yields = []

            # loop over output nodes
            for j, node_cls in enumerate(self.event_classes):

                # get output values of this node
                out_values = self.prediction_vector[:,i]

                nodeIndex = self.data.class_translation[node_cls]

                # get yields
                class_yield = sum([ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex])
                class_yields.append(class_yield)



            if i in self.signalIndex:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = False)

                # set signal histogram linewidth
                histogram.SetLineWidth(3)
                sigHists.append(histogram)
                sigLabels.append(truth_cls)


            else:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = True)
                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)

                totalBkgYield += sum(class_yields)



        # scale histograms according to options
        scaleFactors=[]
        for sig in sigHists:
            scaleFactors.append(totalBkgYield/sig.Integral())
        if privateWork:
            for sig in sigHists:
                sig.Scale(1./sig.Integral())
            for h in bkgHists:
                h.Scale(1./totalBkgYield)
        else:
            for i,sig in enumerate(sigHists):
                sig.Scale(scaleFactors[i])

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sigHists, bkgHists, plotOptions,
            canvasName = "event yields per node")

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        for i,sig in enumerate(sigHists):
            legend.AddEntry(sig, sigLabels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

        # add background entries
        for i, h in enumerate(bkgHists):
            legend.AddEntry(h, bkgLabels[i], "F")

        # draw legend
        legend.Draw("same")

        # add lumi
        setup.printLumi(canvas, ratio = plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

        out_path = self.plotdir + "/event_yields.pdf"
        setup.saveCanvas(canvas, out_path)
