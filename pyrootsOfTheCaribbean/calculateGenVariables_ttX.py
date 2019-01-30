import os
import sys
import glob
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(filedir)

import miniAODplotting.sampleProcessor as processor
import miniAODplotting.NAFSubmit as NAFSubmit
import sample_configs.mAODsamples_unskimmed as sampleConfig

samples = {}
samples["ttZJets"] = {
    "data":     sampleConfig.get_ttZJets("samples"),
    "XSWeight": sampleConfig.get_ttZJets("XSWeight")}

samples["ttZll"] = {
    "data":     sampleConfig.get_ttZll("samples"),
    "XSWeight": sampleConfig.get_ttZll("XSWeight")}

samples["ttZqq"] = {
    "data":     sampleConfig.get_ttZqq("samples"),
    "XSWeight": sampleConfig.get_ttZqq("XSWeight")}

samples["ttHbb"] = {
    "data":     sampleConfig.get_ttHbb("samples"),
    "XSWeight": sampleConfig.get_ttHbb("XSWeight")}

# output directory
output_dir = basedir + "/workdir/miniAODGenLevelData/ttbarSystem/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# remove old hdf5 files
h5_files = glob.glob(output_dir+"/*.h5")
for f in h5_files: os.remove(f)

# create shell scripts
shellscripts, sample_parts = processor.generate_submit_scripts(samples, output_dir, filedir)

submit = True
if len(sys.argv) > 1:
    if sys.argv[1] == "--noSubmit":
        submit = False
        
# submit them to naf
if submit:
    jobids = NAFSubmit.submitToBatch(output_dir, shellscripts)
    NAFSubmit.monitorJobStatus(jobids)

# concatenate shell scripts
processor.concat_samples(sample_parts, output_dir)



