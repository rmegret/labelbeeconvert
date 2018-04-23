
#%%
%load_ext autoreload
%autoreload 2

#%%

import sys
import argparse

sys.path.append('/Users/megret/Documents/Research/BeeTracking/Soft/labelbeeconvert')

from labelbeeconvert import labelbee_convert as lb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

# Make sure current directory is where the data is

args = argparse.Namespace()    
args.inputlist = "inputlist.csv"
args.output = "test_output.csv"
args.plot = True
	
#lb.main(args)
evts=lb.load_fileset(args.inputlist)

#%%
evts_val = evts.query('falsealarm!=True & wrongid!=True')
evts_val.index = range(evts_val.shape[0])
evts_val.reindex()

fig=lb.plot_activities(evts_val);
plt.tight_layout();
plt.show()
