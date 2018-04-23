#!/usr/bin/env python3

import sys
import argparse

sys.path.append('PATH to labelbeeconvert')

from labelbeeconvert import labelbee_convert as lb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


args = argparse.Namespace()    
args.inputlist = "PATH TO inputlist.csv"
args.output = "output_merged.csv"
args.plot = False
	
lb.main(args)
