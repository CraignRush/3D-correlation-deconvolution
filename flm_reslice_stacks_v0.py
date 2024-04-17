#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:40:25 2024
Batch script for reslicing stacks for 3DCT and for creating MIPs
Reslicing script adapted from original 3DCT reslicing function
Further features:
- multiple checkpoints 
- binning 
-saving of MIPs as multichannel tif files

NOTE: For large stacks, use a new tifffile version to avoid problems during saving

Dependencies: 
    tifffile for loading stacks (pip install tifffile)
    functions from flm_utility_functions
    
tested with module ANACONDA/3/2023.09 on our vis cluster
@author: anbieber
"""

import os
import re
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import tifffile as tf 
import numpy as np


from flm_utility_functions import tdct_reslice, plot_MIP, lut_from_base_color, bin_image_stack_xy


#%% INPUT

reslice = True
make_MIP = True # whether to save a maximum intensity projection (MIP) as png
make_MIP_tif = True # whether to save as multichannel tif file
MIP_mode = 'max' # 'sum' for Z projection, 'max' for Maximum Intensity Projection

step_z = 300.2 # Input focus step (z step size)
step_xy_orig = 84.651 # Output focus step (xy step size, same unit as step_in)
step_unit_um = 1e-3 # step unit in microns, i.e. 1e-3 for nm
bin_factor = 1 # Factor for binning in xy before reslicing

in_dir = Path(r'/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240416_Yeast_CA_GE_18/huygens_decon/')
out_dir = in_dir / 'resliced'
MIP_dir = in_dir / 'MIP_decon'
MIP_tif_dir = in_dir / 'MIP_decon_tif'

channels_to_process = ['ch00', 'ch01'] # If only one channel, no channel number is given in tif file
color_list = [(1,0,1),(0,1,0)] # color list with normalized (r,g,b) values for channels in right order
hist_clip_factor = 0.3 # histogram clipping factor for enhancing signal, only important for the MIP png


# If FOVs are present multiple times since multiple checkpoints were taken, make sure that they are all found.
# Can also be used if multiple grids are in one folder
FOV_search_term = '(FOV\d*)'
checkpoints = False
checkpoint_search_term = '_(checkpoints\d*)_'
FOV_before_checkpoint = False

logging.basicConfig(level=logging.DEBUG) # set logging levels, e.g. logging.DEBUG or logging.INFO

#%%

log = logging.getLogger("Main")

# Make output directory
if reslice and not os.path.exists(out_dir):
    log.debug("Making output directory {}".format(out_dir))
    os.mkdir(out_dir)

if make_MIP and not os.path.exists(MIP_dir):
    log.debug("Making MIP output directory {}".format(MIP_dir))
    os.mkdir(MIP_dir)
    
if make_MIP_tif and not os.path.exists(MIP_tif_dir):
    log.debug("Making MIP tif output directory {}".format(MIP_tif_dir))
    os.mkdir(MIP_tif_dir)

# find and sort files by channels and FOVs
d = {}
FOV_list = set() # Make FOV list as set so double entries don't occur

log.debug("Sorting files to process...")
for channel in channels_to_process:
    files = list( in_dir.glob('*{}*.tif'.format(channel)) )
    files.sort()
    d[channel] = {}
    
    for file in files:
        file = file.as_posix()
        if file.endswith('_resliced.tif'): # Avoid double reslicing
            continue
        FOV = re.search(FOV_search_term, file)
        FOV = FOV.group(0)
        
        if checkpoints:
            checkpoint = re.search(checkpoint_search_term, file)
            checkpoint = checkpoint.group(1)
            if FOV_before_checkpoint:
                file_key = '{}_{}'.format(FOV, checkpoint)
            else:
                file_key = '{}_{}'.format(checkpoint, FOV)            
        else:
            file_key = FOV
        
        d[channel][file_key] = file
        FOV_list.add(file_key)

log.debug("...done.")        

#%% Process FOVs

for FOV in FOV_list:    # loop over FOVs
    log.debug("Processing {}".format(FOV))
    stack_list = []
    for channel in channels_to_process:
        log.debug("Reading {} of {}".format(channel, FOV))
        fname_in = d[channel][FOV]
        stack = tf.imread(fname_in)
        # sometimes output of tf.imread has 4 dimensions
        if len(stack.shape) > 3 and stack.shape[0] == 1:
            stack = stack[0]
        log.debug("...done.")
        
        # bin stack
        if bin_factor > 1:
            stack = bin_image_stack_xy(stack, binning_factor=bin_factor)
            step_xy = step_xy_orig*bin_factor
        else:
            step_xy = step_xy_orig
            
        # reslice stack
        if reslice:
            fname_out = out_dir / os.path.basename(fname_in).replace('.tif', '_resliced.tif')
            fname_out = fname_out.as_posix()
            log.debug('Resliced stack file: {}'.format(fname_out))
            # Reslice:
            tdct_reslice(stack, step_z, step_xy, interpolationmethod='linear', save_img=True,
               fname_out=fname_out)
        # add to stack list
        stack_list.append(stack) # stack list contains all stacks before reslicing (for MIP)
    
    if make_MIP:
        # Make plot
        fig, ax = plot_MIP(stack_list, color_list, axis=0, mode=MIP_mode, clim_factor=hist_clip_factor)
           
        MIP_file = MIP_dir / '{}_MIP_decon.png'.format(FOV)
        MIP_file = MIP_file.as_posix()
       
        # Save MIP as png
        fig.savefig(MIP_file, dpi=300, bbox_inches='tight')        
        plt.close()
        
    if make_MIP_tif:
        # Make MIPs from stacks
        MIP_array = np.array([np.max(s, axis=0) for s in stack_list]).astype(stack.dtype)
        lut_list = [lut_from_base_color(col) for col in color_list]
        # Assemble metadata
        dict_metadata = {'unit': 'um', 'axes': 'CYX', 'mode':'composite', 'LUTs': lut_list}
        resolution = (1/(step_xy*step_unit_um), 1/(step_xy*step_unit_um))
        # file name
        MIP_tif_file = MIP_tif_dir / f'{FOV}_MIP_decon.tif'

        # Write image    
        tf.imwrite(MIP_tif_file, MIP_array, photometric='minisblack', imagej=True, 
                   resolution = resolution,
                   metadata = dict_metadata)