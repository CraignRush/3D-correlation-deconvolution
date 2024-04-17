#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:41:52 2024

@author: anbieber
"""
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import tifffile as tf
from scipy import interpolate
from matplotlib.colors import ListedColormap, rgb2hex, to_rgb
from itertools import cycle

#%% Colormap-related functions

def lut_from_base_color(col_rgb=(1,1,1), dtype=np.uint8):
    """Make a colormap for ImageJ using the RGB base color.
    The resulting LUT can be used to save images with tifffile that are displayed with correct LUT when opened in ImageJ."""
    # Check type of color
    if type(col_rgb) == str:
        col_rgb = to_rgb(col_rgb)    
    # Intensity value range
    val_max = 256
    val_range = np.arange(val_max, dtype=dtype)
    # Construct a (3,256) array corresponding to a gray lut
    lut_gray = np.tile(val_range, (3,1))
    # Change the color
    lut_res = np.multiply(col_rgb, lut_gray.T).T.astype(dtype) # Setting dtype is important so correct color is shown!
    return lut_res

def cmap_fading(col_rgb=(0,1,0)):
    """Make a colormap for displaying fluorescence images with matplotlib.""" 
    if np.max(col_rgb) > 1:
        c = [val/255 for val in col_rgb] # adjust range
    else:
        c = list(col_rgb)
    colors = np.zeros((1000, 4), np.float64)
    for i, val in enumerate(c):
        colors[:,i] = val # set RGB to base color
    colors[:,3] = np.linspace(0, 1, colors.shape[0]) # make alpha gradient
    name = 'fading_{}'.format(rgb2hex(c))
    cmap = ListedColormap(colors, name=name)
    
    return cmap

def plot_MIP(stack_list, color_list, axis=0, mode='max', clim_factor=1, return_image_list=False, fig_kwargs={}):
    "Make maximum intensity projection along the given axis, overlaying channels."
    if color_list is None:
        color_list = ['magenta', 'green', 'cyan']
    # Check stack and color list
    if not isinstance(stack_list, (list, tuple)):
        stack_list = [stack_list,]
        color_list = [color_list,]
    # Check color list
    color_list = [to_rgb(c) for c in color_list] # turn any type of color input into rgb
    color_list = cycle(color_list)
    
    # Make the figure
    fig, ax = plt.subplots(**fig_kwargs)
    # plot a black base
    ax.imshow(np.ones([s for i,s in enumerate(stack_list[0].shape) if i != axis]),
              cmap=plt.cm.Greys_r)
    # Plot the Maximum intensity projections
    if return_image_list: 
        im_list = []
    for stack, color in zip(stack_list, color_list):
        assert stack.ndim == 3, "Wrong stack dimension"
        # Make the projection
        if mode == 'max':
            img = np.max(stack, axis=axis)
        elif mode == 'sum':
            img = np.sum(stack, axis=axis)
        # Set the colormap
        clim = img.max()*clim_factor
        cmap = cmap_fading(color)
        # Plot
        im = ax.imshow(img, cmap=cmap, clim = [0,clim])
        if return_image_list:
            im_list.append(im)
    # Clean up figure
    ax.axis('off')
    fig.set_facecolor("black")
    
    if return_image_list:
        return fig, ax, im_list
    return fig, ax
        
#%% Binning-related functions

def rebin(arr, new_shape):
    """Change binning of an array, taken from https://scipython.com/blog/binning-a-2d-array-in-numpy/."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def bin_image_stack_xy(stack, binning_factor=2):
    """Assumes that first axis in stack is the z axis."""
    shape_old = stack.shape
    shape_new = [int(a/binning_factor) if i > 0 else a for i, a in enumerate(shape_old)]
    shape_tmp = (shape_new[0],
                 shape_new[1], shape_old[1] // shape_new[1],
                 shape_new[2], shape_old[2] // shape_new[2])
    stack_new = stack.reshape(shape_tmp).mean(-1).mean(2)
    return stack_new.astype(stack.dtype)

#%% Reslicing-related functions: the functions below are adapted from the 3DCT

log = logging.getLogger("Reslice")

def tdct_reslice(img, step_in, step_out, interpolationmethod='linear', save_img=False, fname_out='resliced.tif'):
    """Main function handling the file type and parsing of filenames/directories"""

    if len(img.shape) < 3:
        log.error( "ERROR: This seems to be a 2D image with the shape {0}. Please select a stack image file.".format(img.shape))
        return

    ## Start Processing
    log.debug( "Interpolating...")
    # Actual interpolation
    img_int = interpol(img, step_in, step_out, interpolationmethod)

    if type(img_int) == str:
        log.debug( img_int)
        return
    if img_int is not None:
        if save_img:
            log.debug( "Saving interpolated stack as: {}".format(fname_out) )
            tf.imsave(fname_out, img_int)            
        log.debug( "        ...done.")
        return img_int



def interpol(img, step_in, step_out, interpolationmethod):
    """Main function for interpolating image stacks via polyfit"""
    ## Depending on tiff format the file can have different shapes; e.g. z,y,x or c,z,y,x
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    elif len(img.shape) == 4 and img.shape[0] > 1:
        return "ERROR: I'm sorry, I cannot handle multichannel files: "+str(img.shape)

    if len(img.shape) == 3:
        ## Number of slices in original stack
        sl_in = img.shape[0]
        ## Number of slices in interpolated stack
        # Discarding last data point. e.g. 56 in i.e.
        # 55 steps * (309 nm original spacing / 161.25 nm new spacing) = 105.39 -> int() = 105 + 1 = 106
        sl_out = int((sl_in-1)*(step_in/step_out)) + 1
        ## Interpolate image stack shape
        img_int_shape = (sl_out, img.shape[1], img.shape[2])
    else:
        return "ERROR: I only know tiff stack image formats in z,y,x or c,z,y,x with one channel"

    if interpolationmethod == 'none':
        return None
    elif interpolationmethod == 'linear':
        log.debug( "Nr. of slices: {} in, {} out): ".format(sl_in, sl_out) )
        return interpol_linear(img, img_int_shape, step_in, step_out, sl_in, sl_out)
    elif interpolationmethod == 'spline':
        log.debug( "Nr. of slices: {} in, {} out): ".format(sl_in, sl_out) )
        return interpol_spline(img, img_int_shape, step_in, step_out, sl_in, sl_out)
    else:
        return "Please specify the interpolation method ('linear', 'spline', 'none')."

def interpol_spline(img, img_int_shape, step_in, step_out, sl_in, sl_out):
    """
    Spline interpolation

    # possible depricated due to changes in code -> marked for futur code changes
    step_in : step size input stack
    step_out : step size output stack
    sl_in : slices input stack
    sl_out : slices output stack
    """
    ## Known x values in interpolated stack size.
    zx = np.arange(0,sl_out,step_in/step_out)
    zxnew = np.arange(0, (sl_in-1)*step_in/step_out, 1)  # First slice of original and interpolated are both 0. n-1 to discard last slice
    if step_in/step_out < 1.0:
        zx_mod = []
        for i in range(img.shape[0]):
            zx_mod.append(zx[i])
        zx = zx_mod

    ## Create new numpy array for the interpolated image stack
    img_int = np.zeros(img_int_shape,img.dtype)
    log.debug( "Interpolated stack shape: {}".format(img_int.shape) )

    r_sl_out = list(range(sl_out))

    ping = time.time()
    for px in range(img.shape[-1]):
        for py in range(img.shape[-2]):
            spl = interpolate.InterpolatedUnivariateSpline(zx, img[:,py,px])
            np.put(img_int[:,py,px], r_sl_out, spl(zxnew))
        sys.stdout.write("\r%d%%" % int(px*100/img.shape[-1]))
        sys.stdout.flush()
    pong = time.time()
    log.debug( "This interpolation took {0} seconds".format(pong - ping))
    return img_int


def interpol_linear(img, img_int_shape, step_in, step_out, sl_in, sl_out):
    """Linear interpolation"""
    ##  Determine interpolated slice positions
    sl_int = np.arange(0,sl_in-1,step_out/step_in)  # sl_in-1 because last slice is discarded (no extrapolation)

    ## Create new numpy array for the interpolated image stack
    img_int = np.zeros(img_int_shape,img.dtype)
    log.debug( "Interpolated stack shape: {} ".format(img_int.shape) )

    ## Calculate distances from every interpolated image to its next original image
    sl_counter = 0
    ping = time.time()
    for i in sl_int:
        int_i = int(i)
        lower = i-int_i
        upper = 1-(lower)
        img_int[sl_counter,:,:] = img[int_i,:,:]*upper + img[int_i+1,:,:]*lower
        sl_counter += 1
    pong = time.time()
    log.debug( "This interpolation took {0} seconds".format(pong - ping))
    return img_int
