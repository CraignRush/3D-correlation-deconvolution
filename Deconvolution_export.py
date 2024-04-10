# %%
import logging, sys
LOG_LEVEL = logging.DEBUG
logging.basicConfig(filename='current.log',encoding='utf-8',level=LOG_LEVEL, filemode = 'w', format='%(asctime)s-%(levelname)s: Process %(process)d said: %(message)s')
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_MAX_THREADS'] = '256'

import tensorflow as tf
# @FIXME Maybe listen to the debug msg and increase the thread count?

## Check if Tensorflow really runs on your GPU
# refer first to the README (!), then to this article, especially if you are using a windows machine https://www.tensorflow.org/install/pip#windows-wsl2_1
#tf_config = tf.config.list_physical_devices('GPU')
#logging.debug("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
#logging.debug(tf_config)
    
gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     #tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logging.debug(tf.config.experimental.set_memory_growth(gpus[0], True))
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     logging.debug((len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"))
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     logging.debug(e)


# %%
#HACK fix import and order
from skimage import exposure, io
from flowdec import data as tfd_data
from flowdec import psf as tfd_psf
from flowdec import restoration as tfd_restoration
from skimage.transform import rescale
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from pprint import pprint

from FOV import FOV

# %%
# Microscope Parameters from the Arctis 100x iFLM
# Image properties
# Size of the PSF array, pixels
size_x = 256
size_y = 256
size_z = 128
#
# # Microscope parameters
NA          = 0.9
wavelength  = 0.588 # microns
M           = 100   # magnification
ns          = 1.0  # specimen refractive index (RI)
ng0         = 1.0   # coverslip RI design value
ni0         = 1.0   # immersion medium RI design value
ti0         = 3000   # microns, working distance (immersion medium thickness) design value
tg0         = 0   # microns, coverslip thickness design value
res_lateral = 0.075   # microns
res_axial   = 0.25  # microns

# ng          = 1.5   # coverslip RI experimental value (defaults to ng0 if not given)
# ni          = 1.5   # immersion medium RI experimental value (defaults to ni0 if not given)
# tg          = 170   # microns, coverslip thickness experimental value (defaults to tg0 if not given)
# pZ          = 2     # microns, particle distance from coverslip
#
# # Precision control
# num_basis    = 100  # Number of rescaled Bessels that approximate the phase function
# num_samples  = 1000 # Number of pupil samples along radial direction
# oversampling = 2    # Defines the upsampling ratio on the image space grid for computations

# %%
# Set directory of image stack
# sudo mount -t drvfs '//samba-pool-pool-plitzko3.biochem.mpg.de/pool-plitzko3' /mnt/plitzko3
file_pattern = '/fs/pool/pool-plitzko3/Johann/03-Data/04-LightMicroscopy/EMBO_course_decon_testing/RAGE_01.lif'
output_folder = '/fs/pool/pool-plitzko3/Johann/03-Data/04-LightMicroscopy/EMBO_course_decon_testing/'
LIFFILE = True
logging.debug("Current File: {}".format(file_pattern))

# %%
# Import LIF data with a custom written handler class here
# FIXME Comment class properly
#HACK Loop over all FOVs
#HACK handle exceptions in class
#TODO calibrate images
#TODO in the very future optimize import, this takes ages!!!
#TODO map to overview


if LIFFILE:
    test_fov = FOV(file_pattern,2)
    logging.debug(test_fov.print())

# %%
#HACK loop over channels
if LIFFILE:
    test_stack = test_fov.get_channel_stack(channel_num=1)
    logging.debug(test_stack.shape)

# %%
if LIFFILE:
    if len(test_stack.shape) == 4:
        num = test_stack.shape[0]
    else:
        num = 1
    fig,ax = plt.subplots(1,num, figsize=(15,5))
    for chan in range(num):
        if num == 1:
            ax.imshow(np.max(test_stack,axis=0))
            ax.axis('off')
        else:
            ax[chan].imshow(np.max(test_stack[chan],axis=0))
            ax[chan].axis('off')

# %%
# Load image stack for debugging
if not LIFFILE:
    stack = io.imread('input.tif', plugin="tifffile").astype(np.float16)
else:    
    if len(test_stack.shape) == 4:
        stack = test_stack[0]
    else:
        stack = test_stack
logging.debug(stack.shape)
logging.debug(stack.dtype)

# %%
scalexy, scalez = 1, 1
if np.max(scalexy) == 1 and np.max(scalez) == 1:
    stack_scaled = stack
else:
    stack_scaled = rescale(stack, (scalez,scalexy,scalexy), mode='constant', order=2, anti_aliasing=True) 
logging.debug(stack_scaled.shape)

# %%
# This is meant to be representative of the arctis 100x widefield image capture (all distance units are in microns)
if False:
    psf = np.zeros_like(stack_scaled)
    psf = tfd_psf.GibsonLanni(
        na=NA,           # Numerical aperture
        m=M,             # Magnification
        ni0=ni0,         # Immersion RI
        res_lateral=res_lateral, # X/Y resolution
        res_axial=res_axial,     # Axial resolution
        wavelength=wavelength,  # Emission wavelength 
        size_x=np.max((size_x, int(stack_scaled.shape[2]))), 
        size_y=np.max((size_y, int(stack_scaled.shape[1]))), 
        size_z=np.min((size_z, int(stack_scaled.shape[0]))),
        ns = ns,
        ng0 = ng0,
        ti0 = ti0,
        tg0 = tg0,
    ).generate()
    logging.debug((psf.shape, psf.dtype))


# %%
# This is meant to be representative of the Leica SP8 50x widefield image capture (all distance units are in microns)
if not LIFFILE:
    psf = np.zeros_like(stack_scaled)
    psf = tfd_psf.GibsonLanni(
        na=0.9,           # Numerical aperture
        m=52.5,             # Magnification
        ni0=ni0,         # Immersion RI
        res_lateral=0.085, # X/Y resolution
        res_axial=0.3,     # Axial resolution
        wavelength=0.58,  # Emission wavelength 
        size_x=np.max((size_x, int(stack_scaled.shape[2]))), 
        size_y=np.max((size_y, int(stack_scaled.shape[1]))), 
        size_z=np.min((size_z, int(stack_scaled.shape[0]))),
        ns          = 1,                                     # specimen refractive index (RI)
        ng0         = 1,                                     # Refractive index of coverslip
        ti0         =  280,     # microns, working distance (immersion medium thickness) design value
        tg0 = tg0,
    ).generate()
    logging.debug((psf.shape, psf.dtype))

# %%
# This is meant to be representative of the Leica SP8 50x widefield image capture (all distance units are in microns)
if LIFFILE:
    psf_var = tfd_psf.GibsonLanni(
        na          = test_fov.NA,                           # Numerical aperture
        m           = test_fov.mag,                          # Magnification
        ni0         = ni0,                                   # Immersion RI
        res_lateral = test_fov.resolution[test_fov.resolution['dimension_name'] == 'x']['resolution_nm'].values[0] / 1000 , # X/Y resolution
        res_axial   = test_fov.resolution[test_fov.resolution['dimension_name'] == 'z']['resolution_nm'].values[0] / 1000, # Axial resolution
        wavelength  = test_fov.channels[0]['center_wavelength'] / 1000,  # Emission wavelength 
        size_x      = np.max((size_x, int(stack_scaled.shape[2]))), 
        size_y      = np.max((size_y, int(stack_scaled.shape[1]))), 
        size_z      = np.min((size_z, int(stack_scaled.shape[0]))),
        ns          = 1,                                     # specimen refractive index (RI)
        ng0         = 1,                                     # Refractive index of coverslip
        ti0         = test_fov.working_distance_mm * 1000,     # microns, working distance (immersion medium thickness) design value
        tg0         = 0,                                     # microns, coverslip thickness design value
    )
    #psf_var.save('./current_psf.json')
    psf = psf_var.generate()
    logging.debug((psf.shape, psf.dtype))


# %%
# In case your GPU setup allows for a continuous oberserver or saving of intermediate steps (requires more memory, and is not used here)
imgs = []
scores = {}
def observer(img, i, *args):
    imgs.append(img)
    scores[i] = {
    'mse': mean_squared_error(stack_scaled, img),
    #'ssim': structural_similarity(stack_scaled, img, data_range=1), #@TODO find out why SSIM doesn't work as expected
    'psnr': peak_signal_noise_ratio(stack_scaled, img)
    }
    
    if i % 1 == 0:
        if i == 1:
            logging.info('Observing iteration = {} (dtype = {}, max = {:.3f})'.format(i, img.dtype, img.max()))        
        else:            
            #logging.info('Observing iteration = {} (MSE = {:.2f},SSIM = {:.2f}, PSNR = {:.2f})'.format(i, scores[i]['mse'],scores[i]['ssim'],scores[i]['psnr']))        
            logging.info('Observing iteration = {} (MSE = {:.2f}, PSNR = {:.2f})'.format(i, scores[i]['mse'],scores[i]['psnr']))        


acq = tfd_data.Acquisition(stack_scaled,psf)
logging.info(acq.shape())

# %%
# Run the deconvolution process and note that deconvolution initialization is best kept separate from 
# execution since the "initialize" operation corresponds to creating a TensorFlow graph, which is a 
# relatively expensive operation and should not be repeated across multiple executions
#algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim, observer_fn=observer).initialize()
#res = algo.run(fd_data.Acquisition(data=data, kernel=psf), niter=30).data
channels = 0
niter = 20
algo = tfd_restoration.RichardsonLucyDeconvolver(n_dims=3,pad_mode='none', observer_fn=observer).initialize()#,
res = algo.run(acq, niter=niter)
logging.info("Finished successfully!")
# %%
fig, axs = plt.subplots(1, 2)
axs = axs.ravel()
fig.set_size_inches(30, 15)
center = tuple([slice(None), slice(10, -10), slice(10, -10)])
titles = ['Original Image', 'Deconvolved Image']
for i, d in enumerate([stack_scaled, res.data ]):#res.data
    img = exposure.adjust_gamma(d[center].max(axis=0), gamma=.2)
    axs[i].imshow(img, cmap='Spectral_r')
    axs[i].set_title(titles[i])
    axs[i].axis('off')

'''

# %%
fig, axs = plt.subplots(1, 2)
axs = axs.ravel()
fig.set_size_inches(30, 30)
xz_projection_data = np.max(stack_scaled, axis=1)
xz_projection_decon = np.max(res.data, axis=1)
titles = ['Original Image', 'Deconvolved Image']
for i, d in enumerate([xz_projection_data[:,333:666], xz_projection_decon[:,333:666] ]):#res.data
    img = exposure.adjust_gamma(d,gamma=.2)
    axs[i].imshow(img, cmap='Spectral_r')
    axs[i].set_title(titles[i])
    axs[i].axis('off')

# %%
# Function to plot a selectable image from the stack

def plot_image(i):
    fig,(ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.imshow(exposure.adjust_gamma(stack_scaled[i, slice(10, -10), slice(10, -10)]), cmap='Spectral_r')#)'gray'
    ax2.imshow(exposure.adjust_gamma(res.data[i, slice(10, -10), slice(10, -10)]), cmap='Spectral_r')#'Spectral_r')'gray'
    ax2.axis('off')
    fig.show()

# Create a slider widget
slider = widgets.IntSlider(
    value=0,
    min=0,
    max=res.data.shape[0] - 1,
    step=1,
    description='Image Index:',
    continuous_update=True
)

widgets.interactive(plot_image, i=slider) # Create an interactive widget
'''

# %%
if LIFFILE:
    from datetime import datetime    
    io.imsave(output_folder + datetime.today().strftime("%Y-%m-%d_%H-%M-%S") +'_deconv.tif',res.data)
    io.imsave(output_folder + datetime.today().strftime("%Y-%m-%d_%H-%M-%S") +'_input.tif',stack_scaled)

    io.imsave(output_folder + datetime.today().strftime("%Y-%m-%d_%H-%M-%S") +'_MIP_deconv.tif',np.max(res.data,axis=0))
    io.imsave(output_folder + datetime.today().strftime("%Y-%m-%d_%H-%M-%S") +'_MIP_input.tif',np.max(stack_scaled,axis=0))

