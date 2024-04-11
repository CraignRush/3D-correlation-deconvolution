################################################################################################################

# Set directory of image stack
file_pattern = '/fs/pool/pool-plitzko3/Johann/03-Data/04-LightMicroscopy/EMBO_course_decon_testing/RAGE_01.lif'#
#'/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240402_Yeast_CA_GE_05/CA_GE_05_tiles.lif'
output_folder = '/fs/pool/pool-plitzko3/Johann/03-Data/04-LightMicroscopy/EMBO_course_decon_testing/decon_flowdec/'
#'/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240402_Yeast_CA_GE_05/'#
LIFFILE = True

##########################################################








# %%
import logging, sys
LOG_LEVEL = logging.INFO
logging.basicConfig(filename='current.log',encoding='utf-8',level=LOG_LEVEL, filemode = 'w', format='%(asctime)s-%(levelname)s: Process %(process)d said: %(message)s')
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_MAX_THREADS'] = '256'

import tensorflow as tf    
gpus = tf.config.experimental.list_physical_devices('GPU')

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
logging.info("Current File: {}".format(file_pattern))



# %%
# Import LIF data with a custom written handler class here
# FIXME Comment class properly
#HACK Loop over all FOVs
#HACK handle exceptions in class
#TODO calibrate images
#TODO in the very future optimize import, this takes ages!!!
#TODO map to overview


if LIFFILE:
    test_fov = FOV(file_pattern,0)
    logging.info(test_fov.FOV_name)

# %%
#HACK loop over channels

for fov_num in range(test_fov.FOV_count):    
    test_fov = FOV(file_pattern,fov_num)
    logging.info(test_fov.FOV_name)

    test_stack = test_fov.get_channel_stack(channel_num=range(test_fov.num_channels))
    logging.info(test_stack.shape)

    if len(test_stack.shape) == 4:
        MULTICHANNEL = True
    else:
        MULTICHANNEL = False
    processing_stack = test_stack
    logging.info('Input stack shape: {}, dtype: {}'.format(processing_stack.shape,processing_stack.dtype))

    # %%
    size_x,size_y,size_z = 256,256,256
   
    if LIFFILE:
        psf_var = tfd_psf.GibsonLanni(
            na          = test_fov.NA,                           # Numerical aperture
            m           = test_fov.mag,                          # Magnification
            ni0         = 1.0,                                   # Immersion RI
            res_lateral = float(test_fov.resolution[test_fov.resolution['dimension_name'] == 'x']['resolution_nm'].values[0] / 1000) , # X/Y resolution
            res_axial   = float(test_fov.resolution[test_fov.resolution['dimension_name'] == 'z']['resolution_nm'].values[0] / 1000), # Axial resolution
            wavelength  = float(test_fov.channels[0]['center_wavelength'] / 1000),  # Emission wavelength 
            size_x      = int(np.max((size_x, int(processing_stack.shape[-1])))), 
            size_y      = int(np.max((size_y, int(processing_stack.shape[-2])))), 
            size_z      = int(np.min((size_z, int(processing_stack.shape[-3])))),
            ns          = 1.0,                                     # specimen refractive index (RI)
            ng0         = 1.0,                                     # Refractive index of coverslip
            ti0         = float(test_fov.working_distance_mm * 1000),     # microns, working distance (immersion medium thickness) design value
            tg0         = 0,                                     # microns, coverslip thickness design value
        )
        psf_var.save('./current_psf.json')
        psf = psf_var.generate().astype(np.uint8)
        logging.info('PSF shape: {}, PSF dtype: {}'.format(psf.shape, psf.dtype))
        logging.info('PSF: ' + psf_var.to_json())
    else:
    # This is meant to be representative of the Leica SP8 50x widefield image capture (all distance units are in microns)
        psf = np.zeros_like(processing_stack)
        psf = tfd_psf.GibsonLanni(
            na=0.9,           # Numerical aperture
            m=52.5,             # Magnification
            ni0=1.,         # Immersion RI
            res_lateral=0.085, # X/Y resolution
            res_axial=0.3,     # Axial resolution
            wavelength=0.58,  # Emission wavelength 
            size_x=np.max((size_x, int(processing_stack.shape[2]))), 
            size_y=np.max((size_y, int(processing_stack.shape[1]))), 
            size_z=np.min((size_z, int(processing_stack.shape[0]))),
            ns          = 1,                                     # specimen refractive index (RI)
            ng0         = 1,                                     # Refractive index of coverslip
            ti0         =  280,     # microns, working distance (immersion medium thickness) design value
            tg0 = 0,
        ).generate()
        logging.info((psf.shape, psf.dtype))

    # %%
    # In case your GPU setup allows for a continuous oberserver or saving of intermediate steps (requires more memory, and is not used here)
    imgs = []
    scores = {}
    def observer(img, i, *args):
        imgs.append(img)
        scores[i] = {
        'mse': mean_squared_error(processing_stack, img),
        #'ssim': structural_similarity(processing_stack, img, data_range=1), #@TODO find out why SSIM doesn't work as expected
        'psnr': peak_signal_noise_ratio(processing_stack, img)
        }
        
        if i % 1 == 0:
            if i == 1:
                logging.info('Observing iteration = {} (dtype = {}, max = {:.3f})'.format(i, img.dtype, img.max()))        
            else:            
                #logging.info('Observing iteration = {} (MSE = {:.2f},SSIM = {:.2f}, PSNR = {:.2f})'.format(i, scores[i]['mse'],scores[i]['ssim'],scores[i]['psnr']))        
                logging.info('Observing iteration = {} (MSE = {:.2f}, PSNR = {:.2f})'.format(i, scores[i]['mse'],scores[i]['psnr']))        


# %%
# Run the deconvolution process and note that deconvolution initialization is best kept separate from 
# execution since the "initialize" operation corresponds to creating a TensorFlow graph, which is a 
# relatively expensive operation and should not be repeated across multiple executions
#
# In case you experience CUDA memory errors, remove the observer and add pad_mode='none'

    niter = 50
    logging.info("Starting GPU decon!")
    algo = tfd_restoration.RichardsonLucyDeconvolver(n_dims=3).initialize()#,observer_fn=observer
    res = np.array([algo.run(tfd_data.Acquisition(data=processing_stack[ch],kernel=psf), niter=niter) for ch in range(processing_stack.shape[0])])
    logging.info("Finished successfully!")

    # %%
    if LIFFILE:
        from datetime import datetime    
        for i in range(res.shape[0]):
            output_path = output_folder + test_fov.FOV_name + '_ch{:2d}'.format(i) + '_decon.tif'
            io.imsave(output_path,res[i].data.astype(np.float16))
            logging.info('Saved file under: {}'.format(output_path))
        if LOG_LEVEL == logging.DEBUG:
            io.imsave(output_folder + '_' + test_fov.FOV_name +'_input.tif',processing_stack)
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name + '_decon.tif',res.data.astype(np.float16))
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name + '_input.tif',processing_stack)
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name + '_MIP_decon.tif',np.max(res.data.astype(np.float16),axis=0))
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name +'_MIP_input.tif',np.max(processing_stack,axis=0))


# %%
if LOG_LEVEL == logging.DEBUG:
    import pandas as pd
    pd.DataFrame(scores).T.plot(subplots=True, figsize=(18, 8))


    # %%
    # Visualize the iterations
    n = 32
    iters = np.linspace(0, niter-1, num=n).astype(int)
    fig, axs = plt.subplots(4, 8)
    axs = axs.ravel()
    fig.set_size_inches(24, 12)
    for i, j in enumerate(iters):
        axs[i].axis('off')
        axs[i].set_title('iteration {}'.format(j))
        axs[i].imshow(imgs[j].max(axis=0))

    # %%
    fig, axs = plt.subplots(1, 2)
    axs = axs.ravel()
    fig.set_size_inches(30, 15)
    center = tuple([slice(None), slice(10, -10), slice(10, -10)])
    titles = ['Original Image', 'Deconvolved Image']
    for i, d in enumerate([processing_stack, res.data ]):#res.data
        img = exposure.adjust_gamma(d[center].max(axis=0), gamma=.2)
        axs[i].imshow(img, cmap='Spectral_r')
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    # %%
    fig, axs = plt.subplots(1, 2)
    axs = axs.ravel()
    fig.set_size_inches(30, 30)
    xz_projection_data = np.max(processing_stack, axis=1)
    xz_projection_decon = np.max(res.data, axis=1)
    titles = ['Original Image', 'Deconvolved Image']
    for i, d in enumerate([xz_projection_data, xz_projection_decon]):#res.data
        img = exposure.adjust_gamma(d,gamma=.2)
        axs[i].imshow(img, cmap='Spectral_r')
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    # %%
    # Function to plot a selectable image from the stack

    def plot_image(i):
        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
        ax1.imshow(exposure.adjust_gamma(processing_stack[i, slice(10, -10), slice(10, -10)],gamma=.2), cmap='Spectral_r')#)'gray'
        ax2.imshow(exposure.adjust_gamma(res.data[i, slice(10, -10), slice(10, -10)],gamma=.2), cmap='Spectral_r')#'Spectral_r')'gray'
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




