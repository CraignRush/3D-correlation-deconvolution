################################################################################################################
# get an interactive bash on 93xx and execute the deconvolution:
# 1. ssh hpcl9301
# 1. Move to a directory where you want to execute the code from
# 1. git clone https://github.com/CraignRush/3D-correlation-deconvonvolution
# 1. srun --nodes=1  --partition=p.hpcl93 --ntasks-per-node=1  --gres=gpu:4  --time=01:00:00 --pty bash -i
# 1. module load FLOWDEC
# 1. which python 
#   --> should yield /fs/pool/pool-bmapps/hpcl8/app/soft/FLOWDEC/12-04-2024/conda3/envs/flowdec/bin/python
# 1. "Plugin the correct paths below"
# 1. python runner.py


# Set directory of image stack
file_pattern = '/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240416_Yeast_CA_GE_18/20240416_Yeast_CA_GE_18_tiles.lif'#
#'/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240402_Yeast_CA_GE_05/CA_GE_05_tiles.lif'
output_folder = '/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240416_Yeast_CA_GE_18/20240416_Yeast_CA_GE_18_decon/'
#'/fs/pool/pool-pub/EMBO/FLM/PreImaged_Yeast/20240402_Yeast_CA_GE_05/'#
LIFFILE = True

##########################################################
filter_sigma = 1.5 # this parameter controls the blurring after deconvolution 
iterations = 100 # this parameters controls the deconvolution iterations
### DON'T MODIFY ANYTHING BELOW HERE ###
# %%
import logging, sys
LOG_LEVEL = logging.INFO
logging.basicConfig(filename='current.log',encoding='utf-8',level=LOG_LEVEL, filemode = 'w', format='%(asctime)s-%(levelname)s: Process %(process)d said: %(message)s')
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))

# %%
import os
if not os.path.isdir(output_folder):
    logging.info("Couldn't find the output directory!")
    sys.exit()

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
from scipy.ndimage import gaussian_filter
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import numpy as np

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

# %%
#HACK loop over channels

for fov_num in range(test_fov.FOV_count):    
    test_fov = FOV(file_pattern,fov_num)
    logging.info('Processing: ' + test_fov.FOV_name)

    processing_stack = test_fov.get_channel_stack(channel_num=range(test_fov.num_channels))

    if len(processing_stack.shape) == 4:
        MULTICHANNEL = True
    else:
        MULTICHANNEL = False
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
        #imgs.append(img)
        #scores[i] = {
        #'mse': mean_squared_error(processing_stack, img),
        #'ssim': structural_similarity(processing_stack, img, data_range=1), #@TODO find out why SSIM doesn't work as expected
        #'psnr': peak_signal_noise_ratio(processing_stack, img)
        #}        
        if i % 5 == 0:
            if i == 5:
                logging.info('Observing iteration = {} (dtype = {}, max = {:.3f})'.format(i, img.dtype, img.max()))        
            else:                   
                logging.info('Observing iteration = {}'.format(i))           
                #logging.info('Observing iteration = {} (MSE = {:.2f},SSIM = {:.2f}, PSNR = {:.2f})'.format(i, scores[i]['mse'],scores[i]['ssim'],scores[i]['psnr']))        
                #logging.info('Observing iteration = {} (MSE = {:.2f}, PSNR = {:.2f})'.format(i, scores[i]['mse'],scores[i]['psnr']))        


# %%
# Run the deconvolution process and note that deconvolution initialization is best kept separate from 
# execution since the "initialize" operation corresponds to creating a TensorFlow graph, which is a 
# relatively expensive operation and should not be repeated across multiple executions
#
# In case you experience CUDA memory errors, remove the observer and add pad_mode='none'

    logging.info("Starting GPU decon!")
    algo = tfd_restoration.RichardsonLucyDeconvolver(n_dims=3,observer_fn=observer).initialize()#
    res = [algo.run(tfd_data.Acquisition(data=processing_stack[ch],kernel=psf), niter=iterations) for ch in range(processing_stack.shape[0])]
    logging.info("Finished successfully!")
    decon_list = [res[i].data for i in range(len(res))]
    decon_stack = np.array(decon_list)    
    logging.info('A new np.array would have the shape: {} with dtype: {}'.format(decon_stack.shape,decon_stack.dtype))

    # %%
    if LIFFILE:
        from datetime import datetime    
        
        output_path_MIP = output_folder + test_fov.FOV_name + '_MIP_decon.tif'
        io.imsave(output_path_MIP,np.max(decon_stack,axis=1))        
        logging.info('Saved MIPs under: {}'.format(output_path_MIP))
        for i in range(decon_stack.shape[0]):
            output_path_stack = output_folder + test_fov.FOV_name + '_ch{:02d}'.format(i) + '_decon.tif'
            filtered_stack = gaussian_filter(np.array(res[i].data),filter_sigma)
            io.imsave(output_path_stack, filtered_stack)
            logging.info('Saved stack under: {}'.format(output_path_stack))
        logging.info('Not resliced yet!')
        if LOG_LEVEL == logging.DEBUG:
            io.imsave(output_folder + '_' + test_fov.FOV_name +'_input.tif',processing_stack)
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name + '_decon.tif',res.data.astype(np.float16))
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name + '_input.tif',processing_stack)
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name + '_MIP_decon.tif',np.max(res.data.astype(np.float16),axis=0))
            io.imsave('./' + datetime.today().strftime("%Y-%m-%d_%H-%M-%S_")  + test_fov.FOV_name +'_MIP_input.tif',np.max(processing_stack,axis=0))

#    break
