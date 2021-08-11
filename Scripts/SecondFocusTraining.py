#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATFocus, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import save_json, load_json
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[ ]:


npz_directory = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/foconeatnpz/'
npz_name = 'Foconeat.npz'
npz_val_name = 'FoconeatVal.npz'

#Read and Write the h5 file, directory location and name
model_dir =  '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/Focusoneatmodels/'
model_name = 'cadhistoned38s4f16seq.h5'


# In[ ]:


#Neural network parameters
focus_categories_json = model_dir + 'FocusCategories.json'
key_categories = load_json(focus_categories_json)
focus_cord_json = model_dir + 'FocusCord.json'
key_cord = load_json(focus_cord_json)

#For ORNET use residual = True and for OSNET use residual = False
residual = False
#NUmber of starting convolutional filters, is doubled down with increasing depth
startfilter = 16
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 7

mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 38
#Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
learning_rate = 1.0E-3
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 4

#Training epochs, longer the better with proper chosen learning rate
epochs = 250
nboxes = 1
#The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
show = False
stage_number = 4
last_conv_factor = 8
size_tminus = 1
size_tplus = 1
imagex = 256
imagey = 256


# In[ ]:



config = dynamic_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                         key_categories = key_categories, key_cord = key_cord,  imagex = imagex,
                         imagey = imagey, size_tminus = size_tminus, size_tplus =size_tplus, epochs = epochs,
                         residual = residual, depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number, last_conv_factor = last_conv_factor,
                         show = show, startfilter = startfilter, batch_size = batch_size, model_name = model_name)

config_json = config.to_json()

print(config)
save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')


# In[ ]:


Train = NEATFocus(config, model_dir, model_name)

Train.loadData()

Train.TrainModel()


# In[ ]:




