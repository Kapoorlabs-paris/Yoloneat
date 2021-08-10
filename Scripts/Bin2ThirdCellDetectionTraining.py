#!/usr/bin/env python
# coding: utf-8

# In[12]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATStatic, nets
from NEATModels.Staticconfig  import static_config
from NEATUtils import helpers
from NEATUtils.helpers import save_json, load_json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[13]:


npz_directory = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/oneatnpz/'
npz_name = 'Staticbin2V1.npz'
npz_val_name = 'Staticbin2V1Val.npz'


model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/CellNetBinning1Models/'
#Model name based on wether it is residual or sequntial ONEAT network
model_name = 'CellNetbin2d20s3seqf16.h5'


# In[14]:


static_categories_json = model_dir + 'StaticCategories.json'
key_categories = load_json(static_categories_json)
static_cord_json = model_dir + 'StaticCord.json'
key_cord = load_json(static_cord_json)

#For ORNET use residual = True and for OSNET use residual = False
residual = False
#NUmber of starting convolutional filters, is doubled down with increasing depth
startfilter = 16
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 7
mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 20
#Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
learning_rate = 1.0E-3
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 100
# use softmax for single event per box, sigmoid for multi event per box
multievent = False
# Trainng image size
yolo_v0 = False
show = False
#Training epochs, longer the better with proper chosen learning rate
epochs = 250
nboxes = 1
#The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
stage_number = 3
last_conv_factor = 4


# In[15]:


config = static_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                         key_categories = key_categories, key_cord = key_cord, stride = last_conv_factor,
                         residual = residual, depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel,
                         startfiler = startfilter, nboxes = nboxes, gridx = 1, gridy = 1, show = show,
                         epochs = epochs, learning_rate = learning_rate,stage_number = stage_number, last_conv_factor = last_conv_factor,
                         batch_size = batch_size, model_name = model_name, yolo_v0 = yolo_v0, multievent = multievent)

config_json = config.to_json()

print(config)
save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')


# In[ ]:


static_model = NEATStatic(config, model_dir, model_name)

static_model.loadData()

static_model.TrainModel()


# In[ ]:





# In[ ]:




