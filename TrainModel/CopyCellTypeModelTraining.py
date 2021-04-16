#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATStaticDetection, nets
from NEATModels.Staticconfig import StaticNeatConfig
from NEATUtils import helpers
from NEATUtils.helpers import save_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[ ]:


NpzDirectory = '/home/vkapoor/'
TrainModelName = 'MNMDetection.npz'
ValidationModelName = 'MNMDetectionValidation.npz'

#Read and Write the h5 file, directory location and name
Model_dir = '/home/sancere/VarunNewton/CurieDeepLearningModels/O-NEATweights/'
Model_Name = 'StaticOSNetd29K3.h5'

#Neural network parameters
#For ORNET use residual = True and for OSNET use residual = False
residual = False
startfilter = 32
start_kernel = 3
mid_kernel = 3
depth = 29
epochs = 150
learning_rate = 1.0E-4
batch_size = 50


# In[ ]:


config = StaticNeatConfig(startfilter = startfilter, start_kernel = start_kernel, mid_kernel = mid_kernel, ModelName = Model_Name, residual = residual,
                depth = depth,  learning_rate = learning_rate, batch_size = batch_size, epochs = epochs)
config_json = config.to_json()

show = False
model_weights = Model_dir + Model_Name

if os.path.exists(model_weights):

    model_weights = model_weights
    print('loading weights')
else:
   
    model_weights = None

Categories_Name = []
Categories_Name.append(['Normal', 0])
Categories_Name.append(['MacroKitty', 1])
Categories_Name.append(['NonMatureP1', 2])
Categories_Name.append(['MatureP1', 3])
Categories_Name.append(['Apoptosis', 4])
Categories_Name.append(['Divisions', 5])
print(config)
save_json(config_json, Model_dir + 'CellTypeModelParameterFile.json')


# In[ ]:


Train = NEATStaticDetection(config, NpzDirectory, TrainModelName,ValidationModelName, Categories_Name, Model_dir, Model_Name, model_weights = model_weights, show = show)

Train.loadData()

Train.TrainModel()


# In[ ]:





# In[ ]:





# In[ ]:




