#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATDetection, nets
from NEATModels.config import NeatConfig
from NEATUtils import helpers
from NEATUtils.helpers import save_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


NpzDirectory = '/home/sancere/VarunNewton/CurieTrainingDatasets/O-NEAT/'
TrainModelName = 'BigCenterONEATPrediction.npz'
ValidationModelName = 'BigCenterONEATPredictionValidation.npz'

#Read and Write the h5 file, directory location and name
Modeldir = '/home/sancere/VarunNewton/CurieDeepLearningModels/O-NEATweights/'
ModelName = 'BigCenterPredORNETd29K3.h5'

#Neural network parameters
#For ORNET use residual = True and for OSNET use residual = False
residual = True
#Simple model only trains for classification
simple = False
catsimple = True
startfilter = 48
start_kernel = 3
mid_kernel = 3
depth = 29
epochs = 150
learning_rate = 1.0E-4
batch_size = 10
lstm = 16


# In[4]:


config = NeatConfig(startfilter = startfilter, simple = simple, catsimple = catsimple,  start_kernel = start_kernel, mid_kernel = mid_kernel, ModelName = ModelName, residual = residual,
                depth = depth, lstm = lstm, learning_rate = learning_rate, batch_size = batch_size, epochs = epochs)

config_json = config.to_json()
show = False

model_weights = Modeldir + ModelName

if os.path.exists(model_weights):

    model_weights = model_weights
    print('loading weights')
else:
   
    model_weights = None

Categories_Name = []
Categories_Name.append(['Normal', 0])
Categories_Name.append(['Apoptosis', 1])
Categories_Name.append(['Divisions', 2])
Categories_Name.append(['MacroKitty', 3])
Categories_Name.append(['NonMatureP1', 4])
Categories_Name.append(['MatureP1', 5])
print(config)
save_json(config_json, Modeldir + ModelName + 'ParameterFile.json')


# In[ ]:


Train = NEATDetection(config, NpzDirectory, TrainModelName,ValidationModelName, Categories_Name, Modeldir, ModelName, model_weights = model_weights, show = show)

Train.loadData()

Train.TrainModel()


# In[ ]:





# In[ ]:





# In[ ]:




