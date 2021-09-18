#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
sys.path.append("../NEAT")
from NEATModels import NEATMasterDynamicSegFree, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json

from csbdeep.models import Config, CARE

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir =  '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/'
model_dir = '/home/sancere/VKepler/CurieDeepLearningModels/OneatModels/MicroscopeV1Models/'
savedir= '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/MasterMicrod29f32/'

model_name = 'mastermicronetbin2d29f32'

division_categories_json = model_dir + 'MasterMicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MasterMicroscopeCord.json'
cordconfig = load_json(division_cord_json)
model = NEATMasterDynamicSegFree(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (4,4)
event_threshold = [1,0.99999,0.9999,0.999,0.999,0.999]
iou_threshold = 0.3
yolo_v2 = False


# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
X = sorted(X)
marker_dict = {}
for imagename in X:
   
     
     model.predict(imagename, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, thresh = 10)


# In[3]:




# In[ ]:




