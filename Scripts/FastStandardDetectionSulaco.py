#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
sys.path.append("../NEAT")
from NEATModels import NEATDynamicSegFree, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json

from csbdeep.models import Config, CARE

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir =  '/data/u934/service_imagerie/v_kapoor/WildTypeTest/wt12/'
model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/Binning2V1Models/'
savedir= '/data/u934/service_imagerie/v_kapoor/WildTypeTest/wt12/Oldd29f32/'

model_name = 'micronetbin2d38f32'
mask_name = '_Mask'
division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamicSegFree(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (4,4)
event_threshold = 0.999 #[1,0.999,0.999,0.99,0.99,0.99]
iou_threshold = 0.3
yolo_v2 = False
downsample = 2

# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
X = sorted(X)
marker_dict = {}
for imagename in X:
   
     maskimagename = imagename + mask_name
     model.predict(imagename, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, downsample = downsample, maskimagename = maskimagename)


# In[3]:




# In[ ]:




