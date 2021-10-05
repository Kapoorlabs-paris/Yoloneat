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


imagedir =  '/home/sancere/VKepler/WildTypeTest/wt12/'
maskimagedir =  '/home/sancere/VKepler/WildTypeTest/wt12/Masks/'
model_dir = '/home/sancere/VKepler/CurieDeepLearningModels/OneatModels/MicroscopeV1Models/'
savedir= '/home/sancere/VKepler/WildTypeTest/wt12/HeatMapped/Microd38f32_th20'

model_name = 'micronetbin2d38f32'
mask_name = '_Mask'
division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamicSegFree(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (4,4)
event_threshold = 1 #[1,0.999,0.999,0.99,0.99,0.99]
iou_threshold = 0.3
yolo_v2 = False
downsample = 2
thresh = 20
# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
X = sorted(X)

Mask_path = os.path.join(maskimagedir, '*tif')
Y = glob.glob(Mask_path)
Y = sorted(Y)

marker_dict = {}
for imagename in X:
  Name = os.path.basename(os.path.splitext(imagename)[0])  
  for maskimagename in Y:   
     MaskName = os.path.basename(os.path.splitext(maskimagename)[0]) 
     
     if MaskName == Name + mask_name:
          print(MaskName, Name, mask_name)
          model.predict(imagename, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, downsamplefactor = downsample, thresh = thresh, maskimagename = maskimagename)


# In[3]:




# In[ ]:




