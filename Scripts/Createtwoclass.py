#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tifffile import imread 
import sys
import os
import cv2
import glob
from tqdm import tqdm
import pandas as pd
sys.path.append("../NEAT")
from NEATUtils import MovieCreator
from NEATUtils.helpers import save_json, load_json
from NEATModels.TrainConfig import TrainConfig
from pathlib import Path


# In[ ]:


#Specify the directory containing images
image_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/oneatimages/bin2/'
#Specify the directory contaiing csv files
csv_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/oneatcsv/divisionbin2V1/'
#Specify the directory containing the segmentations
seg_image_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/oneatimages/bin2/segmentation/'
#Specify the model directory where we store the json of categories, training model and parameters
model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/Microscope2V1Models/'
#Directory for storing center ONEAT training data for static classes
save_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/tshift2miniMicrodata/'
Path(model_dir).mkdir(exist_ok = True)
Path(save_dir).mkdir(exist_ok = True)


# In[ ]:


#Name of the static events
event_type_name = ["Normal","Division"]
#Label corresponding to static event
event_type_label = [0, 1]

#The name appended before the CSV files
csv_name_diff = 'ONEAT'
#with xythw and class terms only
yolo_v0 = False
size_tminus = 2
size_tplus = 0
tshift = 2
trainshapeX = 64
trainshapeY = 64
axes= 'STXYC'
npz_name = 'tshift2minimicrodata'
npz_val_name = 'tshift2minimicrodataval'
crop_size = [trainshapeX,trainshapeY,size_tminus,size_tplus]


# In[ ]:


#X Y T dynamic events




event_position_name = ["x", "y", "t", "h", "w", "c"]
event_position_label = [0, 1, 2, 3, 4, 5]

dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

dynamic_json, dynamic_cord_json = dynamic_config.to_json()

save_json(dynamic_json, model_dir + "MicroscopeCategories" + '.json')

save_json(dynamic_cord_json, model_dir + "MicroscopeCord" + '.json')    


# In[ ]:


MovieCreator.MovieLabelDataSet(image_dir, seg_image_dir, csv_dir, save_dir, event_type_name, event_type_label, csv_name_diff,crop_size, tshift = tshift)


# In[ ]:


MovieCreator.createNPZ(save_dir, axes = 'STXYC', save_name = npz_name, save_name_val = npz_val_name)


# In[ ]:




