from ..NEATUtils import plotters
import numpy as np
from ..NEATUtils import helpers
from ..NEATUtils.helpers import get_nearest, save_json, load_json, yoloprediction, normalizeFloatZeroOne, GenerateMarkers, \
    DensityCounter, MakeTrees, nonfcn_yoloprediction, fastnms, averagenms, DownsampleData, save_dynamic_csv, dynamic_nms
from keras import callbacks
import os
import math
import tensorflow as tf
from tqdm import tqdm
from ..NEATModels import nets
from ..NEATModels.nets import Concat
from ..NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
# from IPython.display import clear_output
from keras import optimizers
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv
import napari
import glob
from scipy import spatial
import itertools
from napari.qt.threading import thread_worker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import h5py
import cv2
import imageio

Boxname = 'ImageIDBox'
EventBoxname = 'EventIDBox'
from .neat_goldstandard import NEATDynamic

class NEATDynamicSeg(NEATDynamic):
   

    def __init__(self, config, model_dir, model_name, catconfig=None, cordconfig=None):

                super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)

    

    

    def predict_standard(self, imagename, markers, marker_tree, density_location, savedir, n_tiles=(1, 1), overlap_percent=0.8,
                event_threshold=0.5, iou_threshold=0.1, density_veto=5, downsamplefactor = 1):

        self.imagename = imagename
        self.image = imread(imagename)
        self.Colorimage = np.zeros_like(self.image)
        self.markers = markers
        self.marker_tree = marker_tree
        self.density_location = density_location
        self.savedir = savedir
        self.n_tiles = n_tiles
        self.density_veto = density_veto
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.downsample_regions = {}
        self.upsample_regions = {}
        self.candidate_regions = {}
        self.downsamplefactor = downsamplefactor
        self.image = DownsampleData(self.image, self.downsamplefactor)
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate", "lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model = load_model(self.model_dir + self.model_name + '.h5',
                                custom_objects={'loss': self.yololoss, 'Concat': Concat})

        self.fast_pass_predict()

    def fast_pass_predict(self):

        eventboxes = []
        classedboxes = {}
        count = 0
        heatsavename = self.savedir+ "/"  + (os.path.splitext(os.path.basename(self.imagename))[0])+ '_Heat'
        print('Detecting event locations')
        for inputtime in tqdm(range(0, self.image.shape[0])):
            if inputtime < self.image.shape[0] - self.imaget:

                eventboxes = []
                tree, indices = self.marker_tree[str(int(inputtime))]

                down_region = []
                up_region = []
                # all_density_location = self.density_location[str(inputtime)]
                # density = all_density_location[0]
                # locations = all_density_location[1]

                count = count + 1
                if inputtime%10==0 or inputtime >= self.image.shape[0] - self.imaget - 1:
                                      
                        imwrite((heatsavename + '.tif' ), self.heatmap)

                smallimage = CreateVolume(self.image, self.imaget, inputtime)
                smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
                # Cut off the region for training movie creation
                # for i in range(len(density)):

                # if density[i] <= self.density_veto:
                # down_region.append(location)
                # self.remove_marker_locations(inputtime, location)
                # if density[i] >= 5 * self.density_veto:
                # up_region.append(location)
                # self.remove_marker_locations(inputtime, location)

                self.downsample_regions[str(inputtime)] = down_region
                self.upsample_regions[str(inputtime)] = up_region
                # Cut off the region for training movie creation
                # Break image into tiles if neccessary
                
                
                
                
                predictions, allx, ally = self.predict_main(smallimage)
                # Iterate over tiles
                for p in range(0, len(predictions)):

                    sum_time_prediction = predictions[p]

                    if sum_time_prediction is not None:
                        # For each tile the prediction vector has shape N H W Categories + Training Vector labels
                        for i in range(0, sum_time_prediction.shape[0]):
                            time_prediction = sum_time_prediction[i]
                            boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride,
                                                           inputtime, self.config,
                                                           self.key_categories, self.key_cord, self.nboxes, 'detection',
                                                           'dynamic', marker_tree=self.marker_tree)

                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction

                for (event_name, event_label) in self.key_categories.items():

                    if event_label > 0:
                        current_event_box = []
                        for box in eventboxes:

                            event_prob = box[event_name]
                            event_confidence = box['confidence']
                            if event_prob >= self.event_threshold and event_confidence >= 0.9:
                                current_event_box.append(box)
                        classedboxes[event_name] = [current_event_box]

                self.classedboxes = classedboxes
                self.eventboxes = eventboxes
                # nms over time
                if inputtime % (self.imaget) == 0:
                    self.fast_nms()
                    self.to_csv()
                    eventboxes = []
                    classedboxes = {}
                    count = 0

    def remove_marker_locations(self, tcenter, location):

        tree, indices = self.marker_tree[str(int(round(tcenter)))]

        # if location in indices:
        if location in indices:
            indices.remove(location)

        tree = spatial.cKDTree(indices)

        # Update the tree
        self.marker_tree[str(int(tcenter))] = [tree, indices]

   
 



def CreateVolume(patch, imaget, timepoint):
    starttime = timepoint
    endtime = timepoint + imaget
    smallimg = patch[starttime:endtime, :]

    return smallimg

