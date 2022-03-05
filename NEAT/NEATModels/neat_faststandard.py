#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:49:35 2021

@author: vkapoor
"""

import numpy as np
from ..NEATUtils.helpers import yoloprediction, normalizeFloatZeroOne, DownsampleData, save_dynamic_csv, dynamic_nms
import os
import math
import tensorflow as tf
from tqdm import tqdm
from ..NEATModels.nets import Concat
from keras.models import load_model
from tifffile import imread, imwrite
from scipy import ndimage
from scipy import spatial
import matplotlib.pyplot  as plt
import h5py
from neat_goldstandard import NEATDynamic

class NEATSynamic(NEATDynamic):
    

    """
    Parameters
    ----------
    
    NpzDirectory : Specify the location of npz file containing the training data with movies and labels
    
    TrainModelName : Specify the name of the npz file containing training data and labels
    
    ValidationModelName :  Specify the name of the npz file containing validation data and labels
    
    categories : Number of action classes
    
    Categories_Name : List of class names and labels
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_keras : The model as it appears as a Keras function
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    lstm_hidden_units : Number of hidden uniots for LSTm layer, 64 by default
    
    epochs :  Number of training epochs, 55 by default
    
    batch_size : batch_size to be used for training, 20 by default
    
    
    
    """
    
    
    def __init__(self, config, model_dir, model_name, catconfig = None, cordconfig = None):

        super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)
        
        

        
    def predict(self,imagename, savedir, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, iou_threshold = 0.1, thresh = 5, downsamplefactor = 1, maskimagename = None, maskfilter = 10):
        

        self.predict(imagename,savedir,n_tiles = n_tiles, overlap_percent = overlap_percent, event_threshold = event_threshold, iou_threshold = iou_threshold, 
        thresh = thresh, downsamplefactor = downsamplefactor, maskimagename = maskimagename, maskfilter = maskfilter, density_veto = None, markers = None, marker_tree = None,
        density_location = None, remove_markers = None )
        
                                
                            
    def remove_marker_locations(self, tcenter, location):

                     tree, indices = self.marker_tree[str(int(tcenter))]
                     try:
                        indices.remove(location)
                     except:
                         pass
                     tree = spatial.cKDTree(indices)
                    
                     #Update the tree
                     self.marker_tree[str(int(tcenter))] =  [tree, indices]
            
        
    def nms(self):
        
        
        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               
               #best_sorted_event_box = self.classedboxes[event_name][0]
               best_sorted_event_box = dynamic_nms(self.heatmap,self.maskimage, self.originalimage, self.classedboxes, event_name, event_label, self.downsamplefactor, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, self.imaget, self.thresh )
               
               best_iou_classedboxes[event_name] = [best_sorted_event_box]
               
        self.iou_classedboxes = best_iou_classedboxes                
    

   

    
    def to_csv(self):
        
        save_dynamic_csv(self.imagename, self.key_categories, self.iou_classedboxes, self.savedir, self.downsamplefactor)          
                              
    def overlaptiles(self, sliceregion):
        
             if self.n_tiles == (1, 1):
                               patch = []
                               rowout = []
                               column = []
                               patchx = sliceregion.shape[2] // self.n_tiles[0]
                               patchy = sliceregion.shape[1] // self.n_tiles[1]
                               patchshape = (patchy, patchx) 
                               smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, [0,0])
                               patch.append(smallpatch)
                               rowout.append(smallrowout)
                               column.append(smallcolumn)
                 
             else:     
                     patchx = sliceregion.shape[2] // self.n_tiles[0]
                     patchy = sliceregion.shape[1] // self.n_tiles[1]
                
                     if patchx > self.imagex and patchy > self.imagey:
                          if self.overlap_percent > 1 or self.overlap_percent < 0:
                             self.overlap_percent = 0.8
                         
                          jumpx = int(self.overlap_percent * patchx)
                          jumpy = int(self.overlap_percent * patchy)
                         
                          patchshape = (patchy, patchx)   
                          rowstart = 0; colstart = 0
                          pairs = []  
                          #row is y, col is x
                          
                          while rowstart < sliceregion.shape[1]:
                             colstart = 0
                             while colstart < sliceregion.shape[2]:
                                
                                 # Start iterating over the tile with jumps = stride of the fully convolutional network.
                                 pairs.append([rowstart, colstart])
                                 colstart+=jumpx
                             rowstart+=jumpy 
                            
                          #Include the last patch   
                          rowstart = sliceregion.shape[1] -patchy
                          colstart = 0
                          while colstart < sliceregion.shape[2] -patchx:
                                        pairs.append([rowstart, colstart])
                                        colstart+=jumpx
                          rowstart = 0
                          colstart = sliceregion.shape[2] -patchx
                          while rowstart < sliceregion.shape[1] -patchy:
                                        pairs.append([rowstart, colstart])
                                        rowstart+=jumpy              
                                        
                          if sliceregion.shape[1] >= self.imagey and sliceregion.shape[2]>= self.imagex :          
                              
                                patch = []
                                rowout = []
                                column = []
                                for pair in pairs: 
                                   smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, pair)
                                   if smallpatch.shape[1] >= self.imagey and smallpatch.shape[2] >= self.imagex:
                                           patch.append(smallpatch)
                                           rowout.append(smallrowout)
                                           column.append(smallcolumn) 
                        
                     else:
                         
                               patch = []
                               rowout = []
                               column = []
                               patchx = sliceregion.shape[2] // self.n_tiles[0]
                               patchy = sliceregion.shape[1] // self.n_tiles[1]
                               patchshape = (patchy, patchx)
                               smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, [0,0])
                               patch.append(smallpatch)
                               rowout.append(smallrowout)
                               column.append(smallcolumn)
             self.patch = patch          
             self.sy = rowout
             self.sx = column            
          
    

    
    def predict_main(self,sliceregion):
            try:
                self.overlaptiles(sliceregion)
                predictions = []
                allx = []
                ally = []
                if len(self.patch) > 0:
                    for i in range(0,len(self.patch)):   
                       
                               sum_time_prediction = self.make_patches(self.patch[i])
                               predictions.append(sum_time_prediction)
                               allx.append(self.sx[i])
                               ally.append(self.sy[i])
                      
                       
                else:
                    
                       sum_time_prediction = self.make_patches(self.patch)
                       predictions.append(sum_time_prediction)
                       allx.append(self.sx)
                       ally.append(self.sy)
           
            except tf.errors.ResourceExhaustedError:
                
                print('Out of memory, increasing overlapping tiles for prediction')
                self.list_n_tiles = list(self.n_tiles)
                self.list_n_tiles[0] = self.n_tiles[0]  + 1
                self.list_n_tiles[1] = self.n_tiles[1]  + 1
                self.n_tiles = tuple(self.list_n_tiles) 
                
                self.predict_main(sliceregion)
                
            return predictions, allx, ally
        
    def make_patches(self, sliceregion):
       
       
       predict_im = np.expand_dims(sliceregion,0)
       
       
       prediction_vector = self.model.predict(np.expand_dims(predict_im,-1), verbose = 0)
        
            
       return prediction_vector
   
    

        
def chunk_list(image, patchshape, stride, pair):
            rowstart = pair[0]
            colstart = pair[1]

            endrow = rowstart + patchshape[0]
            endcol = colstart + patchshape[1]

            if endrow >= image.shape[1]:
                endrow = image.shape[1]
            if endcol >= image.shape[2]:
                endcol = image.shape[2]


            region = (slice(0,image.shape[0]),slice(rowstart, endrow),
                      slice(colstart, endcol))

            # The actual pixels in that region.
            patch = image[region]

            # Always normalize patch that goes into the netowrk for getting a prediction score 
            


            return patch, rowstart, colstart
        
        
def CreateVolume(patch, imaget, timepoint, imagey, imagex):
    starttime = timepoint
    endtime = timepoint + imaget
    smallimg = patch[starttime:endtime, :]

    return smallimg
class EventViewer(object):
    
    def __init__(self, viewer, image, event_name, key_categories, imagename, savedir, canvas, ax, figure, yolo_v2):
        
        
           self.viewer = viewer
           self.image = image
           self.event_name = event_name
           self.imagename = imagename
           self.canvas = canvas
           self.key_categories = key_categories
           self.savedir = savedir
           self.ax = ax
           self.yolo_v2 = yolo_v2
           self.figure = figure
           self.plot()
    
    def plot(self):
        
        self.ax.cla()
        
        for (event_name,event_label) in self.key_categories.items():
                        if event_label > 0 and self.event_name == event_name:
                             csvname = self.savedir + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(self.imagename))[0] + '.csv')
                             event_locations, size_locations, angle_locations, line_locations, timelist, eventlist = self.event_counter(csvname)
                             
                             for layer in list(self.viewer.layers):
                                     if event_name in layer.name or layer.name in event_name or event_name + 'angle' in layer.name or layer.name in event_name + 'angle' :
                                            self.viewer.layers.remove(layer)
                                     if 'Image' in layer.name or layer.name in 'Image':
                                            self.viewer.layers.remove(layer)  
                             self.viewer.add_image(self.image, name='Image')               
                             self.viewer.add_points(np.asarray(event_locations), size = size_locations ,name = event_name, face_color = [0]*4, edge_color = "red", edge_width = 1)
                             if self.yolo_v2:
                                self.viewer.add_shapes(np.asarray(line_locations), name = event_name + 'angle',shape_type='line', face_color = [0]*4, edge_color = "red", edge_width = 1)
                             self.viewer.theme = 'light'
                             self.ax.plot(timelist, eventlist, '-r')
                             self.ax.set_title(event_name + "Events")
                             self.ax.set_xlabel("Time")
                             self.ax.set_ylabel("Counts")
                             self.figure.canvas.draw()
                             self.figure.canvas.flush_events()
                             plt.savefig(self.savedir  + event_name   + '.png') 
                             
    def event_counter(self, csv_file):
     
         time,y,x,score,size,confidence,angle  = np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)
         
         radius = 10
         eventcounter = 0
         eventlist = []
         timelist = []   
         listtime = time.tolist()
         listy = y.tolist()
         listx = x.tolist()
         listsize = size.tolist()
         listangle = angle.tolist()
         
         event_locations = []
         size_locations = []
         angle_locations = []
         line_locations = []
         for i in range(len(listtime)):
             tcenter = int(listtime[i])
             ycenter = listy[i]
             xcenter = listx[i]
             size = listsize[i]
             angle = listangle[i]
             eventcounter = listtime.count(tcenter)
             timelist.append(tcenter)
             eventlist.append(eventcounter)
             
             event_locations.append([tcenter, ycenter, xcenter])   
             size_locations.append(size)
             
             xstart = xcenter + radius * math.cos(angle )
             xend = xcenter - radius  * math.cos(angle)
             
             ystart = ycenter + radius * math.sin(angle)
             yend = ycenter - radius * math.sin(angle)
             line_locations.append([[tcenter, ystart, xstart], [tcenter, yend, xend]])
             angle_locations.append(angle)
             
            
         return event_locations, size_locations, angle_locations, line_locations, timelist, eventlist                         

