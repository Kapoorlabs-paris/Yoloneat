#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:32:04 2021

@author: vkapoor
"""
from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import  load_json, yoloprediction, normalizeFloatZeroOne, fastnms, averagenms
from keras import callbacks
import os
import tensorflow as tf
import time
from NEATModels import nets
from NEATModels.nets import Concat
from NEATModels.loss import dynamic_yolo_loss
from tqdm import tqdm
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite, TiffFile, imsave
import csv
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import h5py
import cv2
import imageio
from PIL import Image
import matplotlib.pyplot as plt
class NEATPredict(object):
    

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
    
    lstm_hidden_units : Number of hidden units for LSTm layer, 64 by default
    
    epochs :  Number of training epochs, 55 by default
    
    batch_size : batch_size to be used for training, 20 by default
    
    
    
    """
    
    
    def __init__(self, config, model_dir, model_name, catconfig = None, cordconfig = None):

        
        self.config = config
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.model_name = model_name 
        if self.config !=None:
                self.npz_directory = config.npz_directory
                self.npz_name = config.npz_name
                self.npz_val_name = config.npz_val_name
                self.key_categories = config.key_categories
                self.box_vector = config.box_vector
                self.show = config.show
                self.key_cord = config.key_cord
                self.categories = len(config.key_categories)
                self.depth = config.depth
                self.start_kernel = config.start_kernel
                self.mid_kernel = config.mid_kernel
                self.lstm_kernel = config.lstm_kernel
                self.learning_rate = config.learning_rate
                self.epochs = config.epochs
                self.residual = config.residual
                self.startfilter = config.startfilter
                self.batch_size = config.batch_size
                self.multievent = config.multievent
                self.imagex = config.imagex
                self.imagey = config.imagey
                self.imaget = config.size_tminus
                self.size_tminus = config.size_tminus
                self.size_tplus = config.size_tplus
                self.nboxes = config.nboxes
                self.gridx = config.gridx
                self.gridy = config.gridy
                self.gridt = config.gridt
                self.yolo_v0 = config.yolo_v0
                self.yolo_v1 = config.yolo_v1
                self.yolo_v2 = config.yolo_v2
                self.stride = config.stride
                self.lstm_hidden_unit = config.lstm_hidden_unit
        if self.config == None:
                
            
                
                try:
                   self.config = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
                except:
                   self.config = load_json(self.model_dir + self.model_name + '_Parameter.json')  
                   
                self.npz_directory = self.config['npz_directory']
                self.npz_name = self.config['npz_name']
                self.npz_val_name = self.config['npz_val_name']
                self.key_categories = self.catconfig
                self.box_vector = self.config['box_vector']
                self.show = self.config['show']
                self.key_cord = self.cordconfig
                self.categories = len(self.catconfig)
                self.depth = self.config['depth']
                self.start_kernel = self.config['start_kernel']
                self.mid_kernel = self.config['mid_kernel']
                self.lstm_kernel = self.config['lstm_kernel']
                self.lstm_hidden_unit = self.config['lstm_hidden_unit']
                self.learning_rate = self.config['learning_rate']
                self.epochs = self.config['epochs']
                self.residual = self.config['residual']
                self.startfilter = self.config['startfilter']
                self.batch_size = self.config['batch_size']
                self.multievent = self.config['multievent']
                self.imagex = self.config['imagex']
                self.imagey = self.config['imagey']
                self.imaget = self.config['size_tminus']
                self.size_tminus = self.config['size_tminus']
                self.size_tplus = self.config['size_tplus']
                self.nboxes = self.config['nboxes']
                self.gridx = 1
                self.gridy = 1
                self.gridt = 1
                self.yolo_v0 = self.config['yolo_v0']
                self.yolo_v1 = self.config['yolo_v1']
                self.yolo_v2 = self.config['yolo_v2']
                self.stride = self.config['stride']   
                self.lstm_hidden_unit = self.config['lstm_hidden_unit']
                
                

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        
        if self.residual:
            self.model_keras = nets.ORNET
        else:
            self.model_keras = nets.OSNET
            
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
        
        self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.gridt, self.nboxes, self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)
        
        
    def loadData(self):
        
        (X,Y),  axes = helpers.load_full_training_data(self.npz_directory, self.npz_name, verbose= True)

        (X_val,Y_val), axes = helpers.load_full_training_data(self.npz_directory, self.npz_val_name,  verbose= True)
        
        
        self.Xoriginal = X
        self.Xoriginal_val = X_val
        

                     

        self.X = X
        self.Y = Y[:,:,0]
        self.X_val = X_val
        self.Y_val = Y_val[:,:,0]
        self.axes = axes
        self.Y = self.Y.reshape( (self.Y.shape[0],1,1,self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape( (self.Y_val.shape[0],1,1,self.Y_val.shape[1]))
          

              
    def TrainModel(self):
        
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])
        
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
       
        Y_rest = self.Y[:,:,:,self.categories:]
        Y_main = self.Y[:,:,:,0:self.categories-1]
 
        y_integers = np.argmax(Y_main, axis = -1)
        y_integers = y_integers[:,0,0]

        
        
        d_class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = d_class_weights.reshape(1,d_class_weights.shape[0])
        dummyY = np.zeros([self.Y.shape[0],self.Y.shape[1],self.Y.shape[2],self.categories + self.nboxes* self.box_vector])
        dummyY[:,:,:,:self.Y.shape[3]] = self.Y
        
        dummyY_val = np.zeros([self.Y_val.shape[0],self.Y_val.shape[1],self.Y_val.shape[2],self.categories + self.nboxes* self.box_vector])
        dummyY_val[:,:,:,:self.Y_val.shape[3]] = self.Y_val
        for b in range(1, self.nboxes):
            
            dummyY[:,:,:,self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y[:,:,:, self.categories: self.categories + self.box_vector]
            dummyY_val[:,:,:,self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y_val[:,:,:, self.categories: self.categories + self.box_vector]
            
        self.Y = dummyY
        self.Y_val = dummyY_val
        
        self.Trainingmodel = self.model_keras(input_shape, self.categories,  unit = self.lstm_hidden_unit ,nboxes = self.nboxes, box_vector = Y_rest.shape[-1] , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, lstm_kernel = self.lstm_kernel, startfilter = self.startfilter,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer=sgd, loss = self.yololoss, metrics=['accuracy'])
        
        self.Trainingmodel.summary()
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name, plot = self.show, simple = self.simple, catsimple = self.catsimple)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, class_weight = d_class_weights , batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
    def predict(self, imagedir,  movie_name_list, movie_input, Z_imagedir, Z_movie_name_list, Z_movie_input, start, Z_start, downsample = False, fileextension = '*TIF', nb_prediction = 3, n_tiles = (1,1), Z_n_tiles = (1,2,2), overlap_percent = 0.6, event_threshold = 0.5, iou_threshold = 0.01, projection_model = None):
        
        self.imagedir = imagedir
        self.basedirResults = self.imagedir + '/' + "live_results"
        Path(self.basedirResults).mkdir(exist_ok = True)
        #Recurrsion variables
        self.movie_name_list = movie_name_list
        self.movie_input = movie_input
        self.Z_movie_name_list = Z_movie_name_list
        self.Z_movie_input = Z_movie_input
        self.Z_imagedir = Z_imagedir
        self.start = start
        self.Z_start = Z_start
        self.projection_model = projection_model
        self.nb_prediction = nb_prediction
        self.fileextension = fileextension
        self.n_tiles = n_tiles
        self.Z_n_tiles = Z_n_tiles
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.downsample = downsample
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate","lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model =  load_model( self.model_dir + self.model_name + '.h5',  custom_objects={'loss':self.yololoss, 'Concat':Concat})

        #Z slice folder listener   
        while 1:
            
              Z_Raw_path = os.path.join(self.Z_imagedir, self.fileextension)
              Z_filesRaw = glob.glob(Z_Raw_path)
              Z_filesRaw = natsorted(Z_filesRaw)
              
              Raw_path = os.path.join(self.imagedir, '*tif')
              filesRaw = glob.glob(Raw_path)
              filesRaw = natsorted(filesRaw)
              for Z_movie_name in Z_filesRaw:  
                          Z_Name = os.path.basename(os.path.splitext(Z_movie_name)[0])
                          #Check for unique filename
                          if Z_Name not in self.Z_movie_name_list and "w2" in Z_Name:
                              
                          
                                               
                                     self.Z_movie_name_list.append(Z_Name)
                                     self.Z_movie_input.append(Z_movie_name)
                                     total_Z_movies = len(self.Z_movie_input)
                                  
              
              for movie_name in filesRaw:  
                          Name = os.path.basename(os.path.splitext(movie_name)[0])
                          #Check for unique filename
                          if Name not in self.movie_name_list:
                              
                          
                                   
                                     self.movie_name_list[Name] = Name
                                     self.movie_input[Name] = movie_name 
                                     
                                     total_movies = len(self.movie_input)                       

              doproject = True
             
              for  i in range(len(self.Z_movie_name_list)):
                   
                    Z_Name = self.Z_movie_name_list[i]
                    Z_path = self.Z_movie_input[i]
                    
                    if Z_Name in  self.movie_name_list:
                       
                        Name = self.movie_name_list[Z_Name]
                        path = self.movie_input[Z_Name]
                       
                        
                            
                        doproject = False
                    else:
                         doproject = True    
                   
                    if doproject:
                                    
                                     try:    
                                             start_time = time.time()
                                             print('Reading Z stack for projection')
                                             Z_image = imread(Z_path)
                                             print('Read properly')
                                     except:
                                         
                                           Z_image = None
                                     if Z_image is not None:      
                                             if self.projection_model is not None:
                                                 print('Projecting using the projection model')
                                                 projection = self.projection_model.predict(Z_image, 'ZYX', n_tiles = Z_n_tiles)
                                             else:
                                                 print('Doing max projection')
                                                 projection = np.amax(Z_image, axis = 0)
                                             imwrite(self.imagedir + '/' + Z_Name + '.tif' , projection.astype('float32'))
                                             print("____ Projection took %s seconds ____ ", (time.time() - start_time  ) )
                                 
                                     else:
                                           if Z_Name in self.Z_movie_name_list:
                                              self.Z_movie_name_list.remove(Z_Name)
                                           if Z_movie_name in self.Z_movie_input:      
                                              self.Z_movie_input.remove(Z_movie_name)
                                                      
              self.movie_input_list = []
              for (k,v) in self.movie_input.items():
                                                       
                              self.movie_input_list.append(v)
              total_movies = len(self.movie_input_list)
              if total_movies > self.size_tminus + self.start:
                                                                  current_movies = imread(self.movie_input_list[self.start:self.start + self.size_tminus + 1])
                                                                  
                                                                  sizey = current_movies.shape[1]
                                                                  sizex = current_movies.shape[2]
                                                                  if self.downsample:
                                                                                scale_percent = 50
                                                                                width=int(sizey * scale_percent / 100)
                                                                                height=int(sizex * scale_percent / 100)
                                                                                dim = (width, height)
                                                                                sizex = height
                                                                                sizey = width
                                                                                
                                                                                current_movies_down = np.zeros([current_movies.shape[0], sizey, sizex])
                                                                                # resize image
                                                                                for j in range(current_movies.shape[0]):
                                                                                        current_movies_down[j,:] = cv2.resize(current_movies[j,:], dim, interpolation = cv2.INTER_AREA)
                                                                  else:                             
                                                                        current_movies_down = current_movies
                                                                  #print(current_movies_down.shape) 
                                                                  print('Predicting on Movies:',self.movie_input_list[self.start:self.start + self.size_tminus + 1]) 
                                                                  inputtime = self.start + self.size_tminus
                                                                  
                                                                      
                                                                  eventboxes = []
                                                                  classedboxes = {}
                                                                  smallimage = CreateVolume(current_movies_down, self.size_tminus + 1, 0,sizex, sizey)
                                                                  
                                                                  smallimage = normalizeFloatZeroOne(smallimage,1,99.8)          
                                                                  #Break image into tiles if neccessary
                                                                  self.image = smallimage
                                                                  print('Doing ONEAT prediction')
                                                                  start_time = time.time()
                                                                  predictions, allx, ally = self.predict_main(smallimage)
                                                                  print("____ Prediction took %s seconds ____ ", (time.time() - start_time  ) )
                                                                 
                                                                  #Iterate over tiles
                                                                  for p in tqdm(range(0,len(predictions))):   
                                                        
                                                                          sum_time_prediction = predictions[p]
                                                                          
                                                                          if sum_time_prediction is not None:
                                                                             for i in range(0, sum_time_prediction.shape[0]):
                                                                                  time_prediction =  sum_time_prediction[i]
                                                                                  
                                                                                  boxprediction = yoloprediction(smallimage, ally[p], allx[p], time_prediction, self.stride, inputtime, self.config, self.key_categories, self.key_cord, self.nboxes, 'prediction', 'dynamic')
                                                                                  
                                                                                  if boxprediction is not None:
                                                                                          eventboxes = eventboxes + boxprediction
                                                                             
                                                                  for (event_name,event_label) in self.key_categories.items(): 
                                                                         
                                                                      if event_label > 0:
                                                                           current_event_box = []
                                                                           for box in eventboxes:
                                                                    
                                                                              event_prob = box[event_name]
                                                                              if event_prob >= self.event_threshold:
                                                                               
                                                                                  current_event_box.append(box)
                                                                           classedboxes[event_name] = [current_event_box]
                                                                     
                                                                  self.classedboxes = classedboxes    
                                                                  self.eventboxes =  eventboxes  
                                                                  print('Performining non maximal supression')
                                                                  start_time = time.time()
                                                                  self.iou_classedboxes = classedboxes
                                                                  self.nms()
                                                                  print("____ NMS took %s seconds ____ ", (time.time() - start_time  ) )
                                                                  print('Generating ini file')
                                                                  self.to_csv()
                                                                  self.predict(self.imagedir,  self.movie_name_list, self.movie_input, self.Z_imagedir, self.Z_movie_name_list, self.Z_movie_input, self.start + 1, Z_start, fileextension = self.fileextension, downsample = self.downsample, nb_prediction = self.nb_prediction, n_tiles = self.n_tiles, Z_n_tiles = self.Z_n_tiles, overlap_percent =self.overlap_percent, event_threshold = self.event_threshold, iou_threshold = self.iou_threshold, projection_model = self.projection_model)
                               
                                 
                                                
        
        

    def nms(self):
        
        
        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               #Get all events
               
               sorted_event_box = self.classedboxes[event_name][0]
               scores = [ sorted_event_box[i][event_name]  for i in range(len(sorted_event_box))]
               #best_sorted_event_box = averagenms(sorted_event_box, scores, self.iou_threshold, self.event_threshold, event_name, 'dynamic')
               nms_indices = fastnms(sorted_event_box, scores, self.iou_threshold, self.event_threshold, event_name)
               best_sorted_event_box = [sorted_event_box[nms_indices[i]] for i in range(len(nms_indices))]
               
               best_iou_classedboxes[event_name] = [best_sorted_event_box]
               
        self.iou_classedboxes = best_iou_classedboxes                  
        
    def to_csv(self):
        
        for (event_name,event_label) in self.key_categories.items():
                   
                   if event_label > 0:
                                  
                                      xlocations = []
                                      ylocations = []
                                      scores = []
                                      tlocations = []   
                                      radiuses = []
                                      predcount = 0
                                      iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                                      #iou_current_event_boxes = sorted(iou_current_event_boxes, key = lambda x:x[event_name], reverse = True)
                                      iou_current_event_boxes = sorted(iou_current_event_boxes, key = lambda x:abs(x['xcenter'] - self.image.shape[2]//2) + abs(x['ycenter'] - self.image.shape[1]//2), reverse = False) 
                                      
                                      for iou_current_event_box in iou_current_event_boxes:
                                              if predcount > self.nb_prediction:
                                                   break
                                             
                                              xcenter = iou_current_event_box['xcenter']
                                              ycenter = iou_current_event_box['ycenter']
                                              tcenter = iou_current_event_box['real_time_event']
                                              score = iou_current_event_box[event_name]
                                              radius = np.sqrt( iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box['width'] * iou_current_event_box['width']  )// 2
                                              print(round(xcenter), round(ycenter), score)
                                              xlocations.append(round(xcenter))
                                              ylocations.append(round(ycenter))
                                              scores.append(score)
                                              tlocations.append(tcenter)
                                              radiuses.append(radius)
                                              predcount = predcount + 1
                                      event_count = np.column_stack([xlocations,ylocations]) 
                                      csvname = self.basedirResults + "/" + event_name
                                      

                                      writer = csv.writer(open(csvname + ".ini", 'w'))
                                      writer.writerow(["[main]"])  
                                      writer.writerow(["nbPredictions="+str(self.nb_prediction)])
                                      live_event_data = []
                                      count = 1
                                      
                                      for line in event_count:
                                                                                            
                                              live_event_data.append(line)
                                              writer.writerow(["["+str(count - 1)+"]"])
                                              writer.writerow(["x="+str(live_event_data[0][0])])
                                              writer.writerow(["y="+str(live_event_data[0][1])])
                                              live_event_data = []
                                                  
                                              count = count + 1
                                           
                                      ImageResults = self.basedirResults + '/' + 'ImageLocations'
                                      Path(ImageResults).mkdir(exist_ok=True)

                                      csvimagename = ImageResults + "/" + event_name + 'LocationData'
                                      name = str(self.start)
                                      self.saveimage(xlocations, ylocations, radiuses, scores, csvimagename, name)
  
                                      
                 
                                   
    def saveimage(self, xlocations, ylocations, radius, scores, csvimagename, name):

                        

                                      StaticImage = self.image[self.image.shape[0] - 1,:]
                                      StaticImage = normalizeFloatZeroOne(StaticImage,1,99.8)
                                      Colorimage = np.zeros_like(StaticImage)

                                      copyxlocations = xlocations.copy()
                                      copyylocations = ylocations.copy()
                                      for j in range(len(copyxlocations)):
                                         startlocation = (int(copyxlocations[j] - radius[j]), int(copyylocations[j]-radius[j]))
                                         endlocation =  (int(copyxlocations[j] + radius[j]), int(copyylocations[j]+radius[j]))
                                         score = scores[j]
                                         if score == 1:
                                             color = (0,0,255)
                                         if score > 0.95 and score <=1:
                                             color = (0, 255,0)
                                         if score > 0.9 and score <= 0.95:
                                             color = (255,0,0)
                                         else:
                                             color = (0,0,0)
                                                 
                                         cv2.rectangle(Colorimage, startlocation, endlocation, color, 1 )
                                      RGBImage = [StaticImage, Colorimage, Colorimage]
                                      RGBImage = np.swapaxes(np.asarray(RGBImage),0, 2)
                                      RGBImage = np.swapaxes(RGBImage, 0,1) 
                                      imageio.imwrite((csvimagename  + name + '.tif' ), RGBImage)

       
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
                          
                          while rowstart < sliceregion.shape[1] :
                             colstart = 0
                             while colstart < sliceregion.shape[2]:
                                
                                 # Start iterating over the tile with jumps = stride of the fully convolutional network.
                                 pairs.append([rowstart, colstart])
                                 colstart+=jumpx
                             rowstart+=jumpy 
                            
                          #Include the last patch   
                          rowstart = sliceregion.shape[1]
                          colstart = 0
                          while colstart < sliceregion.shape[2]:
                                        pairs.append([rowstart, colstart])
                                        colstart+=jumpx
                          rowstart = 0
                          colstart = sliceregion.shape[2] - patchx
                          while rowstart < sliceregion.shape[1] - patchy:
                                        pairs.append([rowstart, colstart])
                                        rowstart+=jumpy              
                                        
                          if sliceregion.shape[1] >= self.imagey and sliceregion.shape[2]>= self.imagex :          
                              
                                patch = []
                                rowout = []
                                column = []
                                for pair in pairs: 
                                   smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, pair)
                                   patch.append(smallpatch)
                                   rowout.append(smallrowout)
                                   column.append(smallcolumn) 
                        
                     else:
                         
                               patch = []
                               rowout = []
                               column = []
                               
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
                       try:
                               sum_time_prediction = self.make_patches(self.patch[i])
                               predictions.append(sum_time_prediction)
                               allx.append(self.sx[i])
                               ally.append(self.sy[i])
                       except:
                           
                           pass
                       
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

            if endrow > image.shape[1]:
                endrow = image.shape[1]
            if endcol > image.shape[2]:
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
       
