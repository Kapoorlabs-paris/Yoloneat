from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import get_nearest, save_json, load_json, yoloprediction, normalizeFloatZeroOne, GenerateMarkers, DensityCounter, MakeTrees, focyoloprediction, fastnms, simpleaveragenms
from keras import callbacks
import os
import math
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from NEATModels import nets
from NEATModels.nets import Concat
from NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv
import napari
import glob
from scipy import spatial
import itertools
from napari.qt.threading import thread_worker
import matplotlib.pyplot  as plt
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

class NEATFocus(object):
    

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
                self.stage_number = config.stage_number 
                self.last_conv_factor = config.last_conv_factor
                self.show = config.show
                self.key_cord = config.key_cord
                self.box_vector = len(config.key_cord)
                self.categories = len(config.key_categories)
                self.depth = config.depth
                self.start_kernel = config.start_kernel
                self.mid_kernel = config.mid_kernel
                self.learning_rate = config.learning_rate
                self.epochs = config.epochs
                self.residual = config.residual
                self.startfilter = config.startfilter
                self.batch_size = config.batch_size
                self.multievent = config.multievent
                self.imagex = config.imagex
                self.imagey = config.imagey
                self.imagez = config.size_tminus + config.size_tplus + 1
                self.size_zminus = config.size_tminus
                self.size_zplus = config.size_tplus
                self.nboxes = 1
                self.gridx = 1
                self.gridy = 1
                self.gridz = 1
                self.yolo_v0 = True
                self.yolo_v1 = False
                self.yolo_v2 = False
                self.stride = config.last_conv_factor
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
                self.learning_rate = self.config['learning_rate']
                self.epochs = self.config['epochs']
                self.residual = self.config['residual']
                self.startfilter = self.config['startfilter']
                self.batch_size = self.config['batch_size']
                self.multievent = self.config['multievent']
                self.imagex = self.config['imagex']
                self.imagey = self.config['imagey']
                self.imagez = self.config['size_tminus'] + self.config['size_tplus'] + 1
                self.size_zminus = self.config['size_tminus']
                self.size_zplus = self.config['size_tplus']
                self.nboxes = 1
                self.stage_number = self.config['stage_number'] 
                self.last_conv_factor = self.config['last_conv_factor']
                self.gridx = 1
                self.gridy = 1
                self.gridz = 1
                self.yolo_v0 = False
                self.yolo_v1 = True
                self.yolo_v2 = False
                self.stride = self.config['last_conv_factor']   
                
                
 
        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        
        if self.residual:
            self.model_keras = nets.resnet_3D_v2
        else:
            self.model_keras = nets.seqnet_3D_v2
            
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
        self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.gridz, 1, self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)
        
        
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
        self.Y = self.Y.reshape( (self.Y.shape[0],1,1,1,self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape( (self.Y_val.shape[0],1,1,1,self.Y_val.shape[1]))
          

              
    def TrainModel(self):
        
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])
        
        print(input_shape)
        Path(self.model_dir).mkdir(exist_ok=True)
       
        

        
        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):
        
            self.model_weights = model_weights
            print('loading weights')
        else:
           
            self.model_weights = None
        
        dummyY = np.zeros([self.Y.shape[0],1 ,self.Y.shape[1],self.Y.shape[2],self.categories + self.box_vector])
        dummyY[:,:,:,:,:self.Y.shape[4]] = self.Y
       
        dummyY_val = np.zeros([self.Y_val.shape[0],1,self.Y_val.shape[1],self.Y_val.shape[2],self.categories + self.box_vector])
        dummyY_val[:,:,:,:,:self.Y_val.shape[4]] = self.Y_val
        
        
        self.Y = dummyY
        self.Y_val = dummyY_val
        
        
        
        self.Trainingmodel = self.model_keras(input_shape, self.categories,  box_vector = self.box_vector, stage_number = self.stage_number, last_conv_factor = self.last_conv_factor, depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel,  startfilter = self.startfilter,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer=sgd, loss = self.yololoss, metrics=['accuracy'])
        
        self.Trainingmodel.summary()
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.key_categories, self.key_cord, self.gridx, self.gridy, plot = self.show, nboxes = 1)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
        
    
        
        
    def predict(self,imagename, savedir, interest_event, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0, iou_threshold = 0.01):
        
        self.imagename = imagename
        self.image = imread(imagename)
        self.Colorimage = np.zeros_like(self.image)
        self.savedir = savedir
        self.n_tiles = n_tiles
        self.interest_event = interest_event
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate","lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model =  load_model( self.model_dir + self.model_name + '.h5',  custom_objects={'loss':self.yololoss, 'Concat':Concat})
         
        self.first_pass_predict()
        
    
                    
            
    def first_pass_predict(self):

        eventboxes = []
        classedboxes = {}    
        print('Detecting focus planes in', os.path.basename(os.path.splitext(self.imagename)[0]))
        for inputz in tqdm(range(0, self.image.shape[0])):
                    if inputz <= self.image.shape[0] - self.imagez:
                                
                                eventboxes = []
                                
                                
                                smallimage = CreateVolume(self.image, self.imagez, inputz,self.imagex, self.imagey)
                                smallimage = normalizeFloatZeroOne(smallimage,1,99.8)
                                
                                # Cut off the region for training movie creation
                                #Break image into tiles if neccessary
                                predictions, allx, ally = self.predict_main(smallimage)
                                #Iterate over tiles
                                for p in range(0,len(predictions)):   
                        
                                  sum_z_prediction = predictions[p]
                                  
                                  if sum_z_prediction is not None:
                                     #For each tile the prediction vector has shape N H W Categories + Training Vector labels
                                     for i in range(0, sum_z_prediction.shape[0]):
                                          z_prediction =  sum_z_prediction[i]
                                          boxprediction = focyoloprediction(ally[p], allx[p], z_prediction, self.stride, inputz, self.config, self.key_categories, self.key_cord, 1, 'detection', 'dynamic' )
                                          
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
                                
                                self.nms()
                                self.to_csv()
                                eventboxes = []
                                classedboxes = {}    
                                                            
        self.print_planes()                                                    
                          
        
    def nms(self):
        
        
        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               #Get all events
               
               sorted_event_box = self.classedboxes[event_name][0]
              
               sorted_event_box = sorted(sorted_event_box, key = lambda x:x[event_name], reverse = True)
               
               scores = [ sorted_event_box[i][event_name]  for i in range(len(sorted_event_box))]
               best_sorted_event_box = simpleaveragenms(sorted_event_box, scores, self.iou_threshold, self.event_threshold, event_name)
               #nms_indices = fastnms(sorted_event_box, scores, self.iou_threshold, self.event_threshold, event_name)
               #best_sorted_event_box = [sorted_event_box[nms_indices[i]] for i in range(len(nms_indices))]
               
               
               best_iou_classedboxes[event_name] = [best_sorted_event_box]
               #print("nms",best_iou_classedboxes[event_name])
        self.iou_classedboxes = best_iou_classedboxes                
    

   

    
    def to_csv(self):
        
        
        
        if len(self.interest_event) > 1:
                zlocations = []
                scores = []
                event = self.interest_event[0]
                iou_current_event_box = self.iou_classedboxes[event][0]
                zcenter = iou_current_event_box['real_z_event']
                
                score = iou_current_event_box[event]
                for sec_event in self.interest_event:
                    
                     sec_iou_current_event_box = self.iou_classedboxes[sec_event][0]
                     sec_zcenter = sec_iou_current_event_box['real_z_event']
                     if event is not sec_event and zcenter == sec_zcenter:
                         
                              sec_score = sec_iou_current_event_box[sec_event]
                              score = score + sec_score
                              
                zlocations.append(zcenter)
                scores.append(score/len(self.interest_event))
                
                event_count = np.column_stack([zlocations,scores]) 
                event_count = sorted(event_count, key = lambda x:x[0], reverse = False)
                event_data = []
                csvname = self.savedir+ "/"  + "ComboFocusQuality" + (os.path.splitext(os.path.basename(self.imagename))[0])
                writer = csv.writer(open(csvname  +".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size
                if filesize < 1:
                   writer.writerow(['Z','Score'])
                for line in event_count:
                   if line not in event_data:  
                      event_data.append(line)
                   writer.writerows(event_data)
                   event_data = []
        
        for (event_name,event_label) in self.key_categories.items():
                   
            
                   
                   if event_label > 0:
                                            zlocations = []
                                            scores = []
                                            max_scores = []
                                            iou_current_event_box = self.iou_classedboxes[event_name][0]
                                            zcenter = iou_current_event_box['real_z_event']
                                            max_score = iou_current_event_box['max_score']
                                            score = iou_current_event_box[event_name]
                                            print(event_name, zcenter, score, max_score)           
                                            zlocations.append(zcenter)
                                            scores.append(score)
                                            max_scores.append(max_score)
                                            event_count = np.column_stack([zlocations,scores, max_scores]) 
                                            event_count = sorted(event_count, key = lambda x:x[0], reverse = False)
                                            event_data = []
                                            csvname = self.savedir+ "/" + event_name + "FocusQuality" + (os.path.splitext(os.path.basename(self.imagename))[0])
                                            writer = csv.writer(open(csvname  +".csv", "a"))
                                            filesize = os.stat(csvname + ".csv").st_size
                                            if filesize < 1:
                                               writer.writerow(['Z','Score','Max_score'])
                                            for line in event_count:
                                               if line not in event_data:  
                                                  event_data.append(line)
                                               writer.writerows(event_data)
                                               event_data = []           
                              
                                              
                                              
                                              
    
    def print_planes(self):
        
        Csv_path = os.path.join(self.savedir, '*csv')
        filesCsv = glob.glob(Csv_path)
        savename = self.savedir+ "/"  + "Stats" + (os.path.splitext(os.path.basename(self.imagename))[0])
        writer = csv.writer(open(savename  +".csv", "w"))
        filesize = os.stat(savename + ".csv").st_size
        if filesize < 1:
           writer.writerow(['FileName','Z found','Average Score'])
        filelist = []
        zlist = []
        scorelist = []
        for csvfname in filesCsv:
                                 Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                                 dataset = pd.read_csv(csvfname, skiprows = 1)
                                 z = dataset[dataset.keys()[0]][1:]
                                 score = dataset[dataset.keys()[1]][1:]
                                 try:
                                     maxscore = np.max(score)
                                     maxz = z[np.argmax(score)]
                                     filelist.append(Csvname)
                                     zlist.append(maxz)
                                     scorelist.append(maxscore)
                                 except:
                                    pass
        
        event_count = np.column_stack([filelist,zlist,scorelist]) 
        event_data = []
        for line in event_count:
              event_data.append(line)
              writer.writerows(event_data)
        
        
    def showNapari(self, imagedir, savedir, yolo_v2 = False):
         
         
         Raw_path = os.path.join(imagedir, '*tif')
         X = glob.glob(Raw_path)
         self.savedir = savedir
         Imageids = []
         self.viewer = napari.Viewer()
         napari.run()
         for imagename in X:
             Imageids.append(imagename)
         
         
         eventidbox = QComboBox()
         eventidbox.addItem(EventBoxname)
         for (event_name,event_label) in self.key_categories.items():
             
             eventidbox.addItem(event_name)
            
         imageidbox = QComboBox()   
         imageidbox.addItem(Boxname)   
         detectionsavebutton = QPushButton(' Save detection Movie')
         
         for i in range(0, len(Imageids)):
             
             
             imageidbox.addItem(str(Imageids[i]))
             
             
         figure = plt.figure(figsize=(4, 4))
         multiplot_widget = FigureCanvas(figure)
         ax = multiplot_widget.figure.subplots(1, 1)
         width = 400
         dock_widget = self.viewer.window.add_dock_widget(
         multiplot_widget, name="EventStats", area='right')
         multiplot_widget.figure.tight_layout()
         self.viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)    
         eventidbox.currentIndexChanged.connect(lambda eventid = eventidbox : EventViewer(
                 self.viewer,
                 imread(imageidbox.currentText()),
                 eventidbox.currentText(),
                 self.key_categories,
                 os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                 savedir,
                 multiplot_widget,
                 ax,
                 figure,
                 yolo_v2,
            
        )
    )    
         
         imageidbox.currentIndexChanged.connect(
         lambda trackid = imageidbox: EventViewer(
                 self.viewer,
                 imread(imageidbox.currentText()),
                 eventidbox.currentText(),
                 self.key_categories,
                 os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                 savedir,
                 multiplot_widget,
                 ax,
                 figure,
                 yolo_v2,
            
        )
    )            
         
         
         self.viewer.window.add_dock_widget(eventidbox, name="Event", area='left')  
         self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left')     
                                      
                                                             
                             
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
                          
                          while rowstart < sliceregion.shape[1] -patchy:
                             colstart = 0
                             while colstart < sliceregion.shape[2] -patchx:
                                
                                 # Start iterating over the tile with jumps = stride of the fully convolutional network.
                                 pairs.append([rowstart, colstart])
                                 colstart+=jumpx
                             rowstart+=jumpy 
                            
                          #Include the last patch   
                          rowstart = sliceregion.shape[1] -patchy
                          colstart = 0
                          while colstart < sliceregion.shape[2]:
                                        pairs.append([rowstart, colstart])
                                        colstart+=jumpx
                          rowstart = 0
                          colstart = sliceregion.shape[2] -patchx
                          while rowstart < sliceregion.shape[1]:
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
   
    def make_batch_patches(self, sliceregion): 
   
      
               prediction_vector = self.model.predict(np.expand_dims(sliceregion,-1), verbose = 0)
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
        
        
def CreateVolume(patch, imagez, timepoint, imagey, imagex):
    
               starttime = timepoint
               endtime = timepoint + imagez
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
             print(tcenter)
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
