#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:13:01 2020

@author: aimachine
"""

from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import save_json, load_json, yoloprediction, normalizeFloatZeroOne
from keras import callbacks
import os
from tqdm import tqdm
from NEATModels import nets
from NEATModels.nets import Concat
from NEATModels.loss import static_yolo_loss
from keras import backend as K
import tensorflow as tf
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv
import napari
from napari.qt.threading import thread_worker
import matplotlib.pyplot  as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import glob



Boxname = 'ImageIDBox'
CellTypeBoxname = 'CellIDBox'

class NEATStatic(object):
    

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
    
    
    def __init__(self, staticconfig, model_dir, model_name):

        

        
        self.staticconfig = staticconfig 
        
        if self.staticconfig !=None:
                self.npz_directory = staticconfig.npz_directory
                self.npz_name = staticconfig.npz_name
                self.npz_val_name = staticconfig.npz_val_name
                self.key_categories = staticconfig.key_categories
                self.box_vector = staticconfig.box_vector
                self.show = staticconfig.show
                self.key_cord = staticconfig.key_cord
                self.categories = len(staticconfig.key_categories)
                self.depth = staticconfig.depth
                self.start_kernel = staticconfig.start_kernel
                self.mid_kernel = staticconfig.mid_kernel
                self.learning_rate = staticconfig.learning_rate
                self.epochs = staticconfig.epochs
                self.residual = staticconfig.residual
                self.startfilter = staticconfig.startfilter
                self.batch_size = staticconfig.batch_size
                self.multievent = staticconfig.multievent
                self.imagex = staticconfig.imagex
                self.imagey = staticconfig.imagey
                self.nboxes = staticconfig.nboxes
                self.gridx = staticconfig.gridx
                self.gridy = staticconfig.gridy
                self.yolo_v0 = staticconfig.yolo_v0
                self.stride = staticconfig.stride
                
        if self.staticconfig == None:
               
                try:
                   self.staticconfig = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
                except:
                   self.staticconfig = load_json(self.model_dir + self.model_name + '_Parameter.json')  
                   
                self.npz_directory = staticconfig['npz_directory']
                self.npz_name = staticconfig['npz_name']
                self.npz_val_name = staticconfig['npz_val_name']
                self.key_categories = staticconfig['key_categories']
                self.box_vector = staticconfig['box_vector']
                self.show = staticconfig['show']
                self.key_cord = staticconfig['key_cord']
                self.categories = len(staticconfig['key_categories'])
                self.depth = staticconfig['depth']
                self.start_kernel = staticconfig['start_kernel']
                self.mid_kernel = staticconfig['mid_kernel']
                self.learning_rate = staticconfig['learning_rate']
                self.epochs = staticconfig['epochs']
                self.residual = staticconfig['residual']
                self.startfilter = staticconfig['startfilter']
                self.batch_size = staticconfig['batch_size']
                self.multievent = staticconfig['multievent']
                self.imagex = staticconfig['imagex']
                self.imagey = staticconfig['imagey']
                self.nboxes = staticconfig['nboxes']
                self.gridx = staticconfig['gridx']
                self.gridy = staticconfig['gridy']
                self.yolo_v0 = staticconfig['yolo_v0']
                self.stride = staticconfig['stride']    
        
        self.model_dir = model_dir
        self.model_name = model_name        
        self.model_weights = None        
        self.last_activation = None
        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        
        if self.residual:
            self.model_keras = nets.resnet_v2
        else:
            self.model_keras = nets.seqnet_v2
            
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
         
        self.yololoss = static_yolo_loss(self.categories, self.gridx, self.gridy, self.nboxes, self.box_vector, self.entropy, self.yolo_v0)
       
            
   
        

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
        
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3])
        
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
       
        Y_main = self.Y[:,:,:,0:self.categories-1]
 
        y_integers = np.argmax(Y_main, axis = -1)
        y_integers = y_integers[:,0,0]

      
        
        d_class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = d_class_weights.reshape(1,d_class_weights.shape[0])
        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):
        
            self.model_weights = model_weights
            print('loading weights')
        else:
           
            self.model_weights = None
        
        dummyY = np.zeros([self.Y.shape[0],self.Y.shape[1],self.Y.shape[2],self.categories + self.nboxes* self.box_vector])
        dummyY[:,:,:,:self.Y.shape[3]] = self.Y
        
        dummyY_val = np.zeros([self.Y_val.shape[0],self.Y_val.shape[1],self.Y_val.shape[2],self.categories + self.nboxes* self.box_vector])
        dummyY_val[:,:,:,:self.Y_val.shape[3]] = self.Y_val
        for b in range(1, self.nboxes):
            
            dummyY[:,:,:,self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y[:,:,:, self.categories: self.categories + self.box_vector]
            dummyY_val[:,:,:,self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y_val[:,:,:, self.categories: self.categories + self.box_vector]
            
        self.Y = dummyY
        self.Y_val = dummyY_val
        
        print(self.Y.shape, self.nboxes)
        
        self.Trainingmodel = self.model_keras(input_shape, self.categories, box_vector = self.box_vector ,nboxes = self.nboxes, depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, startfilter = self.startfilter,last_activation = self.last_activation,  input_weights  =  self.model_weights)
        
        
        
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer=sgd, loss = self.yololoss, metrics=['accuracy'])
        self.Trainingmodel.summary()
        
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotStaticHistory(self.Trainingmodel, self.X_val, self.Y_val, self.key_categories, self.key_cord, self.gridx, self.gridy, plot = self.show, nboxes = self.nboxes)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, class_weight = d_class_weights,  batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])
        #clear_output(wait=True) 

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
        

    def predict(self, imagename, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, iou_threshold = 0.01):
        
        self.imagename = imagename
        self.image = imread(imagename)
        self.n_tiles = n_tiles
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        try:
            self.model =  load_model( self.model_dir + self.model_name + '.h5',  custom_objects={'loss':self.yolo_loss, 'Concat':Concat})
        except:
            self.model =  load_model( self.model_dir + self.model_name,  custom_objects={'loss':self.yolo_loss, 'Concat':Concat})
            
        for inputtime in tqdm(0, self.image.shape[0]):
            
            smallimage = self.image[inputtime,:]
            eventboxes = []
            classedboxes = {}
            smallimage = normalizeFloatZeroOne(smallimage,1,99.8)          
            #Break image into tiles if neccessary
            predictions, allx, ally = self.predict_main(smallimage)
            #Iterate over tiles
            for p in range(0,len(predictions)):   
    
              sum_time_prediction = predictions[p]
              
              if sum_time_prediction is not None:
                 #For each tile the prediction vector has shape N H W Categories + Trainng Vector labels
                 for i in range(0, sum_time_prediction.shape[0]):
                      time_prediction =  sum_time_prediction[i]
                      boxprediction = yoloprediction(smallimage, ally[p], allx[p], time_prediction, self.stride, inputtime, self.staticconfig, self.key_categories, self.key_cord, self.nboxes, 'detection', 'static')
                      
                      if boxprediction is not None:
                              eventboxes = eventboxes + boxprediction
                         
            for (event_name,event_label) in self.key_categories.items(): 
                     
                if event_label > 0:
                     current_event_box = []
                     for box in eventboxes:
                
                        event_prob = box[event_name]
                        if event_prob > self.event_threshold:
                           
                            current_event_box.append(box)
                     classedboxes[event_name] = [current_event_box]
                 
            self.classedboxes = classedboxes    
            self.eventboxes =  eventboxes  
            
            self.nms()
            self.to_csv()
        
        
    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1['xstart'], box1['xstart'] + box1['xcenter']], [box2['xstart'], box2['xstart'] + box2['xcenter']])
        intersect_h = self._interval_overlap([box1['ystart'], box1['ystart'] + box1['ycenter']], [box2['ystart'], box2['ystart'] + box2['ycenter']])

        intersect = intersect_w * intersect_h

        w1, h1 = box1['width'], box1['height']
        w2, h2 = box2['width'], box2['height']

        union = w1*h1 + w2*h2 - intersect

        return float(np.true_divide(intersect, union))
    
    
    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3
        
    def nms(self):
        
        
        iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               current_event_box =  self.classedboxes.event_name
               iou_current_event_box = []
               if current_event_box is not None:
                    
                    if len(current_event_box) == 0 :
                        
                        return []
                    else:
                        
                        index_boxes = []  
                        current_event_box = np.array(current_event_box, dtype = float)
                        assert current_event_box.shape[0] > 0
                        if current_event_box.dtype.kind!="f":
                            current_event_box = current_event_box.astype(np.float16)
        
                        idxs = current_event_box[:,event_name].argsort()[::-1]
                        
                        for i in range(len(idxs)):
                            
                            index_i = idxs[i]
                            index_boxes.append(index_i)
                            for j in range(i + 1, len(idxs)):
                                
                                index_j = idxs[j]
                                bbox_iou = self.bbox_iou(current_event_box[index_i], current_event_box[index_j])
                                if bbox_iou >= self.iou_threshold:
                                    
                                    iou_current_event_box.append(current_event_box[index_j])
                                    
               iou_classedboxes[event_name] = [iou_current_event_box]
                                
        self.iou_classedboxes = iou_classedboxes                
        
    def to_csv(self):
        
        for (event_name,event_label) in self.key_categories.items():
                   
                   if event_label > 0:
                              xlocations = []
                              ylocations = []
                              scores = []
                              confidences = []
                              tlocations = []   
                              radiuses = []
                              
                              iou_current_event_boxes = self.iou_classedboxes.event_name
                              
                              for iou_current_event_box in iou_current_event_boxes:
                                      xcenter = iou_current_event_box['xcenter']
                                      ycenter = iou_current_event_box['ycenter']
                                      tcenter = iou_current_event_box['real_time_event']
                                      confidence = iou_current_event_box['confidence']
                                      score = iou_current_event_box['event_name']
                                      radius = np.sqrt( iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box['width'] * iou_current_event_box['width']  )// 2
                                      xlocations.append(xcenter)
                                      ylocations.append(ycenter)
                                      scores.append(score)
                                      confidences.append(confidence)
                                      tlocations.append(tcenter)
                                      radiuses.append(radius)
                              
                              
                              event_count = np.column_stack([tlocations,ylocations,xlocations,scores,radiuses,confidences]) 
                              event_data = []
                              writer = csv.writer(open(os.path.dirname(self.imagename) + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(self.imagename))[0])  +".csv", "a"))
                              writer.writerow(['T','Y','X','Score','Size','Confidence'])
                              for line in event_count:
                                 event_data.append(line)
                                 writer.writerows(event_data)
                                 event_data = []           
                              
         
    def showNapari(self, imagedir, savedir):
         
         
         Raw_path = os.path.join(imagedir, '*tif')
         X = glob.glob(Raw_path)
         self.savedir = savedir
         Imageids = []
         self.viewer = napari.Viewer()
         napari.run()
         for imagename in X:
             Imageids.append(imagename)
         
         
         celltypeidbox = QComboBox()
         celltypeidbox.addItem(CellTypeBoxname)
         for (event_name,event_label) in self.key_categories.items():
             
             celltypeidbox.addItem(event_name)
            
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
         multiplot_widget, name="CellTypeStats", area='right')
         multiplot_widget.figure.tight_layout()
         self.viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)    
         celltypeidbox.currentIndexChanged.connect(lambda eventid = celltypeidbox : CellTypeViewer(
                 self.viewer,
                 imread(imageidbox.currentText()),
                 celltypeidbox.currentText(),
                 self.key_categories,
                 os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                 savedir,
                 multiplot_widget,
                 ax,
                 figure,
            
        )
    )    
         
         imageidbox.currentIndexChanged.connect(
         lambda trackid = imageidbox: CellTypeViewer(
                 self.viewer,
                 imread(imageidbox.currentText()),
                 celltypeidbox.currentText(),
                 self.key_categories,
                 os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                 savedir,
                 multiplot_widget,
                 ax,
                 figure,
            
        )
    )            
         
         
         self.viewer.window.add_dock_widget(celltypeidbox, name="CellType", area='left')  
         self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left')           
            
            
          
    def overlaptiles(self):
        
            if self.n_tiles == 1:
                
                       patchshape = (self.image.shape[0], self.image.shape[1])  
                      
                       image = zero_pad(self.image, self.stride,self.stride)
        
                       patch = []
                       rowout = []
                       column = []
                       
                       patch.append(image)
                       rowout.append(0)
                       column.append(0)
                     
            else:
                  
             patchx = self.image.shape[1] // self.n_tiles
             patchy = self.image.shape[0] // self.n_tiles
        
             if patchx > self.imagex and patchy > self.imagey:
              if self.overlap_percent > 1 or self.overlap_percent < 0:
                 self.overlap_percent = 0.8
             
              jumpx = int(self.overlap_percent * patchx)
              jumpy = int(self.overlap_percent * patchy)
             
              patchshape = (patchy, patchx)   
              rowstart = 0; colstart = 0
              pairs = []  
              #row is y, col is x
              
              while rowstart < self.image.shape[0] - patchy:
                 colstart = 0
                 while colstart < self.image.shape[1] - patchx:
                    
                     # Start iterating over the tile with jumps = stride of the fully convolutional network.
                     pairs.append([rowstart, colstart])
                     colstart+=jumpx
                 rowstart+=jumpy 
                
              #Include the last patch   
              rowstart = self.image.shape[0] - patchy
              colstart = 0
              while colstart < self.image.shape[1] - patchx:
                            pairs.append([rowstart, colstart])
                            colstart+=jumpx
              rowstart = 0
              colstart = self.image.shape[1] - patchx
              while rowstart < self.image.shape[0] - patchy:
                            pairs.append([rowstart, colstart])
                            rowstart+=jumpy              
                            
              if self.image.shape[0] >= self.imagey and self.image.shape[1]>= self.imagex :          
                  
                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs: 
                       smallpatch, smallrowout, smallcolumn =  chunk_list(self.image, patchshape, self.stride, pair)
                       patch.append(smallpatch)
                       rowout.append(smallrowout)
                       column.append(smallcolumn) 
                
             else:
                 
                       patch = []
                       rowout = []
                       column = []
                       image = zero_pad(self.image, self.stride,self.stride)
                       
                       patch.append(image)
                       rowout.append(0)
                       column.append(0)
                       
            self.patch = patch          
            self.sy = rowout
            self.sx = column            
          
        
    def predict_main(self,sliceregion):
            try:
                self.overlaptiles()
                predictions = []
                allx = []
                ally = []
                for i in range(0,len(self.patch)):   
                   
                   sum_time_prediction = self.make_patches(self.patch[i])

                   predictions.append(sum_time_prediction)
                   allx.append(self.sx[i])
                   ally.append(self.sy[i])
           
            except tf.errors.ResourceExhaustedError:
                
                print('Out of memory, increasing overlapping tiles for prediction')
                
                self.n_tiles = self.n_tiles  + 1
                
                print('Tiles: ', self.n_tiles)
                
                self.predict_main(sliceregion)
                
            return predictions, allx, ally
        
    def make_patches(self, sliceregion):
       
       
       predict_im = np.expand_dims(sliceregion,0)
       
       
       prediction_vector = self.model.predict(np.expand_dims(predict_im,-1), verbose = 0)
         
            
       return prediction_vector
   
    
def zero_pad(patch, jumpx, jumpy):

        
          sizey = patch.shape[0]
          sizex = patch.shape[1]

          sizexextend = sizex
          sizeyextend = sizey


          while sizexextend%jumpx!=0:
              sizexextend = sizexextend + 1

          while sizeyextend%jumpy!=0:
              sizeyextend = sizeyextend + 1

          extendimage = np.zeros([sizeyextend, sizexextend])

          extendimage[0:sizey, 0:sizex] = patch


          return extendimage  
      
        
def chunk_list(image, patchshape, stride, pair):
            rowstart = pair[0]
            colstart = pair[1]

            endrow = rowstart + patchshape[0]
            endcol = colstart + patchshape[1]

            if endrow > image.shape[0]:
                endrow = image.shape[0]
            if endcol > image.shape[1]:
                endcol = image.shape[1]


            region = (slice(rowstart, endrow),
                      slice(colstart, endcol))

            # The actual pixels in that region.
            patch = image[region]

            # Always normalize patch that goes into the netowrk for getting a prediction score 
            patch = zero_pad(patch, stride, stride)


            return patch, rowstart, colstart   
        
        
class CellTypeViewer(object):
    
    def __init__(self, viewer, image, celltype_name, key_categories, imagename, savedir, canvas, ax, figure):
        
        
           self.viewer = viewer
           self.image = image
           self.celltype_name = celltype_name
           self.imagename = imagename
           self.canvas = canvas
           self.key_categories = key_categories
           self.savedir = savedir
           self.ax = ax
           self.figure = figure
           self.plot()
    
    def plot(self):
        
        self.ax.cla()
        
        for (celltype_name,event_label) in self.key_categories.items():
                        if event_label > 0 and self.celltype_name == celltype_name:
                             csvname = self.savedir + "/" + celltype_name + "Location" + (os.path.splitext(os.path.basename(self.imagename))[0] + '.csv')
                             event_locations, size_locations, angle_locations, timelist, eventlist = self.event_counter(csvname)
                             
                             self.viewer.add_image(self.image, name='Image')
                             for layer in list(self.viewer.layers):
                                     if celltype_name in layer.name or layer.name in celltype_name:
                                            self.viewer.layers.remove(layer)
                             self.viewer.add_points(np.asarray(event_locations), size = size_locations ,name = celltype_name, face_color = [0]*4, edge_color = "red", edge_width = 1)
                             self.viewer.theme = 'light'
                             self.ax.plot(timelist, eventlist, '-r')
                             self.ax.set_title(celltype_name + "Instances")
                             self.ax.set_xlabel("Time")
                             self.ax.set_ylabel("Counts")
                             self.figure.canvas.draw()
                             self.figure.canvas.flush_events()
                             plt.savefig(self.savedir  + celltype_name   + '.png') 
                             
    def event_counter(self, csv_file):
     
         time,y,x,score,size,confidence  = np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)
         
         
         eventcounter = 0
         eventlist = []
         timelist = []   
         listtime = time.tolist()
         listy = y.tolist()
         listx = x.tolist()
         listsize = size.tolist()
         
         event_locations = []
         size_locations = []
         
         for i in range(len(listtime)):
             tcenter = listtime[i] 
             ycenter = listy[i]
             xcenter = listx[i]
             size = listsize[i]
             eventcounter = listtime.count(tcenter)
             timelist.append(tcenter)
             eventlist.append(eventcounter)
             
             event_locations.append([tcenter, ycenter, xcenter])   
             size_locations.append(size)
             
            
         return event_locations, size_locations, timelist, eventlist                                 
    