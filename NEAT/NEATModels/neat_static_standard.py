#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:13:01 2020

@author: aimachine
"""

from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from keras import callbacks
import os
from NEATModels import nets
from NEATModels.loss import static_yolo_loss, yolo_loss_v1, yolo_loss_v0
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
rom pathlib import Path


class NEATStaticDetection(object):
    

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
    
    
    def __init__(self, staticconfig, NpzDirectory,KeyCatagories, KeyCord, TrainModelName, ValidationModelName, model_dir, model_name,  show = False ):

        self.NpzDirectory = NpzDirectory
        self.TrainModelName = TrainModelName
        self.ValidationModelName = ValidationModelName
        self.model_dir = model_dir
        self.model_name = model_name
        self.KeyCatagories = KeyCatagories
        self.box_vector = staticconfig.box_vector
        self.model_weights = None
        self.show = show
        self.simple = staticconfig.simple
        self.categories = len(KeyCatagories)
        self.depth = staticconfig.depth
        self.start_kernel = staticconfig.start_kernel
        self.mid_kernel = staticconfig.mid_kernel
        self.learning_rate = staticconfig.learning_rate
        self.epochs = staticconfig.epochs
        self.residual = staticconfig.residual
        self.startfilter = staticconfig.startfilter
        self.batch_size = staticconfig.batch_size
        self.multievent = staticconfig.multievent
        self.ImageX = staticconfig.ImageX
        self.ImageY = staticconfig.ImageY
        self.nboxes = staticconfig.nboxes
        self.gridX = staticconfig.gridX
        self.gridY = staticconfig.gridY
        self.yoloV0 = yoloV0
        self.last_activation = None
        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
    
    
   
        

    def loadData(self):
        
        (X,Y),  axes = helpers.load_full_training_data(self.NpzDirectory, self.TrainModelName, verbose= True)

        (X_val,Y_val), axes = helpers.load_full_training_data(self.NpzDirectory, self.ValidationModelName,  verbose= True)
        
        
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
        
       
        Y_rest = self.Y[:,:,:,self.categories:]
        Y_main = self.Y[:,:,:,0:self.categories-1]
 
        y_integers = np.argmax(Y_main, axis = -1)
        y_integers = y_integers[:,0,0]

        
        
        d_class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = d_class_weights.reshape(1,d_class_weights.shape[0])
        
        
        
        if self.residual:
            model_keras = nets.resnet_v2
        else:
            model_keras = nets.seqnet_v2
            
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
           
        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):
        
            self.model_weights = model_weights
            print('loading weights')
        else:
           
            self.model_weights = None
        
        self.Trainingmodel = model_keras(input_shape, self.categories,   box_vector = self.box_vector , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, startfilter = self.startfilter,last_activation = self.last_activation,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer=sgd, loss = yolo_loss_v0(self.categories, self.gridX, self.gridY, self.nboxes, self.box_vector, self.entropy), metrics=['accuracy'])
        self.Trainingmodel.summary()
        
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotStaticHistory(self.Trainingmodel, self.X_val, self.Y_val, self.KeyCatagories, self.KeyCord, self.gridX, self.gridY, plot = self.show, nboxes = self.nboxes)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, class_weight = d_class_weights , batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])
        #clear_output(wait=True) 

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
        
    def plot_prediction(self, idx):
        
        helpers.Printpredict(idx, self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name)

    

    
        
