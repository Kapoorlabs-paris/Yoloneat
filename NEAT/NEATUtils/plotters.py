import keras
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
"""
@author: Varun Kapoor
"""

class PlotHistory(keras.callbacks.Callback):
    
    
    def __init__(self, Trainingmodel, X, Y, KeyCatagories, gridX, gridY, plot = False, nboxes = 1):
       self.Trainingmodel = Trainingmodel 
       self.X = X
       self.Y = Y
       self.plot = plot
       self.gridX = gridX
       self.gridY = gridY
       self.nboxes = nboxes
      
       self.KeyCatagories = KeyCatagories
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
         self.fig = plt.figure()
        
        self.logs = []
       
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
      
        self.i += 1
        if self.plot:
         f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        
        
         ax1.set_yscale('log')
         ax1.plot(self.x, self.losses, label="loss")
         ax1.plot(self.x, self.val_losses, label="val_loss")
         ax1.legend()
        
         ax2.plot(self.x, self.acc, label="acc")
         ax2.plot(self.x, self.val_acc, label="val_acc")
         ax2.legend()
         plt.show()
         #clear_output(True)
        idx = random.randint(1,self.X.shape[0] - 1)
        Printpredict(idx,self.Trainingmodel, self.X, self.Y, self.KeyCatagories, self.gridX, self.gridY, plot = self.plot, nboxes = self.nboxes)
        
def Printpredict(idx, model, data, Truelabel, KeyCatagories, gridX, gridY, plot = False, nboxes = 1):

    
    Image = data[idx]
    Truelabel = Truelabel[idx]
    data = np.expand_dims(Image, 0)
    prediction = model.predict(data)
    prediction = model.predict(data)
   
    cols = 5
    
    if plot:  
          import matplotlib.pyplot as plt  
          fig, ax = plt.subplots(1,data.shape[1],figsize=(5*cols,5))
          fig.figsize=(20,10)
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N T H W C) C is 1 in our case
    for j in range(0,data.shape[1]):
            
            img = Image[j,:,:,0]
            if plot:
              ax[j].imshow(img, cm.Spectral)
    print('Prediction :', prediction[0,:,:,0:len(KeyCategories) + len(KeyCord) ]) 
            
    print('True Label : ', Truelabel)

    if plot:
              plt.show()     
        

class PlotStaticHistory(keras.callbacks.Callback):
    
    
    def __init__(self, Trainingmodel, X, Y, KeyCatagories, KeyCord, gridX, gridY, plot = False, nboxes = 1):
       self.Trainingmodel = Trainingmodel 
       self.X = X
       self.Y = Y
       self.gridX = gridX
       self.gridY = gridY
       self.plot = plot
       self.nboxes = nboxes
       self.KeyCord = KeyCord
       self.KeyCatagories = KeyCatagories
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
         self.fig = plt.figure()
        
        self.logs = []
       
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        if self.plot:
         f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        
        
         ax1.set_yscale('log')
         ax1.plot(self.x, self.losses, label="loss")
         ax1.plot(self.x, self.val_losses, label="val_loss")
         ax1.legend()
         ax2.plot(self.x, self.acc, label="acc")
         ax2.plot(self.x, self.val_acc, label="val_acc")
         ax2.legend()
         plt.show()
         #clear_output(True)
        idx = random.randint(1,self.X.shape[0] - 1)
        PrintStaticpredict(idx,self.Trainingmodel, self.X, self.Y, self.KeyCatagories, self.KeyCord, self.gridX, self.gridY, plot = self.plot, nboxes = self.nboxes)
        
def PrintStaticpredict(idx, model, data, Truelabel, KeyCategories, KeyCord, gridX, gridY, plot = False, nboxes = 1):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    data = np.expand_dims(Image, 0)
    prediction = model.predict(data)
   
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N H W C) C is 1 in our case
            
    img = Image[:,:,0]
    if plot:
        plt.imshow(img, cm.Spectral)
    for i in range(0, prediction.shape[0]):
        maxevent = np.argmax(prediction[i,:,:,:len(KeyCategories)], axis = -1)
        trueevent = np.argmax(Truelabel[0,0,:len(KeyCategories)], axis = -1)
        for (k,v) in KeyCategories.items(): 
           if v == maxevent: 
               maxlabel =  k
           if v == trueevent:
               truelabel = k
        
        print( "Predicted cell:", maxlabel , "Probability:", prediction[i,0,0,maxevent])
        print('True Cell type:', truelabel)
        for b in range(1,nboxes - 1):
                prediction[i,:,:,len(KeyCategories):len(KeyCategories) + len(KeyCord)] += prediction[i,:,:,len(KeyCategories) + b*len(KeyCord):len(KeyCategories) + (b + 1)*len(KeyCord) ] 
        prediction[i,:,:,len(KeyCategories):len(KeyCategories) + len(KeyCord)] = prediction[i,:,:,len(KeyCategories):len(KeyCategories) + len(KeyCord)] / nboxes        
        for (k,v) in KeyCord.items():
            
            print(k, prediction[i,:,:,len(KeyCategories) + v])
            print('True positional value', k, Truelabel[0,0,len(KeyCategories) + v])

    if plot:
              plt.show()          
