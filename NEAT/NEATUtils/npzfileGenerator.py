import numpy as np
from .helpers import normalizeFloatZeroOne
from glob import glob
from tifffile import imread
from tqdm import tqdm
import collections
from sklearn.model_selection import train_test_split
import os

def generate_training_data(Masteroutputdir, Masterlabel,SaveNpzDirectory, SaveName, SaveNameVal, starttime, endtime, TrainshapeX, TrainshapeY):
    
                assert len(Masteroutputdir) == len(Masterlabel) 
                
                axes = 'STXYC'
                data = []
                label = []   

                
                for i in range(0,len(Masteroutputdir)):

                       outputdir =  Masteroutputdir[i]
                       print(outputdir)
                       Images = sorted(glob(outputdir + '/' +'*.tif'))
                       Images = list(map(imread, Images))
                       #Normalize everything before it goes inside the training
                       NormalizeImages = [normalizeFloatZeroOne(image.astype('uint16') ,1,99.8) for image in tqdm(Images)]
                    
                       for n in NormalizeImages:
                      
                           blankX = n[starttime:endtime,:TrainshapeX,:TrainshapeY]
                          

                           blankY = Masterlabel[i]
                            
                           blankY = np.expand_dims(blankY, -1)
                           blankX = np.expand_dims(blankX, -1)
                            
                           data.append(blankX)
                           label.append(blankY)
                       
            
                          
                          
                dataarr = np.array(data)
                labelarr = np.array(label)
                print(dataarr.shape, labelarr.shape)
                traindata, validdata, trainlabel, validlabel = train_test_split(dataarr, labelarr, train_size=0.95,test_size=0.05, shuffle= True)
                save_full_training_data(SaveNpzDirectory, SaveName, traindata, trainlabel, axes)
                save_full_training_data(SaveNpzDirectory, SaveNameVal, validdata, validlabel, axes)
    
def generate_training_data_V2(Masteroutputdir, Mode,  SaveNpzDirectory, SaveName, SaveNameVal, starttime, endtime, TrainshapeX, TrainshapeY):
    
                
                axes = 'STXYC'
                data = []
                label = []   

                
                for i in range(0,len(Masteroutputdir)):

                       outputdir =  Masteroutputdir[i]
                       print(outputdir)
                       
                       Name =  sorted(glob(outputdir + '/' +'*.tif'))
                       
                       #Normalize everything before it goes inside the training
                       
                    
                       for j in tqdm(range(0, len(Name))):
                           
                           image = imread(Name[j]) 
                           image = normalizeFloatZeroOne(image ,1,99.8)
                            
                           blankX = image[starttime:endtime,:TrainshapeX,:TrainshapeY]
                           NameImage = os.path.basename(os.path.splitext(Name[j])[0])
                           
                           CsvFile = outputdir  + NameImage + Mode + '.csv' 
                           blankY = np.loadtxt(CsvFile, unpack = True)
                            
                           blankY = np.expand_dims(blankY, -1)
                           blankX = np.expand_dims(blankX, -1)
                            
                           data.append(blankX)
                           label.append(blankY)
                       
            
                          
                          
                dataarr = np.array(data)
                labelarr = np.array(label)
                print(dataarr.shape, labelarr.shape)
                traindata, validdata, trainlabel, validlabel = train_test_split(dataarr, labelarr, train_size=0.95,test_size=0.05, shuffle= True)
                save_full_training_data(SaveNpzDirectory, SaveName, traindata, trainlabel, axes)
                save_full_training_data(SaveNpzDirectory, SaveNameVal, validdata, validlabel, axes)
    


def generate_2D_training_data_V2(Masteroutputdir, Mode,  SaveNpzDirectory, SaveName, SaveNameVal, TrainshapeX, TrainshapeY):
    
                
                axes = 'SXYC'
                data = []
                label = []   

                
                for i in range(0,len(Masteroutputdir)):

                       outputdir =  Masteroutputdir[i]
                       print(outputdir)
                       
                       Name =  sorted(glob(outputdir + '/' +'*.tif'))
                       
                       #Normalize everything before it goes inside the training
                       
                    
                       for j in tqdm(range(0, len(Name))):
                           
                           image = imread(Name[j]) 
                           image = normalizeFloatZeroOne(image ,1,99.8)
                            
                           blankX = image[:TrainshapeX,:TrainshapeY]
                           NameImage = os.path.basename(os.path.splitext(Name[j])[0])
                           
                           CsvFile = outputdir  + NameImage + Mode + '.csv' 
                           blankY = np.loadtxt(CsvFile, unpack = True)
                            
                           blankY = np.expand_dims(blankY, -1)
                           blankX = np.expand_dims(blankX, -1)
                            
                           data.append(blankX)
                           label.append(blankY)
                       
            
                          
                          
                dataarr = np.array(data)
                labelarr = np.array(label)
                print(dataarr.shape, labelarr.shape)
                traindata, validdata, trainlabel, validlabel = train_test_split(dataarr, labelarr, train_size=0.95,test_size=0.05, shuffle= True)
                save_full_training_data(SaveNpzDirectory, SaveName, traindata, trainlabel, axes)
                save_full_training_data(SaveNpzDirectory, SaveNameVal, validdata, validlabel, axes)
    


 
def generate_2D_training_data(Masteroutputdir, Masterlabel,SaveNpzDirectory, SaveName, SaveNameVal, TrainshapeX, TrainshapeY):
    
                assert len(Masteroutputdir) == len(Masterlabel) 
                
                axes = 'SXYC'
                data = []
                label = []   

                
                for i in range(0,len(Masteroutputdir)):

                       outputdir =  Masteroutputdir[i]
                      
                       Images = sorted(glob(outputdir + '/' +'*.tif'))
                       Images = list(map(imread, Images))
                       #Normalize everything before it goes inside the training
                       NormalizeImages = [normalizeFloatZeroOne(image.astype('uint16'),1,99.8) for image in tqdm(Images)]
                    
                    
                       for n in NormalizeImages:
                      
                         blankX = n
                         if blankX.ndim == 2: 
                          if blankX.shape[0] == TrainshapeX and blankX.shape[1] == TrainshapeY: 

                           blankY = Masterlabel[i]
                            
                           blankY = np.expand_dims(blankY, -1)
                           blankX = np.expand_dims(blankX, -1)
                            
                           data.append(blankX)
                           label.append(blankY)
                       
            
                          else : 
                              print(blankX.shape,blankY.shape, len(data), len(label))
                          
                dataarr = np.asarray(data)
                labelarr = np.asarray(label)
                print(dataarr.shape, labelarr.shape)
                traindata, validdata, trainlabel, validlabel = train_test_split(dataarr, labelarr, train_size=0.95,test_size=0.05, shuffle= True)
                save_full_training_data(SaveNpzDirectory, SaveName, traindata, trainlabel, axes)
                save_full_training_data(SaveNpzDirectory, SaveNameVal, validdata, validlabel, axes)
    
                
                
                
def _raise(e):
    raise e

               
def save_training_data(directory, filename, data, label, sublabel, axes):
    """Save training data in ``.npz`` format."""
  
    
  
    len(axes) == data.ndim or _raise(ValueError())
    np.savez(directory + filename, data = data, label = label, label2 = sublabel, axes = axes)
    
    
def save_full_training_data(directory, filename, data, label, axes):
    """Save training data in ``.npz`` format."""
  

    len(axes) == data.ndim or _raise(ValueError())
    np.savez(directory + filename, data = data, label = label, axes = axes)     
    
    
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)



           
